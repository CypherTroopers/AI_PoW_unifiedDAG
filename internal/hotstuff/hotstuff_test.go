package hotstuff

import (
	"crypto/ed25519"
	"crypto/sha256"
	"testing"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

func TestFourReplicaThreePhaseCommit(t *testing.T) {
	ids := []string{"node0", "node1", "node2", "node3"}
	privateKeys := make(map[string]ed25519.PrivateKey)
	publicKeys := make(map[string]ed25519.PublicKey)
	for _, id := range ids {
		seed := sha256.Sum256([]byte(id))
		privateKeys[id] = ed25519.NewKeyFromSeed(seed[:])
		publicKeys[id] = privateKeys[id].Public().(ed25519.PublicKey)
	}
	set, err := NewValidatorSet(publicKeys)
	if err != nil {
		t.Fatal(err)
	}
	replicas := make([]*Replica, 0, len(ids))
	for _, id := range ids {
		replica, err := NewReplica(id, privateKeys[id], set, nil)
		if err != nil {
			t.Fatal(err)
		}
		replicas = append(replicas, replica)
	}

	block := protocol.Block{Header: protocol.BlockHeader{Version: 1, Height: 1, ParentHash: protocol.GenesisHash, ProposerID: "node0"}}
	hash := block.Hash()
	prepareVotes := make([]protocol.Vote, 0, 4)
	for _, replica := range replicas {
		vote, err := replica.OnProposal(block)
		if err != nil {
			t.Fatal(err)
		}
		prepareVotes = append(prepareVotes, vote)
	}
	prepareQC, err := set.MakeQC(protocol.PhasePrepare, 1, hash, prepareVotes[:3])
	if err != nil {
		t.Fatal(err)
	}

	preCommitVotes := advanceVotes(t, replicas, block, prepareQC, protocol.PhasePreCommit)
	preCommitQC, err := set.MakeQC(protocol.PhasePreCommit, 1, hash, preCommitVotes[:3])
	if err != nil {
		t.Fatal(err)
	}
	commitVotes := advanceVotes(t, replicas, block, preCommitQC, protocol.PhaseCommit)
	commitQC, err := set.MakeQC(protocol.PhaseCommit, 1, hash, commitVotes[:3])
	if err != nil {
		t.Fatal(err)
	}
	for _, replica := range replicas {
		vote, finalized, err := replica.OnQC(block, commitQC)
		if err != nil || vote != nil || !finalized {
			t.Fatalf("commit failed: vote=%v finalized=%v err=%v", vote, finalized, err)
		}
		height, tip := replica.HeightAndTip()
		if height != 1 || tip != hash {
			t.Fatalf("replica did not finalize: height=%d tip=%s", height, tip)
		}
	}
}

func TestReplicaRefusesPrepareDoubleVote(t *testing.T) {
	keys := make(map[string]ed25519.PrivateKey)
	pubs := make(map[string]ed25519.PublicKey)
	for _, id := range []string{"node0", "node1", "node2", "node3"} {
		seed := sha256.Sum256([]byte("double-vote-" + id))
		keys[id] = ed25519.NewKeyFromSeed(seed[:])
		pubs[id] = keys[id].Public().(ed25519.PublicKey)
	}
	set, _ := NewValidatorSet(pubs)
	replica, _ := NewReplica("node0", keys["node0"], set, nil)
	first := protocol.Block{Header: protocol.BlockHeader{Height: 1, ParentHash: protocol.GenesisHash, ProposerID: "node0"}}
	second := protocol.Block{Header: protocol.BlockHeader{Height: 1, ParentHash: protocol.GenesisHash, ProposerID: "node1"}}
	if _, err := replica.OnProposal(first); err != nil {
		t.Fatal(err)
	}
	if _, err := replica.OnProposal(second); err == nil {
		t.Fatal("replica signed conflicting prepare votes")
	}
}

func advanceVotes(t *testing.T, replicas []*Replica, block protocol.Block, qc protocol.QuorumCertificate, phase protocol.Phase) []protocol.Vote {
	t.Helper()
	votes := make([]protocol.Vote, 0, len(replicas))
	for _, replica := range replicas {
		vote, finalized, err := replica.OnQC(block, qc)
		if err != nil || finalized || vote == nil || vote.Phase != phase {
			t.Fatalf("advance to %s failed: vote=%v finalized=%v err=%v", phase, vote, finalized, err)
		}
		votes = append(votes, *vote)
	}
	return votes
}
