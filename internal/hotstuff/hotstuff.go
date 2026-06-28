package hotstuff

import (
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type ValidatorSet struct {
	keys map[string]ed25519.PublicKey
	ids  []string
}

func NewValidatorSet(keys map[string]ed25519.PublicKey) (*ValidatorSet, error) {
	if len(keys) < 4 {
		return nil, errors.New("HotStuff devnet requires at least four validators")
	}
	set := &ValidatorSet{keys: make(map[string]ed25519.PublicKey, len(keys))}
	for id, key := range keys {
		if id == "" || len(key) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("invalid validator %q", id)
		}
		set.keys[id] = append(ed25519.PublicKey(nil), key...)
		set.ids = append(set.ids, id)
	}
	sort.Strings(set.ids)
	return set, nil
}

func (s *ValidatorSet) IDs() []string {
	return append([]string(nil), s.ids...)
}

func (s *ValidatorSet) Len() int { return len(s.ids) }

func (s *ValidatorSet) Quorum() int { return (2*s.Len())/3 + 1 }

func (s *ValidatorSet) PublicKey(id string) (ed25519.PublicKey, bool) {
	key, ok := s.keys[id]
	return append(ed25519.PublicKey(nil), key...), ok
}

func (s *ValidatorSet) VerifyVote(vote protocol.Vote) error {
	key, ok := s.keys[vote.VoterID]
	if !ok {
		return fmt.Errorf("unknown voter %q", vote.VoterID)
	}
	sig, err := base64.StdEncoding.DecodeString(vote.Signature)
	if err != nil {
		return fmt.Errorf("decode vote signature: %w", err)
	}
	if !ed25519.Verify(key, protocol.VoteSigningBytes(vote.Phase, vote.Height, vote.BlockHash), sig) {
		return errors.New("invalid vote signature")
	}
	return nil
}

func (s *ValidatorSet) MakeQC(phase protocol.Phase, height uint64, blockHash string, votes []protocol.Vote) (protocol.QuorumCertificate, error) {
	unique := make(map[string]protocol.Vote, len(votes))
	for _, vote := range votes {
		if vote.Phase != phase || vote.Height != height || vote.BlockHash != blockHash {
			return protocol.QuorumCertificate{}, errors.New("vote does not match QC subject")
		}
		if err := s.VerifyVote(vote); err != nil {
			return protocol.QuorumCertificate{}, fmt.Errorf("vote from %s: %w", vote.VoterID, err)
		}
		if _, exists := unique[vote.VoterID]; exists {
			return protocol.QuorumCertificate{}, fmt.Errorf("duplicate vote from %s", vote.VoterID)
		}
		unique[vote.VoterID] = vote
	}
	if len(unique) < s.Quorum() {
		return protocol.QuorumCertificate{}, fmt.Errorf("not enough votes: got=%d need=%d", len(unique), s.Quorum())
	}
	canonical := make([]protocol.Vote, 0, len(unique))
	for _, vote := range unique {
		canonical = append(canonical, vote)
	}
	sort.Slice(canonical, func(i, j int) bool { return canonical[i].VoterID < canonical[j].VoterID })
	return protocol.QuorumCertificate{Phase: phase, Height: height, BlockHash: blockHash, Votes: canonical}, nil
}

func (s *ValidatorSet) VerifyQC(qc protocol.QuorumCertificate) error {
	_, err := s.MakeQC(qc.Phase, qc.Height, qc.BlockHash, qc.Votes)
	return err
}

type BlockValidator func(protocol.Block) error

type Replica struct {
	mu sync.RWMutex

	id         string
	privateKey ed25519.PrivateKey
	validators *ValidatorSet
	validate   BlockValidator

	height  uint64
	tipHash string
	blocks  []protocol.Block
	pending map[string]protocol.Block
	voted   map[uint64]map[protocol.Phase]string

	lockedHeight uint64
	lockedHash   string
}

func NewReplica(id string, privateKey ed25519.PrivateKey, validators *ValidatorSet, validate BlockValidator) (*Replica, error) {
	if _, ok := validators.keys[id]; !ok {
		return nil, fmt.Errorf("node %q is not in validator set", id)
	}
	if len(privateKey) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid Ed25519 private key")
	}
	return &Replica{
		id: id, privateKey: append(ed25519.PrivateKey(nil), privateKey...), validators: validators, validate: validate,
		tipHash: protocol.GenesisHash, pending: make(map[string]protocol.Block), voted: make(map[uint64]map[protocol.Phase]string),
	}, nil
}

func (r *Replica) HeightAndTip() (uint64, string) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.height, r.tipHash
}

func (r *Replica) Blocks() []protocol.Block {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]protocol.Block(nil), r.blocks...)
}

func (r *Replica) OnProposal(block protocol.Block) (protocol.Vote, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.validateBlockLocked(block); err != nil {
		return protocol.Vote{}, err
	}
	hash := block.Hash()
	if r.lockedHeight == block.Header.Height && r.lockedHash != "" && r.lockedHash != hash {
		return protocol.Vote{}, errors.New("proposal conflicts with locked block")
	}
	r.pending[hash] = block
	return r.voteLocked(protocol.PhasePrepare, block.Header.Height, hash)
}

func (r *Replica) OnQC(block protocol.Block, qc protocol.QuorumCertificate) (*protocol.Vote, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.validators.VerifyQC(qc); err != nil {
		return nil, false, fmt.Errorf("invalid QC: %w", err)
	}
	hash := block.Hash()
	if qc.BlockHash != hash || qc.Height != block.Header.Height {
		return nil, false, errors.New("QC does not certify supplied block")
	}
	if err := r.validateBlockLocked(block); err != nil {
		return nil, false, err
	}
	r.pending[hash] = block

	switch qc.Phase {
	case protocol.PhasePrepare:
		vote, err := r.voteLocked(protocol.PhasePreCommit, qc.Height, hash)
		return &vote, false, err
	case protocol.PhasePreCommit:
		r.lockedHeight = qc.Height
		r.lockedHash = hash
		vote, err := r.voteLocked(protocol.PhaseCommit, qc.Height, hash)
		return &vote, false, err
	case protocol.PhaseCommit:
		if r.height == block.Header.Height && r.tipHash == hash {
			return nil, true, nil
		}
		if block.Header.Height != r.height+1 || block.Header.ParentHash != r.tipHash {
			return nil, false, errors.New("commit does not extend finalized chain")
		}
		r.blocks = append(r.blocks, block)
		r.height = block.Header.Height
		r.tipHash = hash
		delete(r.pending, hash)
		return nil, true, nil
	default:
		return nil, false, fmt.Errorf("unsupported QC phase %q", qc.Phase)
	}
}

func (r *Replica) validateBlockLocked(block protocol.Block) error {
	hash := block.Hash()
	if block.Header.Height == r.height && hash == r.tipHash {
		return nil
	}
	if block.Header.Height != r.height+1 {
		return fmt.Errorf("unexpected block height: got=%d want=%d", block.Header.Height, r.height+1)
	}
	if block.Header.ParentHash != r.tipHash {
		return fmt.Errorf("parent mismatch: got=%s want=%s", block.Header.ParentHash, r.tipHash)
	}
	if r.validate != nil {
		if err := r.validate(block); err != nil {
			return fmt.Errorf("block validation: %w", err)
		}
	}
	return nil
}

func (r *Replica) voteLocked(phase protocol.Phase, height uint64, blockHash string) (protocol.Vote, error) {
	if r.voted[height] == nil {
		r.voted[height] = make(map[protocol.Phase]string)
	}
	if previous := r.voted[height][phase]; previous != "" && previous != blockHash {
		return protocol.Vote{}, fmt.Errorf("refusing double vote at height=%d phase=%s", height, phase)
	}
	r.voted[height][phase] = blockHash
	sig := ed25519.Sign(r.privateKey, protocol.VoteSigningBytes(phase, height, blockHash))
	return protocol.Vote{
		Phase: phase, Height: height, BlockHash: blockHash, VoterID: r.id,
		Signature: base64.StdEncoding.EncodeToString(sig),
	}, nil
}
