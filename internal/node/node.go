package node

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aipow"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aiseal"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/engine"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/evm"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/hotstuff"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/storage"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/zkml"
)

type Config struct {
	ID               string
	PrivateKey       ed25519.PrivateKey
	Validators       *hotstuff.ValidatorSet
	Peers            map[string]string
	Miners           map[string]string
	Difficulty       uint8
	ManifestRoot     string
	AISealManifest   *aiseal.Manifest
	AISealTarget     string
	MinPoWSamples    int
	MinTensorSamples int
	EVMURL           string
	ExecutionEngine  engine.Engine
	ObjectStore      *storage.Store
	AIRunner         aipow.Runner
	ZKMLVerifier     *zkml.Verifier
	MinZKMLProofs    int
}

type Node struct {
	id               string
	privateKey       ed25519.PrivateKey
	validators       *hotstuff.ValidatorSet
	difficulty       uint8
	manifestRoot     string
	aiSealManifest   *aiseal.Manifest
	aiSealTarget     string
	minPoWSamples    int
	minTensorSamples int
	runner           aipow.Runner
	zkmlVerifier     *zkml.Verifier
	minZKMLProofs    int
	evm              evm.Forwarder
	execution        engine.Engine
	objectStore      *storage.Store
	replica          *hotstuff.Replica
	httpClient       *http.Client

	peersMu sync.RWMutex
	peers   map[string]string
	miners  map[string]string

	roundMu            sync.Mutex
	proposeMu          sync.Mutex
	executionMu        sync.Mutex
	validatedExecution map[string]bool
	validationMu       sync.Mutex
	validatedBlocks    map[string]bool
	validatedZKML      map[string]bool
}

func New(cfg Config) (*Node, error) {
	if cfg.ID == "" || cfg.Validators == nil {
		return nil, errors.New("node id and validator set are required")
	}
	if cfg.AISealManifest != nil {
		if err := aiseal.ValidateManifest(*cfg.AISealManifest); err != nil {
			return nil, fmt.Errorf("AISeal manifest: %w", err)
		}
		cfg.ManifestRoot = cfg.AISealManifest.ManifestRoot
	}
	if cfg.ExecutionEngine == nil {
		cfg.ExecutionEngine = engine.NewMock(engine.ForkOsaka, protocol.GenesisHash)
	}
	if cfg.ObjectStore == nil {
		cfg.ObjectStore, _ = storage.New("", 32<<20)
	}
	if cfg.AIRunner == nil {
		cfg.AIRunner = aipow.DummyRunner{ManifestRoot: cfg.ManifestRoot}
	}
	if cfg.ZKMLVerifier != nil && cfg.MinZKMLProofs == 0 {
		cfg.MinZKMLProofs = cfg.Validators.Quorum()
	}
	if cfg.MinZKMLProofs < 0 || cfg.MinZKMLProofs > cfg.Validators.Len() {
		return nil, errors.New("invalid minimum zkML proof count")
	}
	n := &Node{
		id: cfg.ID, privateKey: append(ed25519.PrivateKey(nil), cfg.PrivateKey...), validators: cfg.Validators,
		difficulty: cfg.Difficulty, manifestRoot: cfg.ManifestRoot, aiSealManifest: cfg.AISealManifest,
		aiSealTarget: cfg.AISealTarget, minPoWSamples: cfg.MinPoWSamples, minTensorSamples: cfg.MinTensorSamples,
		runner: cfg.AIRunner, zkmlVerifier: cfg.ZKMLVerifier, minZKMLProofs: cfg.MinZKMLProofs,
		evm: evm.NewClient(cfg.EVMURL), execution: cfg.ExecutionEngine,
		objectStore: cfg.ObjectStore,
		peers:       make(map[string]string), miners: make(map[string]string), httpClient: &http.Client{Timeout: 20 * time.Second},
		validatedExecution: make(map[string]bool), validatedBlocks: make(map[string]bool), validatedZKML: make(map[string]bool),
	}
	for id, url := range cfg.Peers {
		n.peers[id] = url
	}
	for id, url := range cfg.Miners {
		n.miners[id] = url
	}
	replica, err := hotstuff.NewReplica(cfg.ID, cfg.PrivateKey, cfg.Validators, n.validateBlock)
	if err != nil {
		return nil, err
	}
	n.replica = replica
	return n, nil
}

func DevnetPrivateKey(id string) ed25519.PrivateKey {
	seed := sha256.Sum256([]byte("AIDCHAIN_INSECURE_DEVNET_KEY_V1|" + id))
	return ed25519.NewKeyFromSeed(seed[:])
}

func DevnetValidatorSet(ids []string) (*hotstuff.ValidatorSet, map[string]ed25519.PrivateKey, error) {
	keys := make(map[string]ed25519.PrivateKey, len(ids))
	pubs := make(map[string]ed25519.PublicKey, len(ids))
	for _, id := range ids {
		private := DevnetPrivateKey(id)
		keys[id] = private
		pubs[id] = append(ed25519.PublicKey(nil), private.Public().(ed25519.PublicKey)...)
	}
	set, err := hotstuff.NewValidatorSet(pubs)
	return set, keys, err
}

func (n *Node) ID() string { return n.id }

func (n *Node) SetPeers(peers map[string]string) {
	n.peersMu.Lock()
	defer n.peersMu.Unlock()
	n.peers = make(map[string]string, len(peers))
	for id, url := range peers {
		n.peers[id] = url
	}
}

func (n *Node) SetMiners(miners map[string]string) {
	n.peersMu.Lock()
	defer n.peersMu.Unlock()
	n.miners = make(map[string]string, len(miners))
	for id, url := range miners {
		n.miners[id] = url
	}
}

func (n *Node) Peers() map[string]string {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()
	out := make(map[string]string, len(n.peers))
	for id, url := range n.peers {
		out[id] = url
	}
	return out
}

func (n *Node) Miners() map[string]string {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()
	out := make(map[string]string, len(n.miners))
	for id, url := range n.miners {
		out[id] = url
	}
	return out
}

func (n *Node) Status() protocol.Status {
	height, tip := n.replica.HeightAndTip()
	return protocol.Status{NodeID: n.id, Height: height, TipHash: tip, ExecutionHead: n.execution.Head(), ValidatorNum: n.validators.Len(), Quorum: n.validators.Quorum()}
}

func (n *Node) Blocks() []protocol.Block { return n.replica.Blocks() }

func (n *Node) OnProposal(block protocol.Block) (protocol.Vote, error) {
	return n.replica.OnProposal(block)
}

func (n *Node) OnQC(block protocol.Block, qc protocol.QuorumCertificate) (*protocol.Vote, bool, error) {
	if qc.Phase == protocol.PhaseCommit && !equalHash(n.execution.Head(), block.Execution.BlockHash) {
		if err := n.validators.VerifyQC(qc); err != nil {
			return nil, false, err
		}
		if err := n.validateBlock(block); err != nil {
			return nil, false, err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
		defer cancel()
		if err := n.execution.Finalize(ctx, block.Execution); err != nil {
			return nil, false, fmt.Errorf("finalize execution payload: %w", err)
		}
		blockBytes, err := json.Marshal(block)
		if err != nil {
			return nil, false, err
		}
		if _, err := n.objectStore.Put(blockBytes); err != nil {
			return nil, false, fmt.Errorf("persist finalized block object: %w", err)
		}
	}
	return n.replica.OnQC(block, qc)
}

func (n *Node) Object(id string) ([]byte, bool) { return n.objectStore.Get(id) }

func (n *Node) validateBlock(block protocol.Block) error {
	blockHash := block.Hash()
	n.validationMu.Lock()
	blockValidated := n.validatedBlocks[blockHash]
	n.validationMu.Unlock()
	if blockValidated {
		return nil
	}
	if block.Header.Version != protocol.ProtocolVersion {
		return fmt.Errorf("unsupported protocol version %d", block.Header.Version)
	}
	if err := protocol.ValidateAIJob(block.Job); err != nil {
		return err
	}
	if n.zkmlVerifier == nil {
		if err := n.runner.Verify(context.Background(), block.Job, block.Result); err != nil {
			return err
		}
	} else if err := n.validateResultCommitment(block.Job, block.Result); err != nil {
		return err
	}
	if block.Header.AIReceiptRoot != block.Result.AIDigest {
		return errors.New("AI receipt root mismatch")
	}
	timestamp, err := time.Parse(time.RFC3339Nano, block.Header.Timestamp)
	if err != nil {
		return errors.New("invalid block timestamp")
	}
	if timestamp.After(time.Now().UTC().Add(2 * time.Minute)) {
		return errors.New("block timestamp is too far in the future")
	}
	if err := evm.VerifyReceipt(block.Header.ParentHash, block.EVMRequest, block.EVM); err != nil {
		return err
	}
	if block.Header.EVMReceiptRoot != block.EVM.StateRoot {
		return errors.New("EVM receipt root mismatch")
	}
	if block.Header.EVMStateRoot != block.Execution.StateRoot || block.Header.ExecutionHash != protocol.HashJSON(block.Execution) {
		return errors.New("execution payload commitment mismatch")
	}
	if err := engine.ValidateEnvelope(block.Execution); err != nil {
		return err
	}
	if block.Execution.Timestamp != fmt.Sprintf("0x%x", timestamp.Unix()) {
		return errors.New("execution payload timestamp mismatch")
	}
	n.executionMu.Lock()
	alreadyValidated := n.validatedExecution[block.Execution.BlockHash]
	n.executionMu.Unlock()
	if !alreadyValidated {
		ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
		err := n.execution.Validate(ctx, block.Execution)
		cancel()
		if err != nil {
			return fmt.Errorf("execution payload rejected: %w", err)
		}
		n.executionMu.Lock()
		n.validatedExecution[block.Execution.BlockHash] = true
		n.executionMu.Unlock()
	}
	if block.Header.TicketsRoot != protocol.TicketsRoot(block.Tickets) {
		return errors.New("ticket root mismatch")
	}
	if block.Header.AIProofsRoot != protocol.AIProofsRoot(block.AIProofs) {
		return errors.New("AI proof root mismatch")
	}
	if block.Header.ResultQCRoot != protocol.ResultQCRoot(block.ResultQC) {
		return errors.New("result QC root mismatch")
	}
	if block.Header.ZKMLProofsRoot != protocol.ZKMLProofsRoot(block.ZKMLProofs) {
		return errors.New("zkML proof root mismatch")
	}
	if len(block.Tickets) < n.validators.Quorum() {
		return fmt.Errorf("not enough mining tickets: got=%d need=%d", len(block.Tickets), n.validators.Quorum())
	}

	winner, err := n.validateCompetition(block.Header.Height, block.Header.ParentHash, block.Job, block.Result, block.Tickets, block.AIProofs)
	if err != nil {
		return err
	}
	if err := n.validateResultCertificate(block.Header.Height, block.Header.ParentHash, block.Job, block.Result, block.Tickets, block.ResultQC, block.ZKMLProofs); err != nil {
		return err
	}
	if protocol.HashJSON(winner) != protocol.HashJSON(block.Winner) {
		return errors.New("declared PoW winner is not the best valid ticket")
	}
	if winner.MinerID != block.Header.ProposerID {
		return errors.New("block proposer is not PoW winner")
	}
	n.validationMu.Lock()
	n.validatedBlocks[blockHash] = true
	n.validationMu.Unlock()
	return nil
}

func (n *Node) validateResultCommitment(job protocol.AIJob, result protocol.AIResult) error {
	if result.JobID != job.ID || result.ModelHash != protocol.HashBytes([]byte(job.Model)) || result.InputHash != protocol.HashBytes([]byte(job.Prompt)) {
		return errors.New("AI result input commitment mismatch")
	}
	if result.Output == "" || result.OutputHash != protocol.HashBytes([]byte(result.Output)) {
		return errors.New("AI result output commitment mismatch")
	}
	wantDigest := protocol.HashJSON(struct {
		Domain       string `json:"domain"`
		JobID        string `json:"jobId"`
		ModelHash    string `json:"modelHash"`
		InputHash    string `json:"inputHash"`
		OutputHash   string `json:"outputHash"`
		ManifestRoot string `json:"manifestRoot"`
	}{"AIDCHAIN_AI_RECEIPT_V1", job.ID, result.ModelHash, result.InputHash, result.OutputHash, n.manifestRoot})
	if result.AIDigest != wantDigest {
		return errors.New("AI result digest mismatch")
	}
	if n.zkmlVerifier != nil && result.ProofScheme != zkml.Scheme {
		return errors.New("AI result does not declare the required zkML scheme")
	}
	return nil
}

func (n *Node) validateResultCertificate(height uint64, parentHash string, job protocol.AIJob, result protocol.AIResult, tickets []protocol.PoWTicket, certificate protocol.ResultCertificate, proofs []protocol.ZKMLProofBundle) error {
	resultHash := protocol.HashJSON(result)
	if certificate.ResultHash != resultHash {
		return errors.New("result certificate hash mismatch")
	}
	if len(certificate.Attestations) < n.validators.Quorum() {
		return fmt.Errorf("result attestation quorum not reached: got=%d need=%d", len(certificate.Attestations), n.validators.Quorum())
	}
	if len(certificate.Attestations) != len(tickets) {
		return errors.New("result attestation count must match ticket count")
	}
	ticketByWorker := make(map[string]protocol.PoWTicket, len(tickets))
	for _, ticket := range tickets {
		ticketByWorker[ticket.MinerID] = ticket
	}
	proofByWorker := make(map[string]protocol.ZKMLProofBundle, len(proofs))
	for _, proof := range proofs {
		if _, exists := proofByWorker[proof.WorkerID]; exists {
			return fmt.Errorf("duplicate zkML proof from %s", proof.WorkerID)
		}
		proofByWorker[proof.WorkerID] = proof
	}
	if n.zkmlVerifier == nil && len(proofs) != 0 {
		return errors.New("unexpected zkML proofs when verifier is disabled")
	}
	if n.zkmlVerifier != nil {
		if len(proofs) < n.minZKMLProofs {
			return fmt.Errorf("zkML proof quorum not reached: got=%d need=%d", len(proofs), n.minZKMLProofs)
		}
		if len(proofs) != len(certificate.Attestations) {
			return errors.New("every result attestation must carry an independent zkML proof")
		}
	}
	seen := make(map[string]bool, len(certificate.Attestations))
	wantModelCommitment, wantInputCommitment, _ := zkml.StatementFor(job.Model, job.Prompt)
	for _, attestation := range certificate.Attestations {
		if seen[attestation.WorkerID] {
			return fmt.Errorf("duplicate result attestation from %s", attestation.WorkerID)
		}
		seen[attestation.WorkerID] = true
		publicKey, ok := n.validators.PublicKey(attestation.WorkerID)
		if !ok {
			return fmt.Errorf("result attestation from non-validator %s", attestation.WorkerID)
		}
		ticket, ok := ticketByWorker[attestation.WorkerID]
		if !ok {
			return fmt.Errorf("result attestation has no ticket from %s", attestation.WorkerID)
		}
		if attestation.Height != height || attestation.ParentHash != parentHash || attestation.JobID != job.ID || attestation.ResultHash != resultHash || attestation.ModelHash != result.ModelHash || attestation.InputHash != result.InputHash || attestation.OutputHash != result.OutputHash {
			return fmt.Errorf("result attestation binding mismatch from %s", attestation.WorkerID)
		}
		signature, err := base64.StdEncoding.DecodeString(attestation.Signature)
		if err != nil || !ed25519.Verify(publicKey, protocol.InferenceSigningBytes(attestation), signature) {
			return fmt.Errorf("invalid result attestation signature from %s", attestation.WorkerID)
		}
		if ticket.ZKMLProofHash != attestation.ZKMLProofHash {
			return fmt.Errorf("ticket/attestation zkML binding mismatch from %s", attestation.WorkerID)
		}
		if n.zkmlVerifier == nil {
			if attestation.ZKMLProofHash != "" {
				return fmt.Errorf("unexpected zkML commitment from %s", attestation.WorkerID)
			}
			continue
		}
		bundle, ok := proofByWorker[attestation.WorkerID]
		if !ok {
			return fmt.Errorf("missing zkML proof from %s", attestation.WorkerID)
		}
		proof := zkml.Proof{
			Scheme: bundle.Scheme, CircuitID: bundle.CircuitID, ModelCommitment: bundle.ModelCommitment,
			InputCommitment: bundle.InputCommitment, Output: bundle.Output, Proof: bundle.Proof,
		}
		if bundle.ProofHash != zkml.ProofHash(proof) || bundle.ProofHash != attestation.ZKMLProofHash {
			return fmt.Errorf("zkML proof hash mismatch from %s", attestation.WorkerID)
		}
		if proof.ModelCommitment != wantModelCommitment || proof.InputCommitment != wantInputCommitment || result.Output != "zkml-linear-v1:"+proof.Output {
			return fmt.Errorf("zkML public statement mismatch from %s", attestation.WorkerID)
		}
		n.validationMu.Lock()
		proofValidated := n.validatedZKML[bundle.ProofHash]
		n.validationMu.Unlock()
		if !proofValidated {
			if err := n.zkmlVerifier.Verify(proof); err != nil {
				return fmt.Errorf("invalid zkML proof from %s: %w", attestation.WorkerID, err)
			}
			n.validationMu.Lock()
			n.validatedZKML[bundle.ProofHash] = true
			n.validationMu.Unlock()
		}
	}
	return nil
}

func (n *Node) validateCompetition(height uint64, parentHash string, job protocol.AIJob, result protocol.AIResult, tickets []protocol.PoWTicket, proofs []protocol.AIProofBundle) (protocol.PoWTicket, error) {
	seen := make(map[string]bool, len(tickets))
	resultHash := protocol.HashJSON(result)
	proofByMiner := make(map[string]protocol.AIProofBundle, len(proofs))
	for _, proof := range proofs {
		if _, exists := proofByMiner[proof.MinerID]; exists {
			return protocol.PoWTicket{}, fmt.Errorf("duplicate AISeal proof from %s", proof.MinerID)
		}
		proofByMiner[proof.MinerID] = proof
	}
	if n.aiSealManifest != nil && len(proofs) != len(tickets) {
		return protocol.PoWTicket{}, errors.New("AISeal proof count must match ticket count")
	}
	if n.aiSealManifest == nil && len(proofs) != 0 {
		return protocol.PoWTicket{}, errors.New("unexpected AISeal proofs in dummy mode")
	}
	for _, ticket := range tickets {
		if seen[ticket.MinerID] {
			return protocol.PoWTicket{}, fmt.Errorf("duplicate mining ticket from %s", ticket.MinerID)
		}
		seen[ticket.MinerID] = true
		publicKey, ok := n.validators.PublicKey(ticket.MinerID)
		if !ok {
			return protocol.PoWTicket{}, fmt.Errorf("ticket from non-validator %s", ticket.MinerID)
		}
		if ticket.Height != height || ticket.ParentHash != parentHash || ticket.JobID != job.ID {
			return protocol.PoWTicket{}, fmt.Errorf("ticket challenge mismatch from %s", ticket.MinerID)
		}
		if ticket.ResultHash != resultHash {
			return protocol.PoWTicket{}, fmt.Errorf("ticket AI result mismatch from %s", ticket.MinerID)
		}
		if ticket.Difficulty != n.difficulty || ticket.ManifestRoot != n.manifestRoot {
			return protocol.PoWTicket{}, fmt.Errorf("ticket AI-PoW policy mismatch from %s", ticket.MinerID)
		}
		if n.aiSealManifest != nil {
			bundle, ok := proofByMiner[ticket.MinerID]
			if !ok {
				return protocol.PoWTicket{}, fmt.Errorf("missing AISeal proof from %s", ticket.MinerID)
			}
			if bundle.ContentHash != protocol.HashBytes(bundle.Proof) || !equalHash(bundle.ProofHash, ticket.AIProofHash) {
				return protocol.PoWTicket{}, fmt.Errorf("AISeal content binding mismatch from %s", ticket.MinerID)
			}
			verification, proof, err := aiseal.VerifyBytes(*n.aiSealManifest, bundle.Proof, n.aiSealTarget, aiseal.DefaultLimits())
			if err != nil {
				return protocol.PoWTicket{}, fmt.Errorf("AISeal proof from %s: %w", ticket.MinerID, err)
			}
			if err := aiseal.VerifyExpected(proof, verification, aiseal.ExpectedChallenge{
				BlockHash: parentHash, Miner: ticket.MinerID, Epoch: height, ManifestRoot: n.manifestRoot,
				MinPoWSamples: n.minPoWSamples, MinTensorSamples: n.minTensorSamples,
			}); err != nil {
				return protocol.PoWTicket{}, err
			}
			if !equalHash(verification.ProofHash, ticket.AIProofHash) || !equalHash(verification.AIDigest, ticket.AIDigest) {
				return protocol.PoWTicket{}, fmt.Errorf("AISeal ticket binding mismatch from %s", ticket.MinerID)
			}
		} else if ticket.AIDigest != result.AIDigest || ticket.AIProofHash != result.AIDigest {
			return protocol.PoWTicket{}, fmt.Errorf("dummy AI proof mismatch from %s", ticket.MinerID)
		}
		if err := aipow.Verify(ticket); err != nil {
			return protocol.PoWTicket{}, err
		}
		signature, err := base64.StdEncoding.DecodeString(ticket.Signature)
		if err != nil || !ed25519.Verify(publicKey, protocol.TicketSigningBytes(ticket), signature) {
			return protocol.PoWTicket{}, fmt.Errorf("invalid ticket signature from %s", ticket.MinerID)
		}
	}
	return aipow.SelectWinner(tickets)
}

func equalHash(a, b string) bool {
	return strings.EqualFold(strings.TrimPrefix(a, "0x"), strings.TrimPrefix(b, "0x"))
}

func sortedPeerIDs(peers map[string]string) []string {
	ids := make([]string, 0, len(peers))
	for id := range peers {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
