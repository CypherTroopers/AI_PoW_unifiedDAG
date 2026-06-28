package protocol

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"time"
)

const (
	ProtocolVersion = uint32(2)
	GenesisHash     = "0x4a1ae0a4b9f7a0c663e89b9df38b1bb2148b0977eacff33a4a9a320c4ca63b6a"
)

type AIJob struct {
	ID        string `json:"id"`
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	CreatedAt string `json:"createdAt"`
}

type AIResult struct {
	JobID       string `json:"jobId"`
	ModelHash   string `json:"modelHash"`
	InputHash   string `json:"inputHash"`
	Output      string `json:"output"`
	OutputHash  string `json:"outputHash"`
	AIDigest    string `json:"aiDigest"`
	ProofScheme string `json:"proofScheme"`
}

type EVMRequest struct {
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

type EVMReceipt struct {
	Method       string          `json:"method"`
	RequestHash  string          `json:"requestHash"`
	Response     json.RawMessage `json:"response"`
	ResponseHash string          `json:"responseHash"`
	StateRoot    string          `json:"stateRoot"`
	Upstream     string          `json:"upstream"`
	Mock         bool            `json:"mock"`
}

type ExecutionPayloadEnvelope struct {
	Fork                        string          `json:"fork"`
	Payload                     json.RawMessage `json:"executionPayload"`
	BlockHash                   string          `json:"blockHash"`
	ParentHash                  string          `json:"parentHash"`
	StateRoot                   string          `json:"stateRoot"`
	BlockNumber                 string          `json:"blockNumber"`
	Timestamp                   string          `json:"timestamp"`
	ParentBeaconBlockRoot       string          `json:"parentBeaconBlockRoot"`
	ExpectedBlobVersionedHashes []string        `json:"expectedBlobVersionedHashes"`
	ExecutionRequests           []string        `json:"executionRequests"`
	BlockValue                  string          `json:"blockValue,omitempty"`
}

type PoWTicket struct {
	Height        uint64 `json:"height"`
	ParentHash    string `json:"parentHash"`
	JobID         string `json:"jobId"`
	ResultHash    string `json:"resultHash"`
	AIDigest      string `json:"aiDigest"`
	MinerID       string `json:"minerId"`
	Nonce         uint64 `json:"nonce"`
	Difficulty    uint8  `json:"difficulty"`
	WorkHash      string `json:"workHash"`
	Signature     string `json:"signature"`
	ManifestRoot  string `json:"manifestRoot,omitempty"`
	AIProofHash   string `json:"aiProofHash,omitempty"`
	ZKMLProofHash string `json:"zkmlProofHash,omitempty"`
}

type AIProofBundle struct {
	MinerID     string          `json:"minerId"`
	ProofHash   string          `json:"proofHash"`
	ContentHash string          `json:"contentHash"`
	Proof       json.RawMessage `json:"proof"`
}

type ZKMLProofBundle struct {
	WorkerID        string `json:"workerId"`
	Scheme          string `json:"scheme"`
	CircuitID       string `json:"circuitId"`
	ModelCommitment string `json:"modelCommitment"`
	InputCommitment string `json:"inputCommitment"`
	Output          string `json:"output"`
	Proof           []byte `json:"proof"`
	ProofHash       string `json:"proofHash"`
}

type InferenceAttestation struct {
	WorkerID      string `json:"workerId"`
	Height        uint64 `json:"height"`
	ParentHash    string `json:"parentHash"`
	JobID         string `json:"jobId"`
	ResultHash    string `json:"resultHash"`
	ModelHash     string `json:"modelHash"`
	InputHash     string `json:"inputHash"`
	OutputHash    string `json:"outputHash"`
	ZKMLProofHash string `json:"zkmlProofHash,omitempty"`
	Signature     string `json:"signature"`
}

type ResultCertificate struct {
	ResultHash   string                 `json:"resultHash"`
	Attestations []InferenceAttestation `json:"attestations"`
}

type BlockHeader struct {
	Version        uint32 `json:"version"`
	Height         uint64 `json:"height"`
	ParentHash     string `json:"parentHash"`
	ProposerID     string `json:"proposerId"`
	Timestamp      string `json:"timestamp"`
	AIReceiptRoot  string `json:"aiReceiptRoot"`
	EVMStateRoot   string `json:"evmStateRoot"`
	EVMReceiptRoot string `json:"evmReceiptRoot"`
	ExecutionHash  string `json:"executionHash"`
	TicketsRoot    string `json:"ticketsRoot"`
	AIProofsRoot   string `json:"aiProofsRoot"`
	ResultQCRoot   string `json:"resultQcRoot"`
	ZKMLProofsRoot string `json:"zkmlProofsRoot"`
}

type Block struct {
	Header     BlockHeader              `json:"header"`
	Job        AIJob                    `json:"job"`
	Result     AIResult                 `json:"result"`
	EVMRequest EVMRequest               `json:"evmRequest"`
	EVM        EVMReceipt               `json:"evm"`
	Execution  ExecutionPayloadEnvelope `json:"execution"`
	Winner     PoWTicket                `json:"winner"`
	Tickets    []PoWTicket              `json:"tickets"`
	AIProofs   []AIProofBundle          `json:"aiProofs,omitempty"`
	ResultQC   ResultCertificate        `json:"resultQC"`
	ZKMLProofs []ZKMLProofBundle        `json:"zkmlProofs,omitempty"`
}

type Phase string

const (
	PhasePrepare   Phase = "prepare"
	PhasePreCommit Phase = "precommit"
	PhaseCommit    Phase = "commit"
)

type Vote struct {
	Phase     Phase  `json:"phase"`
	Height    uint64 `json:"height"`
	BlockHash string `json:"blockHash"`
	VoterID   string `json:"voterId"`
	Signature string `json:"signature"`
}

type QuorumCertificate struct {
	Phase     Phase  `json:"phase"`
	Height    uint64 `json:"height"`
	BlockHash string `json:"blockHash"`
	Votes     []Vote `json:"votes"`
}

type MiningRequest struct {
	Height     uint64 `json:"height"`
	ParentHash string `json:"parentHash"`
	Job        AIJob  `json:"job"`
}

type MiningResponse struct {
	Ticket      PoWTicket            `json:"ticket"`
	Result      AIResult             `json:"result"`
	Attestation InferenceAttestation `json:"attestation"`
	AIProof     *AIProofBundle       `json:"aiProof,omitempty"`
	ZKMLProof   *ZKMLProofBundle     `json:"zkmlProof,omitempty"`
}

type ProposeRequest struct {
	Job        AIJob             `json:"job"`
	Result     AIResult          `json:"result"`
	EVM        EVMRequest        `json:"evm"`
	Tickets    []PoWTicket       `json:"tickets"`
	AIProofs   []AIProofBundle   `json:"aiProofs,omitempty"`
	ResultQC   ResultCertificate `json:"resultQC"`
	ZKMLProofs []ZKMLProofBundle `json:"zkmlProofs,omitempty"`
}

type ProposalRequest struct {
	Block Block `json:"block"`
}

type QCRequest struct {
	Block Block             `json:"block"`
	QC    QuorumCertificate `json:"qc"`
}

type RoundRequest struct {
	Job AIJob      `json:"job"`
	EVM EVMRequest `json:"evm"`
}

type RoundResponse struct {
	Block     Block             `json:"block"`
	BlockHash string            `json:"blockHash"`
	CommitQC  QuorumCertificate `json:"commitQC"`
}

type Status struct {
	NodeID        string `json:"nodeId"`
	Height        uint64 `json:"height"`
	TipHash       string `json:"tipHash"`
	ExecutionHead string `json:"executionHead"`
	ValidatorNum  int    `json:"validatorCount"`
	Quorum        int    `json:"quorum"`
}

func NewAIJob(model, prompt string) AIJob {
	created := time.Now().UTC().Format(time.RFC3339Nano)
	j := AIJob{Model: model, Prompt: prompt, CreatedAt: created}
	j.ID = HashJSON(struct {
		Model     string `json:"model"`
		Prompt    string `json:"prompt"`
		CreatedAt string `json:"createdAt"`
	}{model, prompt, created})
	return j
}

func ValidateAIJob(job AIJob) error {
	want := HashJSON(struct {
		Model     string `json:"model"`
		Prompt    string `json:"prompt"`
		CreatedAt string `json:"createdAt"`
	}{job.Model, job.Prompt, job.CreatedAt})
	if job.ID != want {
		return fmt.Errorf("AI job id mismatch: got=%s want=%s", job.ID, want)
	}
	return nil
}

func HashBytes(data []byte) string {
	h := sha256.Sum256(data)
	return "0x" + hex.EncodeToString(h[:])
}

func HashJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("hash JSON: %v", err))
	}
	return HashBytes(b)
}

func (b Block) Hash() string {
	return HashJSON(b)
}

func TicketsRoot(tickets []PoWTicket) string {
	ordered := append([]PoWTicket(nil), tickets...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].MinerID == ordered[j].MinerID {
			return ordered[i].WorkHash < ordered[j].WorkHash
		}
		return ordered[i].MinerID < ordered[j].MinerID
	})
	return HashJSON(ordered)
}

func AIProofsRoot(proofs []AIProofBundle) string {
	ordered := append([]AIProofBundle(nil), proofs...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].MinerID == ordered[j].MinerID {
			return ordered[i].ContentHash < ordered[j].ContentHash
		}
		return ordered[i].MinerID < ordered[j].MinerID
	})
	return HashJSON(ordered)
}

func ZKMLProofsRoot(proofs []ZKMLProofBundle) string {
	ordered := append([]ZKMLProofBundle(nil), proofs...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].WorkerID == ordered[j].WorkerID {
			return ordered[i].ProofHash < ordered[j].ProofHash
		}
		return ordered[i].WorkerID < ordered[j].WorkerID
	})
	return HashJSON(ordered)
}

func ResultQCRoot(certificate ResultCertificate) string {
	ordered := certificate
	ordered.Attestations = append([]InferenceAttestation(nil), certificate.Attestations...)
	sort.Slice(ordered.Attestations, func(i, j int) bool {
		return ordered.Attestations[i].WorkerID < ordered.Attestations[j].WorkerID
	})
	return HashJSON(ordered)
}

func VoteSigningBytes(phase Phase, height uint64, blockHash string) []byte {
	b, _ := json.Marshal(struct {
		Domain    string `json:"domain"`
		Phase     Phase  `json:"phase"`
		Height    uint64 `json:"height"`
		BlockHash string `json:"blockHash"`
	}{"AIDCHAIN_HOTSTUFF_VOTE_V1", phase, height, blockHash})
	return b
}

func TicketSigningBytes(ticket PoWTicket) []byte {
	return []byte(ticket.WorkHash)
}

func InferenceSigningBytes(attestation InferenceAttestation) []byte {
	return mustJSON(struct {
		Domain        string `json:"domain"`
		WorkerID      string `json:"workerId"`
		Height        uint64 `json:"height"`
		ParentHash    string `json:"parentHash"`
		JobID         string `json:"jobId"`
		ResultHash    string `json:"resultHash"`
		ModelHash     string `json:"modelHash"`
		InputHash     string `json:"inputHash"`
		OutputHash    string `json:"outputHash"`
		ZKMLProofHash string `json:"zkmlProofHash,omitempty"`
	}{
		Domain: "AIDCHAIN_INFERENCE_ATTESTATION_V1", WorkerID: attestation.WorkerID,
		Height: attestation.Height, ParentHash: attestation.ParentHash,
		JobID: attestation.JobID, ResultHash: attestation.ResultHash, ModelHash: attestation.ModelHash,
		InputHash: attestation.InputHash, OutputHash: attestation.OutputHash, ZKMLProofHash: attestation.ZKMLProofHash,
	})
}

func mustJSON(value any) []byte {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("marshal canonical protocol value: %v", err))
	}
	return encoded
}
