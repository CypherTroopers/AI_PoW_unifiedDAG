package aipow

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type DummyRunner struct {
	ManifestRoot string
}

func (r DummyRunner) Run(_ context.Context, job protocol.AIJob) (protocol.AIResult, error) {
	if job.ID == "" || job.Model == "" || job.Prompt == "" {
		return protocol.AIResult{}, errors.New("job id, model, and prompt are required")
	}

	modelHash := protocol.HashBytes([]byte(job.Model))
	inputHash := protocol.HashBytes([]byte(job.Prompt))
	seed := protocol.HashJSON(struct {
		Domain    string `json:"domain"`
		ModelHash string `json:"modelHash"`
		InputHash string `json:"inputHash"`
	}{"AIDCHAIN_DUMMY_INFERENCE_V1", modelHash, inputHash})
	output := "dummy-ai:" + seed[2:34]
	outputHash := protocol.HashBytes([]byte(output))
	aiDigest := protocol.HashJSON(struct {
		Domain       string `json:"domain"`
		JobID        string `json:"jobId"`
		ModelHash    string `json:"modelHash"`
		InputHash    string `json:"inputHash"`
		OutputHash   string `json:"outputHash"`
		ManifestRoot string `json:"manifestRoot"`
	}{"AIDCHAIN_AI_RECEIPT_V1", job.ID, modelHash, inputHash, outputHash, r.ManifestRoot})

	return protocol.AIResult{
		JobID:       job.ID,
		ModelHash:   modelHash,
		InputHash:   inputHash,
		Output:      output,
		OutputHash:  outputHash,
		AIDigest:    aiDigest,
		ProofScheme: "dummy-deterministic-v1",
	}, nil
}

func (r DummyRunner) Verify(ctx context.Context, job protocol.AIJob, result protocol.AIResult) error {
	want, err := r.Run(ctx, job)
	if err != nil {
		return err
	}
	want.ProofScheme = ""
	got := result
	got.ProofScheme = ""
	if protocol.HashJSON(want) != protocol.HashJSON(got) {
		return errors.New("dummy AI result does not match deterministic execution")
	}
	return nil
}

func Mine(ctx context.Context, ticket protocol.PoWTicket, maxNonce uint64) (protocol.PoWTicket, error) {
	if ticket.Height == 0 || ticket.ParentHash == "" || ticket.JobID == "" || ticket.MinerID == "" {
		return protocol.PoWTicket{}, errors.New("incomplete PoW ticket challenge")
	}
	if ticket.Difficulty > 255 {
		return protocol.PoWTicket{}, errors.New("difficulty must be <= 255")
	}

	for nonce := uint64(0); nonce <= maxNonce; nonce++ {
		if nonce&1023 == 0 {
			select {
			case <-ctx.Done():
				return protocol.PoWTicket{}, ctx.Err()
			default:
			}
		}
		ticket.Nonce = nonce
		ticket.WorkHash = CalculateWorkHash(ticket)
		if MeetsDifficulty(ticket.WorkHash, ticket.Difficulty) {
			return ticket, nil
		}
	}
	return protocol.PoWTicket{}, fmt.Errorf("no nonce found within %d attempts", maxNonce+1)
}

func CalculateWorkHash(ticket protocol.PoWTicket) string {
	ticket.WorkHash = ""
	ticket.Signature = ""
	return protocol.HashJSON(struct {
		Domain string             `json:"domain"`
		Ticket protocol.PoWTicket `json:"ticket"`
	}{"AIDCHAIN_AI_POW_V1", ticket})
}

func Verify(ticket protocol.PoWTicket) error {
	if CalculateWorkHash(ticket) != ticket.WorkHash {
		return errors.New("PoW work hash mismatch")
	}
	if !MeetsDifficulty(ticket.WorkHash, ticket.Difficulty) {
		return errors.New("PoW work hash does not meet difficulty")
	}
	return nil
}

func MeetsDifficulty(hashHex string, difficulty uint8) bool {
	raw := hashHex
	if len(raw) >= 2 && raw[:2] == "0x" {
		raw = raw[2:]
	}
	b, err := hex.DecodeString(raw)
	if err != nil || len(b) != sha256.Size {
		return false
	}
	remaining := int(difficulty)
	for _, v := range b {
		if remaining == 0 {
			return true
		}
		if remaining >= 8 {
			if v != 0 {
				return false
			}
			remaining -= 8
			continue
		}
		mask := byte(0xff << (8 - remaining))
		return v&mask == 0
	}
	return remaining == 0
}

func SelectWinner(tickets []protocol.PoWTicket) (protocol.PoWTicket, error) {
	if len(tickets) == 0 {
		return protocol.PoWTicket{}, errors.New("no PoW tickets")
	}
	ordered := append([]protocol.PoWTicket(nil), tickets...)
	for _, ticket := range ordered {
		if err := Verify(ticket); err != nil {
			return protocol.PoWTicket{}, fmt.Errorf("invalid ticket from %s: %w", ticket.MinerID, err)
		}
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].WorkHash == ordered[j].WorkHash {
			return ordered[i].MinerID < ordered[j].MinerID
		}
		return ordered[i].WorkHash < ordered[j].WorkHash
	})
	return ordered[0], nil
}
