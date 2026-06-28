package aipow

import (
	"context"
	"testing"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

func TestMineVerifyAndRejectTamper(t *testing.T) {
	job := protocol.NewAIJob("model.gguf", "test prompt")
	result, err := (DummyRunner{}).Run(context.Background(), job)
	if err != nil {
		t.Fatal(err)
	}
	ticket, err := Mine(context.Background(), protocol.PoWTicket{
		Height: 1, ParentHash: protocol.GenesisHash, JobID: job.ID,
		ResultHash: protocol.HashJSON(result), AIDigest: result.AIDigest,
		MinerID: "node0", Difficulty: 6, AIProofHash: result.AIDigest,
	}, 100_000)
	if err != nil {
		t.Fatal(err)
	}
	if err := Verify(ticket); err != nil {
		t.Fatalf("valid ticket rejected: %v", err)
	}

	ticket.AIDigest = protocol.HashBytes([]byte("tampered"))
	if err := Verify(ticket); err == nil {
		t.Fatal("tampered ticket was accepted")
	}
}

func TestLeadingZeroDifficulty(t *testing.T) {
	if !MeetsDifficulty("0x00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 8) {
		t.Fatal("expected eight leading zero bits")
	}
	if MeetsDifficulty("0x01ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 8) {
		t.Fatal("accepted hash without eight leading zero bits")
	}
}
