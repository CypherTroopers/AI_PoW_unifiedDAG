package node

import (
	"crypto/ed25519"
	"encoding/base64"
	"testing"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aipow"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

func TestResultCertificateRequiresQuorumAndValidSignatures(t *testing.T) {
	ids := []string{"node0", "node1", "node2", "node3"}
	validators, keys, err := DevnetValidatorSet(ids)
	if err != nil {
		t.Fatal(err)
	}
	n, err := New(Config{ID: "node0", PrivateKey: keys["node0"], Validators: validators})
	if err != nil {
		t.Fatal(err)
	}
	job := protocol.NewAIJob("model", "input")
	result, err := (aipow.DummyRunner{}).Run(t.Context(), job)
	if err != nil {
		t.Fatal(err)
	}
	resultHash := protocol.HashJSON(result)
	var tickets []protocol.PoWTicket
	certificate := protocol.ResultCertificate{ResultHash: resultHash}
	for _, id := range ids[:3] {
		tickets = append(tickets, protocol.PoWTicket{MinerID: id})
		attestation := protocol.InferenceAttestation{
			WorkerID: id, Height: 1, ParentHash: protocol.GenesisHash,
			JobID: job.ID, ResultHash: resultHash, ModelHash: result.ModelHash,
			InputHash: result.InputHash, OutputHash: result.OutputHash,
		}
		attestation.Signature = base64.StdEncoding.EncodeToString(ed25519.Sign(keys[id], protocol.InferenceSigningBytes(attestation)))
		certificate.Attestations = append(certificate.Attestations, attestation)
	}
	if err := n.validateResultCertificate(1, protocol.GenesisHash, job, result, tickets, certificate, nil); err != nil {
		t.Fatalf("valid result QC rejected: %v", err)
	}
	minority := certificate
	minority.Attestations = append([]protocol.InferenceAttestation(nil), certificate.Attestations[:2]...)
	if err := n.validateResultCertificate(1, protocol.GenesisHash, job, result, tickets[:2], minority, nil); err == nil {
		t.Fatal("two-of-four result attestations were accepted")
	}
	forged := certificate
	forged.Attestations = append([]protocol.InferenceAttestation(nil), certificate.Attestations...)
	forged.Attestations[0].OutputHash = protocol.HashBytes([]byte("forged"))
	if err := n.validateResultCertificate(1, protocol.GenesisHash, job, result, tickets, forged, nil); err == nil {
		t.Fatal("forged result attestation was accepted")
	}
	if err := n.validateResultCertificate(2, protocol.HashBytes([]byte("new-parent")), job, result, tickets, certificate, nil); err == nil {
		t.Fatal("result certificate was replayed into another height")
	}
}
