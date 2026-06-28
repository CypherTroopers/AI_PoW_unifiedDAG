package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type Mock struct {
	mu     sync.Mutex
	fork   Fork
	head   string
	height uint64
}

func NewMock(fork Fork, genesisHash string) *Mock {
	if fork == "" {
		fork = ForkOsaka
	}
	return &Mock{fork: fork, head: normalizeHash(genesisHash)}
}

func (m *Mock) Head() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.head
}

func (m *Mock) Build(_ context.Context, input BuildInput) (protocol.ExecutionPayloadEnvelope, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if input.Timestamp == 0 {
		return protocol.ExecutionPayloadEnvelope{}, errors.New("execution timestamp is required")
	}
	feeRecipient := input.SuggestedFeeRecipient
	if feeRecipient == "" {
		feeRecipient = zeroAddress()
	}
	prevRandao := normalizeHash(input.PrevRandao)
	beaconRoot := normalizeHash(input.ParentBeaconBlockRoot)
	stateRoot := protocol.HashJSON(struct {
		Domain    string `json:"domain"`
		Parent    string `json:"parent"`
		Timestamp uint64 `json:"timestamp"`
		Height    uint64 `json:"height"`
	}{"AIDCHAIN_MOCK_EXECUTION_STATE_V1", m.head, input.Timestamp, m.height + 1})
	payload := map[string]any{
		"parentHash": m.head, "feeRecipient": feeRecipient, "stateRoot": stateRoot,
		"receiptsRoot": protocol.HashBytes([]byte("mock-receipts" + stateRoot)), "logsBloom": "0x" + strings.Repeat("0", 512),
		"prevRandao": prevRandao, "blockNumber": quantity(m.height + 1), "gasLimit": quantity(max(input.TargetGasLimit, 30_000_000)),
		"gasUsed": "0x0", "timestamp": quantity(input.Timestamp), "extraData": "0x", "baseFeePerGas": "0x1",
		"blockHash": zeroHash(), "transactions": []string{}, "withdrawals": []any{}, "blobGasUsed": "0x0", "excessBlobGas": "0x0",
	}
	if m.fork == ForkAmsterdam {
		payload["blockAccessList"] = "0xc0"
		payload["slotNumber"] = quantity(input.SlotNumber)
	}
	payload["blockHash"] = protocol.HashJSON(payload)
	raw, _ := json.Marshal(payload)
	blockHash := payload["blockHash"].(string)
	return protocol.ExecutionPayloadEnvelope{
		Fork: string(m.fork), Payload: raw, BlockHash: blockHash, ParentHash: m.head, StateRoot: stateRoot,
		BlockNumber: quantity(m.height + 1), Timestamp: quantity(input.Timestamp), ParentBeaconBlockRoot: beaconRoot,
		ExpectedBlobVersionedHashes: []string{}, ExecutionRequests: []string{}, BlockValue: "0x0",
	}, nil
}

func (m *Mock) Validate(_ context.Context, envelope protocol.ExecutionPayloadEnvelope) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.validateLocked(envelope)
}

func (m *Mock) validateLocked(envelope protocol.ExecutionPayloadEnvelope) error {
	if err := ValidateEnvelope(envelope); err != nil {
		return err
	}
	if envelope.Fork != string(m.fork) || !strings.EqualFold(envelope.ParentHash, m.head) {
		return errors.New("mock execution payload does not extend current head")
	}
	var payload map[string]any
	if err := json.Unmarshal(envelope.Payload, &payload); err != nil {
		return err
	}
	got, _ := payload["blockHash"].(string)
	payload["blockHash"] = zeroHash()
	want := protocol.HashJSON(payload)
	if !strings.EqualFold(got, want) || !strings.EqualFold(got, envelope.BlockHash) {
		return fmt.Errorf("mock execution block hash mismatch: got=%s want=%s", got, want)
	}
	if state, _ := payload["stateRoot"].(string); !strings.EqualFold(state, envelope.StateRoot) {
		return errors.New("mock execution state root mismatch")
	}
	return nil
}

func (m *Mock) Finalize(_ context.Context, envelope protocol.ExecutionPayloadEnvelope) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if strings.EqualFold(envelope.BlockHash, m.head) {
		return nil
	}
	if err := m.validateLocked(envelope); err != nil {
		return err
	}
	number, err := parseQuantity(envelope.BlockNumber)
	if err != nil {
		return err
	}
	m.head = envelope.BlockHash
	m.height = number
	return nil
}
