package engine

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type Fork string

const (
	ForkOsaka     Fork = "osaka"
	ForkAmsterdam Fork = "amsterdam"
)

type BuildInput struct {
	Timestamp             uint64
	PrevRandao            string
	SuggestedFeeRecipient string
	ParentBeaconBlockRoot string
	SlotNumber            uint64
	TargetGasLimit        uint64
}

type Engine interface {
	Build(context.Context, BuildInput) (protocol.ExecutionPayloadEnvelope, error)
	Validate(context.Context, protocol.ExecutionPayloadEnvelope) error
	Finalize(context.Context, protocol.ExecutionPayloadEnvelope) error
	Head() string
}

func ValidateEnvelope(envelope protocol.ExecutionPayloadEnvelope) error {
	if envelope.Fork != string(ForkOsaka) && envelope.Fork != string(ForkAmsterdam) {
		return fmt.Errorf("unsupported execution fork %q", envelope.Fork)
	}
	if len(envelope.Payload) == 0 || !isHash32(envelope.BlockHash) || !isHash32(envelope.ParentHash) || !isHash32(envelope.StateRoot) {
		return errors.New("incomplete execution payload envelope")
	}
	if !isHash32(envelope.ParentBeaconBlockRoot) {
		return errors.New("invalid parent beacon block root")
	}
	for _, hash := range envelope.ExpectedBlobVersionedHashes {
		if !isHash32(hash) {
			return errors.New("invalid expected blob versioned hash")
		}
	}
	if _, err := parseQuantity(envelope.BlockNumber); err != nil {
		return fmt.Errorf("invalid execution block number: %w", err)
	}
	if _, err := parseQuantity(envelope.Timestamp); err != nil {
		return fmt.Errorf("invalid execution timestamp: %w", err)
	}
	return nil
}

func isHash32(value string) bool {
	value = strings.TrimPrefix(value, "0x")
	if len(value) != 64 {
		return false
	}
	_, err := strconv.ParseUint(value[:16], 16, 64)
	if err != nil {
		return false
	}
	for _, char := range value[16:] {
		if !((char >= '0' && char <= '9') || (char >= 'a' && char <= 'f') || (char >= 'A' && char <= 'F')) {
			return false
		}
	}
	return true
}

func quantity(value uint64) string { return fmt.Sprintf("0x%x", value) }

func parseQuantity(value string) (uint64, error) {
	value = strings.TrimPrefix(value, "0x")
	if value == "" {
		return 0, errors.New("empty quantity")
	}
	return strconv.ParseUint(value, 16, 64)
}

func zeroHash() string { return "0x" + strings.Repeat("0", 64) }

func zeroAddress() string { return "0x" + strings.Repeat("0", 40) }

func normalizeHash(value string) string {
	if value == "" {
		return zeroHash()
	}
	if !strings.HasPrefix(value, "0x") {
		return "0x" + value
	}
	return strings.ToLower(value)
}
