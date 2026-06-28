package engine

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

func TestMockBuildValidateFinalize(t *testing.T) {
	mock := NewMock(ForkOsaka, protocol.GenesisHash)
	envelope, err := mock.Build(context.Background(), BuildInput{
		Timestamp: 100, PrevRandao: protocol.HashBytes([]byte("randao")),
		SuggestedFeeRecipient: "0x" + strings.Repeat("1", 40), ParentBeaconBlockRoot: protocol.HashBytes([]byte("beacon")),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := mock.Validate(context.Background(), envelope); err != nil {
		t.Fatal(err)
	}
	tampered := envelope
	tampered.StateRoot = protocol.HashBytes([]byte("tampered"))
	if err := mock.Validate(context.Background(), tampered); err == nil {
		t.Fatal("tampered execution state root was accepted")
	}
	if err := mock.Finalize(context.Background(), envelope); err != nil {
		t.Fatal(err)
	}
	if mock.Head() != envelope.BlockHash {
		t.Fatal("mock execution head did not advance")
	}
}

func TestHTTPClientUsesVersionedEngineMethodsAndJWT(t *testing.T) {
	for _, fork := range []Fork{ForkOsaka, ForkAmsterdam} {
		t.Run(string(fork), func(t *testing.T) {
			secret := bytes32(0x42)
			var mu sync.Mutex
			var methods []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !validJWT(r.Header.Get("Authorization"), secret) {
					t.Error("invalid Engine API JWT")
				}
				var request struct {
					JSONRPC string            `json:"jsonrpc"`
					ID      uint64            `json:"id"`
					Method  string            `json:"method"`
					Params  []json.RawMessage `json:"params"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					return
				}
				mu.Lock()
				methods = append(methods, request.Method)
				mu.Unlock()
				var result any
				switch {
				case strings.HasPrefix(request.Method, "engine_forkchoiceUpdated"):
					payloadID := "0x0102030405060708"
					result = map[string]any{"payloadStatus": map[string]any{"status": "VALID", "latestValidHash": protocol.GenesisHash}, "payloadId": payloadID}
				case strings.HasPrefix(request.Method, "engine_getPayload"):
					payload := executionPayloadFixture(fork)
					result = map[string]any{"executionPayload": payload, "blockValue": "0x0", "executionRequests": []string{}}
				case strings.HasPrefix(request.Method, "engine_newPayload"):
					result = map[string]any{"status": "VALID", "latestValidHash": protocol.HashBytes([]byte("block"))}
				default:
					t.Errorf("unexpected method %s", request.Method)
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result})
			}))
			defer server.Close()

			client, err := NewClient(ClientConfig{URL: server.URL, Fork: fork, GenesisHash: protocol.GenesisHash, JWTSecret: secret})
			if err != nil {
				t.Fatal(err)
			}
			envelope, err := client.Build(context.Background(), BuildInput{
				Timestamp: 100, PrevRandao: protocol.HashBytes([]byte("randao")),
				SuggestedFeeRecipient: "0x" + strings.Repeat("1", 40), ParentBeaconBlockRoot: protocol.HashBytes([]byte("beacon")),
				SlotNumber: 1, TargetGasLimit: 30_000_000,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := client.Validate(context.Background(), envelope); err != nil {
				t.Fatal(err)
			}
			if err := client.Finalize(context.Background(), envelope); err != nil {
				t.Fatal(err)
			}
			want := []string{"engine_forkchoiceUpdatedV3", "engine_getPayloadV5", "engine_newPayloadV4", "engine_newPayloadV4", "engine_forkchoiceUpdatedV3"}
			if fork == ForkAmsterdam {
				want = []string{"engine_forkchoiceUpdatedV4", "engine_getPayloadV6", "engine_newPayloadV5", "engine_newPayloadV5", "engine_forkchoiceUpdatedV4"}
			}
			if strings.Join(methods, ",") != strings.Join(want, ",") {
				t.Fatalf("method sequence mismatch: got=%v want=%v", methods, want)
			}
		})
	}
}

func executionPayloadFixture(fork Fork) map[string]any {
	payload := map[string]any{
		"parentHash": protocol.GenesisHash, "feeRecipient": "0x" + strings.Repeat("1", 40),
		"stateRoot": protocol.HashBytes([]byte("state")), "receiptsRoot": protocol.HashBytes([]byte("receipts")),
		"logsBloom": "0x" + strings.Repeat("0", 512), "prevRandao": protocol.HashBytes([]byte("randao")),
		"blockNumber": "0x1", "gasLimit": "0x1c9c380", "gasUsed": "0x0", "timestamp": "0x64",
		"extraData": "0x", "baseFeePerGas": "0x1", "blockHash": protocol.HashBytes([]byte("block")),
		"transactions": []string{}, "withdrawals": []any{}, "blobGasUsed": "0x0", "excessBlobGas": "0x0",
	}
	if fork == ForkAmsterdam {
		payload["blockAccessList"] = "0xc0"
		payload["slotNumber"] = "0x1"
	}
	return payload
}

func validJWT(authorization string, secret []byte) bool {
	token := strings.TrimPrefix(authorization, "Bearer ")
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return false
	}
	mac := hmac.New(sha256.New, secret)
	_, _ = mac.Write([]byte(parts[0] + "." + parts[1]))
	want := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(want), []byte(parts[2])) {
		return false
	}
	claims, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return false
	}
	var payload struct {
		IssuedAt int64 `json:"iat"`
	}
	return json.Unmarshal(claims, &payload) == nil && time.Since(time.Unix(payload.IssuedAt, 0)) < time.Minute
}

func bytes32(value byte) []byte {
	out := make([]byte, 32)
	for i := range out {
		out[i] = value
	}
	return out
}
