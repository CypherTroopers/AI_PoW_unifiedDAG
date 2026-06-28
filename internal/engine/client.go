package engine

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type ClientConfig struct {
	URL         string
	Fork        Fork
	GenesisHash string
	JWTSecret   []byte
	Timeout     time.Duration
}

type Client struct {
	url        string
	fork       Fork
	jwtSecret  []byte
	httpClient *http.Client
	id         atomic.Uint64

	mu   sync.Mutex
	head string
}

func NewClient(cfg ClientConfig) (*Client, error) {
	if strings.TrimSpace(cfg.URL) == "" {
		return nil, errors.New("Engine API URL is required")
	}
	if cfg.Fork == "" {
		cfg.Fork = ForkOsaka
	}
	if cfg.Fork != ForkOsaka && cfg.Fork != ForkAmsterdam {
		return nil, fmt.Errorf("unsupported Engine API fork %q", cfg.Fork)
	}
	if len(cfg.JWTSecret) != 32 {
		return nil, errors.New("Engine API JWT secret must be exactly 32 bytes")
	}
	if !isHash32(normalizeHash(cfg.GenesisHash)) {
		return nil, errors.New("valid execution genesis/head hash is required")
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 10 * time.Second
	}
	return &Client{
		url: strings.TrimRight(cfg.URL, "/"), fork: cfg.Fork, jwtSecret: append([]byte(nil), cfg.JWTSecret...),
		httpClient: &http.Client{Timeout: cfg.Timeout}, head: normalizeHash(cfg.GenesisHash),
	}, nil
}

func LoadJWTSecret(pathOrHex string) ([]byte, error) {
	value := strings.TrimSpace(pathOrHex)
	if value == "" {
		return nil, errors.New("JWT secret is required")
	}
	if data, err := os.ReadFile(value); err == nil {
		value = strings.TrimSpace(string(data))
	}
	value = strings.TrimPrefix(value, "0x")
	secret, err := hex.DecodeString(value)
	if err != nil || len(secret) != 32 {
		return nil, errors.New("JWT secret must be 32-byte hex or a file containing it")
	}
	return secret, nil
}

func (c *Client) Head() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.head
}

func (c *Client) Build(ctx context.Context, input BuildInput) (protocol.ExecutionPayloadEnvelope, error) {
	c.mu.Lock()
	head := c.head
	c.mu.Unlock()
	state := forkchoiceState{HeadBlockHash: head, SafeBlockHash: head, FinalizedBlockHash: head}
	attributes := payloadAttributes{
		Timestamp: quantity(input.Timestamp), PrevRandao: normalizeHash(input.PrevRandao),
		SuggestedFeeRecipient: input.SuggestedFeeRecipient, Withdrawals: []any{},
		ParentBeaconBlockRoot: normalizeHash(input.ParentBeaconBlockRoot),
	}
	method := "engine_forkchoiceUpdatedV3"
	params := []any{state, attributes}
	if c.fork == ForkAmsterdam {
		method = "engine_forkchoiceUpdatedV4"
		attributes.SlotNumber = quantity(input.SlotNumber)
		attributes.TargetGasLimit = quantity(input.TargetGasLimit)
		params = []any{state, attributes, nil}
	}
	var update forkchoiceResponse
	if err := c.call(ctx, method, params, &update); err != nil {
		return protocol.ExecutionPayloadEnvelope{}, err
	}
	if err := requireValidStatus(update.PayloadStatus); err != nil {
		return protocol.ExecutionPayloadEnvelope{}, err
	}
	if update.PayloadID == nil || *update.PayloadID == "" {
		return protocol.ExecutionPayloadEnvelope{}, errors.New("Engine API returned no payload id")
	}
	getMethod := "engine_getPayloadV5"
	if c.fork == ForkAmsterdam {
		getMethod = "engine_getPayloadV6"
	}
	var built getPayloadResponse
	if err := c.call(ctx, getMethod, []any{*update.PayloadID}, &built); err != nil {
		return protocol.ExecutionPayloadEnvelope{}, err
	}
	return envelopeFromPayload(c.fork, built.ExecutionPayload, attributes.ParentBeaconBlockRoot, built.ExecutionRequests, built.BlockValue)
}

func (c *Client) Validate(ctx context.Context, envelope protocol.ExecutionPayloadEnvelope) error {
	if err := ValidateEnvelope(envelope); err != nil {
		return err
	}
	method := "engine_newPayloadV4"
	if c.fork == ForkAmsterdam {
		method = "engine_newPayloadV5"
	}
	params := []any{json.RawMessage(envelope.Payload), envelope.ExpectedBlobVersionedHashes, envelope.ParentBeaconBlockRoot, envelope.ExecutionRequests}
	var status payloadStatus
	if err := c.call(ctx, method, params, &status); err != nil {
		return err
	}
	return requireValidStatus(status)
}

func (c *Client) Finalize(ctx context.Context, envelope protocol.ExecutionPayloadEnvelope) error {
	if strings.EqualFold(c.Head(), envelope.BlockHash) {
		return nil
	}
	if err := c.Validate(ctx, envelope); err != nil {
		return err
	}
	state := forkchoiceState{HeadBlockHash: envelope.BlockHash, SafeBlockHash: envelope.BlockHash, FinalizedBlockHash: envelope.BlockHash}
	method := "engine_forkchoiceUpdatedV3"
	params := []any{state, nil}
	if c.fork == ForkAmsterdam {
		method = "engine_forkchoiceUpdatedV4"
		params = []any{state, nil, nil}
	}
	var update forkchoiceResponse
	if err := c.call(ctx, method, params, &update); err != nil {
		return err
	}
	if err := requireValidStatus(update.PayloadStatus); err != nil {
		return err
	}
	c.mu.Lock()
	c.head = envelope.BlockHash
	c.mu.Unlock()
	return nil
}

type forkchoiceState struct {
	HeadBlockHash      string `json:"headBlockHash"`
	SafeBlockHash      string `json:"safeBlockHash"`
	FinalizedBlockHash string `json:"finalizedBlockHash"`
}

type payloadAttributes struct {
	Timestamp             string `json:"timestamp"`
	PrevRandao            string `json:"prevRandao"`
	SuggestedFeeRecipient string `json:"suggestedFeeRecipient"`
	Withdrawals           []any  `json:"withdrawals"`
	ParentBeaconBlockRoot string `json:"parentBeaconBlockRoot"`
	SlotNumber            string `json:"slotNumber,omitempty"`
	TargetGasLimit        string `json:"targetGasLimit,omitempty"`
}

type payloadStatus struct {
	Status          string  `json:"status"`
	LatestValidHash *string `json:"latestValidHash"`
	ValidationError *string `json:"validationError"`
}

type forkchoiceResponse struct {
	PayloadStatus payloadStatus `json:"payloadStatus"`
	PayloadID     *string       `json:"payloadId"`
}

type getPayloadResponse struct {
	ExecutionPayload  json.RawMessage `json:"executionPayload"`
	BlockValue        string          `json:"blockValue"`
	ExecutionRequests []string        `json:"executionRequests"`
}

func envelopeFromPayload(fork Fork, payload json.RawMessage, beaconRoot string, requests []string, blockValue string) (protocol.ExecutionPayloadEnvelope, error) {
	var view struct {
		ParentHash  string `json:"parentHash"`
		StateRoot   string `json:"stateRoot"`
		BlockNumber string `json:"blockNumber"`
		Timestamp   string `json:"timestamp"`
		BlockHash   string `json:"blockHash"`
		BlobGasUsed string `json:"blobGasUsed"`
	}
	if err := json.Unmarshal(payload, &view); err != nil {
		return protocol.ExecutionPayloadEnvelope{}, err
	}
	if view.BlobGasUsed != "" && view.BlobGasUsed != "0x0" {
		return protocol.ExecutionPayloadEnvelope{}, errors.New("blob transactions require consensus-layer versioned hashes; unsupported in this devnet")
	}
	envelope := protocol.ExecutionPayloadEnvelope{
		Fork: string(fork), Payload: append(json.RawMessage(nil), payload...), BlockHash: view.BlockHash,
		ParentHash: view.ParentHash, StateRoot: view.StateRoot, BlockNumber: view.BlockNumber, Timestamp: view.Timestamp,
		ParentBeaconBlockRoot: beaconRoot, ExpectedBlobVersionedHashes: []string{}, ExecutionRequests: requests, BlockValue: blockValue,
	}
	return envelope, ValidateEnvelope(envelope)
}

func requireValidStatus(status payloadStatus) error {
	if strings.ToUpper(status.Status) != "VALID" {
		message := status.Status
		if status.ValidationError != nil {
			message += ": " + *status.ValidationError
		}
		return fmt.Errorf("execution payload status is not VALID: %s", message)
	}
	return nil
}

func (c *Client) call(ctx context.Context, method string, params any, result any) error {
	id := c.id.Add(1)
	payload, err := json.Marshal(struct {
		JSONRPC string `json:"jsonrpc"`
		ID      uint64 `json:"id"`
		Method  string `json:"method"`
		Params  any    `json:"params"`
	}{"2.0", id, method, params})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.jwt())
	response, err := c.httpClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("Engine API HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	var rpcResponse struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &rpcResponse); err != nil {
		return err
	}
	if rpcResponse.Error != nil {
		return fmt.Errorf("Engine API error %d: %s", rpcResponse.Error.Code, rpcResponse.Error.Message)
	}
	if len(rpcResponse.Result) == 0 {
		return errors.New("Engine API response has no result")
	}
	return json.Unmarshal(rpcResponse.Result, result)
}

func (c *Client) jwt() string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	claims, _ := json.Marshal(struct {
		IssuedAt int64 `json:"iat"`
	}{time.Now().Unix()})
	payload := base64.RawURLEncoding.EncodeToString(claims)
	unsigned := header + "." + payload
	mac := hmac.New(sha256.New, c.jwtSecret)
	_, _ = mac.Write([]byte(unsigned))
	return unsigned + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}
