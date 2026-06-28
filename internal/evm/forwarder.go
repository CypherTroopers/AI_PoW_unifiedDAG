package evm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type Forwarder interface {
	Execute(context.Context, string, protocol.EVMRequest) (protocol.EVMReceipt, error)
}

type Client struct {
	URL        string
	HTTPClient *http.Client
}

func NewClient(url string) *Client {
	return &Client{
		URL:        strings.TrimRight(strings.TrimSpace(url), "/"),
		HTTPClient: &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *Client) Execute(ctx context.Context, parentHash string, request protocol.EVMRequest) (protocol.EVMReceipt, error) {
	request, err := normalizeRequest(request)
	if err != nil {
		return protocol.EVMReceipt{}, err
	}
	if c.URL == "" {
		return executeMock(parentHash, request), nil
	}

	payload, err := json.Marshal(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      uint64          `json:"id"`
		Method  string          `json:"method"`
		Params  json.RawMessage `json:"params"`
	}{"2.0", 1, request.Method, request.Params})
	if err != nil {
		return protocol.EVMReceipt{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL, bytes.NewReader(payload))
	if err != nil {
		return protocol.EVMReceipt{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return protocol.EVMReceipt{}, fmt.Errorf("EVM upstream request: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return protocol.EVMReceipt{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return protocol.EVMReceipt{}, fmt.Errorf("EVM upstream status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var rpcResp struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &rpcResp); err != nil {
		return protocol.EVMReceipt{}, fmt.Errorf("decode EVM response: %w", err)
	}
	if rpcResp.Error != nil {
		return protocol.EVMReceipt{}, fmt.Errorf("EVM JSON-RPC error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}
	if len(rpcResp.Result) == 0 {
		return protocol.EVMReceipt{}, errors.New("EVM response has no result")
	}
	return makeReceipt(parentHash, request, rpcResp.Result, c.URL, false), nil
}

func VerifyReceipt(parentHash string, request protocol.EVMRequest, receipt protocol.EVMReceipt) error {
	request, err := normalizeRequest(request)
	if err != nil {
		return err
	}
	if receipt.Method != request.Method {
		return errors.New("EVM receipt method mismatch")
	}
	want := makeReceipt(parentHash, request, receipt.Response, receipt.Upstream, receipt.Mock)
	if want.RequestHash != receipt.RequestHash || want.ResponseHash != receipt.ResponseHash || want.StateRoot != receipt.StateRoot {
		return errors.New("EVM receipt commitment mismatch")
	}
	return nil
}

func executeMock(parentHash string, request protocol.EVMRequest) protocol.EVMReceipt {
	result, _ := json.Marshal(struct {
		Accepted bool   `json:"accepted"`
		ChainID  string `json:"chainId"`
		Method   string `json:"method"`
	}{true, "0x7a69", request.Method})
	return makeReceipt(parentHash, request, result, "mock://deterministic-evm", true)
}

func makeReceipt(parentHash string, request protocol.EVMRequest, response json.RawMessage, upstream string, mock bool) protocol.EVMReceipt {
	requestHash := protocol.HashJSON(request)
	responseHash := protocol.HashBytes(response)
	stateRoot := protocol.HashJSON(struct {
		Domain       string `json:"domain"`
		ParentHash   string `json:"parentHash"`
		RequestHash  string `json:"requestHash"`
		ResponseHash string `json:"responseHash"`
	}{"AIDCHAIN_EVM_RECEIPT_V1", parentHash, requestHash, responseHash})
	return protocol.EVMReceipt{
		Method:       request.Method,
		RequestHash:  requestHash,
		Response:     append(json.RawMessage(nil), response...),
		ResponseHash: responseHash,
		StateRoot:    stateRoot,
		Upstream:     upstream,
		Mock:         mock,
	}
}

func normalizeRequest(request protocol.EVMRequest) (protocol.EVMRequest, error) {
	request.Method = strings.TrimSpace(request.Method)
	if request.Method == "" {
		request.Method = "eth_chainId"
	}
	if len(request.Params) == 0 {
		request.Params = json.RawMessage("[]")
	}
	if !json.Valid(request.Params) {
		return protocol.EVMRequest{}, errors.New("EVM params must be valid JSON")
	}
	return request, nil
}
