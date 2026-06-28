package node_test

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aipow"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/engine"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/miner"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/node"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/zkml"
)

func TestSeparatedMinersFinalizeAcrossFourLightValidators(t *testing.T) {
	ids := []string{"node0", "node1", "node2", "node3"}
	validatorSet, privateKeys, err := node.DevnetValidatorSet(ids)
	if err != nil {
		t.Fatal(err)
	}

	nodes := make([]*node.Node, 0, 4)
	executionEngines := make([]*countingEngine, 0, 4)
	validatorServers := make([]*httptest.Server, 0, 4)
	minerServers := make([]*httptest.Server, 0, 4)
	for _, id := range ids {
		execution := &countingEngine{Engine: engine.NewMock(engine.ForkOsaka, protocol.GenesisHash)}
		n, err := node.New(node.Config{ID: id, PrivateKey: privateKeys[id], Validators: validatorSet, Difficulty: 4, ExecutionEngine: execution})
		if err != nil {
			t.Fatal(err)
		}
		service, err := miner.New(id, privateKeys[id], 4, 100_000, "")
		if err != nil {
			t.Fatal(err)
		}
		nodes = append(nodes, n)
		executionEngines = append(executionEngines, execution)
		validatorServers = append(validatorServers, httptest.NewServer(n.Handler()))
		minerServers = append(minerServers, httptest.NewServer(service.Handler()))
	}
	for i := range validatorServers {
		defer validatorServers[i].Close()
		defer minerServers[i].Close()
	}
	peers := make(map[string]string, 4)
	miners := make(map[string]string, 4)
	for i, id := range ids {
		peers[id] = validatorServers[i].URL
		miners[id] = minerServers[i].URL
	}
	for _, n := range nodes {
		n.SetPeers(peers)
		n.SetMiners(miners)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	response, err := nodes[0].RunRound(ctx, protocol.RoundRequest{
		Job: protocol.NewAIJob("test.gguf", "separate miners from validators"),
		EVM: protocol.EVMRequest{Method: "eth_chainId", Params: json.RawMessage("[]")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.Block.Header.Height != 1 || len(response.CommitQC.Votes) < 3 {
		t.Fatalf("unexpected finalization response: height=%d votes=%d", response.Block.Header.Height, len(response.CommitQC.Votes))
	}
	if len(response.Block.ResultQC.Attestations) != 4 || response.Block.Header.ResultQCRoot != protocol.ResultQCRoot(response.Block.ResultQC) {
		t.Fatalf("result QC was not committed: attestations=%d", len(response.Block.ResultQC.Attestations))
	}
	for _, n := range nodes {
		status := n.Status()
		if status.Height != 1 || status.TipHash != response.BlockHash {
			t.Fatalf("node %s disagrees: height=%d tip=%s", status.NodeID, status.Height, status.TipHash)
		}
		blockBytes, _ := json.Marshal(response.Block)
		if object, ok := n.Object(protocol.HashBytes(blockBytes)); !ok || len(object) == 0 {
			t.Fatalf("node %s did not persist finalized block object", status.NodeID)
		}
	}
	var buildCalls int64
	for _, execution := range executionEngines {
		buildCalls += execution.builds.Load()
		if execution.validations.Load() == 0 || execution.finalizations.Load() != 1 || execution.Head() != response.Block.Execution.BlockHash {
			t.Fatalf("execution engine did not validate/finalize payload: validate=%d finalize=%d head=%s", execution.validations.Load(), execution.finalizations.Load(), execution.Head())
		}
	}
	if buildCalls != 1 {
		t.Fatalf("expected exactly one winning proposer to build payload, got %d", buildCalls)
	}
}

func TestGroth16ZKMLAndReplicatedExecutionFinalize(t *testing.T) {
	prover, verifier, err := zkml.Setup()
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{"node0", "node1", "node2", "node3"}
	validatorSet, privateKeys, err := node.DevnetValidatorSet(ids)
	if err != nil {
		t.Fatal(err)
	}
	runner := aipow.QuantizedRunner{}
	nodes := make([]*node.Node, 0, 4)
	validatorServers := make([]*httptest.Server, 0, 4)
	minerServers := make([]*httptest.Server, 0, 4)
	for _, id := range ids {
		n, err := node.New(node.Config{
			ID: id, PrivateKey: privateKeys[id], Validators: validatorSet, Difficulty: 2,
			AIRunner: runner, ZKMLVerifier: verifier,
		})
		if err != nil {
			t.Fatal(err)
		}
		service, err := miner.New(id, privateKeys[id], 2, 100_000, "")
		if err != nil {
			t.Fatal(err)
		}
		service.SetRunner(runner)
		service.SetZKMLProver(prover)
		nodes = append(nodes, n)
		validatorServers = append(validatorServers, httptest.NewServer(n.Handler()))
		minerServers = append(minerServers, httptest.NewServer(service.Handler()))
	}
	for i := range validatorServers {
		defer validatorServers[i].Close()
		defer minerServers[i].Close()
	}
	peers := make(map[string]string, 4)
	miners := make(map[string]string, 4)
	for i, id := range ids {
		peers[id] = validatorServers[i].URL
		miners[id] = minerServers[i].URL
	}
	for _, n := range nodes {
		n.SetPeers(peers)
		n.SetMiners(miners)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	response, err := nodes[0].RunRound(ctx, protocol.RoundRequest{
		Job: protocol.NewAIJob("quantized-model-v1", "prove replicated inference"),
		EVM: protocol.EVMRequest{Method: "eth_chainId", Params: json.RawMessage("[]")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.Block.Result.ProofScheme != zkml.Scheme || len(response.Block.ZKMLProofs) != 4 || len(response.Block.ResultQC.Attestations) != 4 {
		t.Fatalf("missing semantic proof/QC: scheme=%s proofs=%d attestations=%d", response.Block.Result.ProofScheme, len(response.Block.ZKMLProofs), len(response.Block.ResultQC.Attestations))
	}
	if response.Block.Header.ZKMLProofsRoot != protocol.ZKMLProofsRoot(response.Block.ZKMLProofs) {
		t.Fatal("zkML proofs are not committed by the block header")
	}
	for _, n := range nodes {
		if status := n.Status(); status.Height != 1 || status.TipHash != response.BlockHash {
			t.Fatalf("node %s rejected zkML block", status.NodeID)
		}
	}
}

type countingEngine struct {
	engine.Engine
	builds        atomic.Int64
	validations   atomic.Int64
	finalizations atomic.Int64
}

func (e *countingEngine) Build(ctx context.Context, input engine.BuildInput) (protocol.ExecutionPayloadEnvelope, error) {
	e.builds.Add(1)
	return e.Engine.Build(ctx, input)
}

func (e *countingEngine) Validate(ctx context.Context, payload protocol.ExecutionPayloadEnvelope) error {
	e.validations.Add(1)
	return e.Engine.Validate(ctx, payload)
}

func (e *countingEngine) Finalize(ctx context.Context, payload protocol.ExecutionPayloadEnvelope) error {
	e.finalizations.Add(1)
	return e.Engine.Finalize(ctx, payload)
}
