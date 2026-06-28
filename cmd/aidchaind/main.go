package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aipow"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aiseal"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/engine"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/miner"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/node"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/storage"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/zkml"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return errors.New("command is required")
	}
	switch args[0] {
	case "demo":
		return runDemo(args[1:])
	case "node":
		return runNode(args[1:])
	case "miner":
		return runMiner(args[1:])
	case "submit":
		return runSubmit(args[1:])
	case "zkml-setup":
		return runZKMLSetup(args[1:])
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		usage()
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `AI-PoW + HotStuff + EVM forwarding development node

Usage:
  aidchaind demo   [flags]  Run a four-node local network and finalize blocks
  aidchaind node   [flags]  Run one lightweight HotStuff validator
  aidchaind miner  [flags]  Run one AI-PoW miner sidecar
  aidchaind submit [flags]  Submit an AI job to any running node
  aidchaind zkml-setup [flags]  Generate Groth16 circuit and proving/verifying keys`)
}

func runDemo(args []string) error {
	flags := flag.NewFlagSet("demo", flag.ContinueOnError)
	rounds := flags.Int("rounds", 3, "number of blocks to finalize")
	basePort := flags.Int("base-port", 19000, "first HTTP port")
	difficulty := flags.Int("difficulty", 10, "leading zero bits required by AI-PoW")
	maxNonce := flags.Uint64("max-nonce", 5_000_000, "maximum nonce per miner")
	evmURL := flags.String("evm-url", "", "optional upstream EVM JSON-RPC URL; empty uses deterministic mock")
	manifestRoot := flags.String("manifest-root", "", "optional unified AI-DAG manifest root")
	aidagPath := flags.String("aidag-dag", "", "unified AI-DAG file used only by miner sidecars")
	aidagMeta := flags.String("aidag-meta", "", "AI-DAG manifest used by miners and lightweight validators")
	aidagSidecar := flags.String("aidag-sidecar", "", "AI-DAG Merkle sidecar; defaults to <dag>.sidecar")
	powSamples := flags.Int("aiseal-pow-samples", 16, "AISeal PoW page samples")
	tensorSamples := flags.Int("aiseal-tensor-samples", 4, "AISeal tensor page samples")
	aisealTarget := flags.String("aiseal-target", "", "optional AISeal 32-byte target")
	maxSealAttempts := flags.Uint64("aiseal-max-attempts", 1024, "maximum AISeal nonce attempts per miner")
	engineURLs := flags.String("engine-urls", "", "optional id=url list for four Engine API endpoints")
	engineJWT := flags.String("engine-jwt", "", "Engine API JWT secret hex or file")
	engineGenesis := flags.String("engine-genesis-hash", protocol.GenesisHash, "execution genesis/head hash")
	engineFork := flags.String("engine-fork", string(engine.ForkOsaka), "Engine API fork: osaka or amsterdam")
	zkmlEnabled := flags.Bool("zkml", false, "require Groth16 zkML proofs and replicated result QC")
	zkmlArtifacts := flags.String("zkml-artifacts", "", "optional pre-generated zkML artifact directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *rounds <= 0 || *difficulty < 0 || *difficulty > 255 {
		return errors.New("rounds must be positive and difficulty must be in 0..255")
	}

	ids := []string{"node0", "node1", "node2", "node3"}
	validatorSet, privateKeys, err := node.DevnetValidatorSet(ids)
	if err != nil {
		return err
	}
	var manifest *aiseal.Manifest
	if *aidagMeta != "" {
		loaded, err := aiseal.LoadManifestFile(*aidagMeta)
		if err != nil {
			return err
		}
		manifest = &loaded
		*manifestRoot = loaded.ManifestRoot
		if *aidagPath == "" {
			return errors.New("--aidag-dag is required when --aidag-meta is set")
		}
	}
	var zkProver *zkml.Prover
	var zkVerifier *zkml.Verifier
	if *zkmlEnabled || *zkmlArtifacts != "" {
		if *zkmlArtifacts == "" {
			zkProver, zkVerifier, err = zkml.Setup()
		} else {
			zkProver, zkVerifier, err = zkml.LoadProver(*zkmlArtifacts)
		}
		if err != nil {
			return fmt.Errorf("initialize zkML: %w", err)
		}
		fmt.Printf("zkML enabled circuit=%s\n", zkVerifier.CircuitID())
	}
	fork, err := parseEngineFork(*engineFork)
	if err != nil {
		return err
	}
	var externalEngines map[string]string
	var jwtSecret []byte
	if *engineURLs != "" {
		externalEngines, err = parsePeers(*engineURLs)
		if err != nil {
			return fmt.Errorf("engine URLs: %w", err)
		}
		jwtSecret, err = engine.LoadJWTSecret(*engineJWT)
		if err != nil {
			return err
		}
	}
	nodes := make([]*node.Node, 0, len(ids))
	servers := make([]*node.Server, 0, len(ids))
	minerServers := make([]*miner.Server, 0, len(ids))
	for i, id := range ids {
		var executionEngine engine.Engine = engine.NewMock(fork, *engineGenesis)
		if externalEngines != nil {
			executionEngine, err = engine.NewClient(engine.ClientConfig{
				URL: externalEngines[id], Fork: fork, GenesisHash: *engineGenesis, JWTSecret: jwtSecret,
			})
			if err != nil {
				shutdownServers(servers)
				shutdownMinerServers(minerServers)
				return err
			}
		}
		var aiRunner aipow.Runner = aipow.DummyRunner{ManifestRoot: *manifestRoot}
		if zkVerifier != nil {
			aiRunner = aipow.QuantizedRunner{ManifestRoot: *manifestRoot}
		}
		n, err := node.New(node.Config{
			ID: id, PrivateKey: privateKeys[id], Validators: validatorSet,
			Difficulty: uint8(*difficulty), ManifestRoot: *manifestRoot, AISealManifest: manifest,
			AISealTarget: *aisealTarget, MinPoWSamples: *powSamples, MinTensorSamples: *tensorSamples,
			EVMURL: *evmURL, ExecutionEngine: executionEngine, AIRunner: aiRunner, ZKMLVerifier: zkVerifier,
		})
		if err != nil {
			shutdownServers(servers)
			return err
		}
		server, err := node.StartServer(n, "127.0.0.1:"+strconv.Itoa(*basePort+i))
		if err != nil {
			shutdownServers(servers)
			return err
		}
		nodes = append(nodes, n)
		servers = append(servers, server)

		minerService, err := miner.New(id, privateKeys[id], uint8(*difficulty), *maxNonce, *manifestRoot)
		if err != nil {
			shutdownServers(servers)
			shutdownMinerServers(minerServers)
			return err
		}
		if manifest != nil {
			minerService.SetProofProvider(&miner.FileProofProvider{
				DAGPath: *aidagPath, SidecarPath: *aidagSidecar, Manifest: *manifest,
				PoWSamples: *powSamples, TensorSamples: *tensorSamples, Target: *aisealTarget, MaxSealAttempts: *maxSealAttempts,
			})
		}
		if zkProver != nil {
			minerService.SetRunner(aipow.QuantizedRunner{ManifestRoot: *manifestRoot})
			minerService.SetZKMLProver(zkProver)
		}
		minerServer, err := miner.StartServer(minerService, "127.0.0.1:"+strconv.Itoa(*basePort+100+i))
		if err != nil {
			shutdownServers(servers)
			shutdownMinerServers(minerServers)
			return err
		}
		minerServers = append(minerServers, minerServer)
	}
	defer shutdownServers(servers)
	defer shutdownMinerServers(minerServers)

	peers := make(map[string]string, len(ids))
	miners := make(map[string]string, len(ids))
	for i, id := range ids {
		peers[id] = servers[i].URL
		miners[id] = minerServers[i].URL
	}
	for _, n := range nodes {
		n.SetPeers(peers)
		n.SetMiners(miners)
	}

	fmt.Printf("four-node devnet ready (quorum=%d, difficulty=%d)\n", validatorSet.Quorum(), *difficulty)
	for _, id := range ids {
		fmt.Printf("  %s validator=%s miner=%s\n", id, peers[id], miners[id])
	}

	wantExecutionHead := ""
	for i := 0; i < *rounds; i++ {
		job := protocol.NewAIJob("cypheriumai-light-v1-alpha.gguf", fmt.Sprintf("distributed AI demo job %d", i+1))
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		response, roundErr := nodes[i%len(nodes)].RunRound(ctx, protocol.RoundRequest{
			Job: job,
			EVM: protocol.EVMRequest{Method: "eth_chainId", Params: json.RawMessage("[]")},
		})
		cancel()
		if roundErr != nil {
			return fmt.Errorf("round %d: %w", i+1, roundErr)
		}
		wantExecutionHead = response.Block.Execution.BlockHash
		fmt.Printf("finalized height=%d proposer=%s block=%s ai=%s execution=%s votes=%d\n",
			response.Block.Header.Height, response.Block.Header.ProposerID, short(response.BlockHash),
			short(response.Block.Result.AIDigest), short(response.Block.Header.EVMStateRoot), len(response.CommitQC.Votes))
	}

	wantHeight := uint64(*rounds)
	for _, n := range nodes {
		status := n.Status()
		if status.Height != wantHeight || !strings.EqualFold(status.ExecutionHead, wantExecutionHead) {
			return fmt.Errorf("node %s disagrees: height=%d/%d execution=%s/%s", status.NodeID, status.Height, wantHeight, status.ExecutionHead, wantExecutionHead)
		}
	}
	fmt.Printf("all four nodes agree at height %d\n", wantHeight)
	return nil
}

func runNode(args []string) error {
	flags := flag.NewFlagSet("node", flag.ContinueOnError)
	id := flags.String("id", "", "validator id, for example node0")
	listen := flags.String("listen", "127.0.0.1:19000", "HTTP listen address")
	peerSpec := flags.String("peers", "", "comma-separated id=url list containing all validators")
	minerSpec := flags.String("miners", "", "comma-separated id=url list containing AI miner sidecars")
	difficulty := flags.Int("difficulty", 10, "leading zero bits required by AI-PoW")
	evmURL := flags.String("evm-url", "", "optional upstream EVM JSON-RPC URL")
	manifestRoot := flags.String("manifest-root", "", "optional unified AI-DAG manifest root")
	aidagMeta := flags.String("aidag-meta", "", "AI-DAG manifest for lightweight AISeal verification")
	powSamples := flags.Int("aiseal-pow-samples", 16, "minimum AISeal PoW samples")
	tensorSamples := flags.Int("aiseal-tensor-samples", 4, "minimum AISeal tensor samples")
	aisealTarget := flags.String("aiseal-target", "", "optional AISeal target")
	objectDir := flags.String("storage-dir", "", "content-addressed finalized block directory")
	engineURL := flags.String("engine-url", "", "Engine API endpoint; empty uses mock engine")
	engineJWT := flags.String("engine-jwt", "", "Engine API JWT secret hex or file")
	engineGenesis := flags.String("engine-genesis-hash", protocol.GenesisHash, "execution genesis/head hash")
	engineFork := flags.String("engine-fork", string(engine.ForkOsaka), "Engine API fork: osaka or amsterdam")
	zkmlArtifacts := flags.String("zkml-artifacts", "", "zkML artifact directory; enables lightweight Groth16 verification")
	minZKMLProofs := flags.Int("zkml-min-proofs", 0, "minimum zkML proofs per result; zero uses HotStuff quorum")
	if err := flags.Parse(args); err != nil {
		return err
	}
	peers, err := parsePeers(*peerSpec)
	if err != nil {
		return err
	}
	if _, ok := peers[*id]; !ok {
		return errors.New("--id must appear in --peers")
	}
	miners, err := parsePeers(*minerSpec)
	if err != nil {
		return fmt.Errorf("miners: %w", err)
	}
	if *difficulty < 0 || *difficulty > 255 {
		return errors.New("difficulty must be in 0..255")
	}
	ids := make([]string, 0, len(peers))
	for peerID := range peers {
		ids = append(ids, peerID)
	}
	sort.Strings(ids)
	validatorSet, privateKeys, err := node.DevnetValidatorSet(ids)
	if err != nil {
		return err
	}
	var manifest *aiseal.Manifest
	if *aidagMeta != "" {
		loaded, err := aiseal.LoadManifestFile(*aidagMeta)
		if err != nil {
			return err
		}
		manifest = &loaded
		*manifestRoot = loaded.ManifestRoot
	}
	var zkVerifier *zkml.Verifier
	if *zkmlArtifacts != "" {
		zkVerifier, err = zkml.LoadVerifier(*zkmlArtifacts)
		if err != nil {
			return fmt.Errorf("load zkML verifier: %w", err)
		}
	}
	fork, err := parseEngineFork(*engineFork)
	if err != nil {
		return err
	}
	var executionEngine engine.Engine = engine.NewMock(fork, *engineGenesis)
	if *engineURL != "" {
		secret, err := engine.LoadJWTSecret(*engineJWT)
		if err != nil {
			return err
		}
		executionEngine, err = engine.NewClient(engine.ClientConfig{
			URL: *engineURL, Fork: fork, GenesisHash: *engineGenesis, JWTSecret: secret,
		})
		if err != nil {
			return err
		}
	}
	var objectStore *storage.Store
	if *objectDir != "" {
		objectStore, err = storage.New(*objectDir, 32<<20)
		if err != nil {
			return err
		}
	}
	var aiRunner aipow.Runner = aipow.DummyRunner{ManifestRoot: *manifestRoot}
	if zkVerifier != nil {
		aiRunner = aipow.QuantizedRunner{ManifestRoot: *manifestRoot}
	}
	n, err := node.New(node.Config{
		ID: *id, PrivateKey: privateKeys[*id], Validators: validatorSet, Peers: peers, Miners: miners,
		Difficulty: uint8(*difficulty), ManifestRoot: *manifestRoot, AISealManifest: manifest,
		AISealTarget: *aisealTarget, MinPoWSamples: *powSamples, MinTensorSamples: *tensorSamples,
		EVMURL: *evmURL, ExecutionEngine: executionEngine, ObjectStore: objectStore,
		AIRunner: aiRunner, ZKMLVerifier: zkVerifier, MinZKMLProofs: *minZKMLProofs,
	})
	if err != nil {
		return err
	}
	server, err := node.StartServer(n, *listen)
	if err != nil {
		return err
	}
	fmt.Printf("%s listening on %s (quorum=%d)\n", *id, server.URL, validatorSet.Quorum())
	fmt.Println("WARNING: deterministic devnet validator keys are not safe for production")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

func runMiner(args []string) error {
	flags := flag.NewFlagSet("miner", flag.ContinueOnError)
	id := flags.String("id", "", "miner identity corresponding to a validator, for example node0")
	listen := flags.String("listen", "127.0.0.1:19100", "miner sidecar HTTP listen address")
	difficulty := flags.Int("difficulty", 10, "leading zero bits required by AI-PoW")
	maxNonce := flags.Uint64("max-nonce", 5_000_000, "maximum nonce per job")
	manifestRoot := flags.String("manifest-root", "", "unified AI-DAG manifest root")
	aidagPath := flags.String("aidag-dag", "", "unified AI-DAG file for real AISeal proof generation")
	aidagMeta := flags.String("aidag-meta", "", "AI-DAG manifest")
	aidagSidecar := flags.String("aidag-sidecar", "", "AI-DAG Merkle sidecar")
	powSamples := flags.Int("aiseal-pow-samples", 16, "AISeal PoW samples")
	tensorSamples := flags.Int("aiseal-tensor-samples", 4, "AISeal tensor samples")
	aisealTarget := flags.String("aiseal-target", "", "optional AISeal target")
	storageDir := flags.String("storage-dir", "", "content-addressed AI proof directory")
	maxSealAttempts := flags.Uint64("aiseal-max-attempts", 1024, "maximum AISeal nonce attempts")
	zkmlArtifacts := flags.String("zkml-artifacts", "", "zkML artifact directory; enables quantized inference proofs")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *id == "" || *difficulty < 0 || *difficulty > 255 {
		return errors.New("--id is required and difficulty must be in 0..255")
	}
	var manifest *aiseal.Manifest
	if *aidagMeta != "" {
		loaded, err := aiseal.LoadManifestFile(*aidagMeta)
		if err != nil {
			return err
		}
		if *aidagPath == "" {
			return errors.New("--aidag-dag is required with --aidag-meta")
		}
		manifest = &loaded
		*manifestRoot = loaded.ManifestRoot
	}
	service, err := miner.New(*id, node.DevnetPrivateKey(*id), uint8(*difficulty), *maxNonce, *manifestRoot)
	if err != nil {
		return err
	}
	if *storageDir != "" {
		store, err := storage.New(*storageDir, 16<<20)
		if err != nil {
			return err
		}
		service.SetProofStore(store)
	}
	if manifest != nil {
		service.SetProofProvider(&miner.FileProofProvider{
			DAGPath: *aidagPath, SidecarPath: *aidagSidecar, Manifest: *manifest,
			PoWSamples: *powSamples, TensorSamples: *tensorSamples, Target: *aisealTarget, MaxSealAttempts: *maxSealAttempts,
		})
	}
	if *zkmlArtifacts != "" {
		prover, _, err := zkml.LoadProver(*zkmlArtifacts)
		if err != nil {
			return fmt.Errorf("load zkML prover: %w", err)
		}
		service.SetRunner(aipow.QuantizedRunner{ManifestRoot: *manifestRoot})
		service.SetZKMLProver(prover)
	}
	server, err := miner.StartServer(service, *listen)
	if err != nil {
		return err
	}
	fmt.Printf("AI miner sidecar %s listening on %s\n", *id, server.URL)
	fmt.Println("DAG/inference work stays in this process; HotStuff validators only verify its commitments")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

func runSubmit(args []string) error {
	flags := flag.NewFlagSet("submit", flag.ContinueOnError)
	url := flags.String("url", "http://127.0.0.1:19000", "URL of any network node")
	model := flags.String("model", "cypheriumai-light-v1-alpha.gguf", "model identifier")
	prompt := flags.String("prompt", "hello decentralized AI", "AI prompt")
	evmMethod := flags.String("evm-method", "eth_chainId", "EVM JSON-RPC method to forward")
	evmParams := flags.String("evm-params", "[]", "EVM JSON-RPC params as JSON")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if !json.Valid([]byte(*evmParams)) {
		return errors.New("--evm-params must be valid JSON")
	}
	request := protocol.RoundRequest{
		Job: protocol.NewAIJob(*model, *prompt),
		EVM: protocol.EVMRequest{Method: *evmMethod, Params: json.RawMessage(*evmParams)},
	}
	payload, _ := json.Marshal(request)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(*url, "/")+"/v1/round", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(httpRequest)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("node returned HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	var round protocol.RoundResponse
	if err := json.Unmarshal(body, &round); err != nil {
		return err
	}
	pretty, _ := json.MarshalIndent(round, "", "  ")
	fmt.Println(string(pretty))
	return nil
}

func runZKMLSetup(args []string) error {
	flags := flag.NewFlagSet("zkml-setup", flag.ContinueOnError)
	out := flags.String("out", "zkml-artifacts", "output directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	prover, verifier, err := zkml.Setup()
	if err != nil {
		return err
	}
	if err := prover.Save(*out); err != nil {
		return err
	}
	fmt.Printf("zkML Groth16 artifacts written to %s (circuit=%s)\n", *out, verifier.CircuitID())
	return nil
}

func parsePeers(spec string) (map[string]string, error) {
	peers := make(map[string]string)
	for _, item := range strings.Split(spec, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		parts := strings.SplitN(item, "=", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
			return nil, fmt.Errorf("invalid peer %q; expected id=url", item)
		}
		peers[strings.TrimSpace(parts[0])] = strings.TrimRight(strings.TrimSpace(parts[1]), "/")
	}
	if len(peers) < 4 {
		return nil, errors.New("at least four peers are required")
	}
	return peers, nil
}

func parseEngineFork(value string) (engine.Fork, error) {
	switch engine.Fork(strings.ToLower(strings.TrimSpace(value))) {
	case engine.ForkOsaka:
		return engine.ForkOsaka, nil
	case engine.ForkAmsterdam:
		return engine.ForkAmsterdam, nil
	default:
		return "", fmt.Errorf("unsupported Engine API fork %q", value)
	}
}

func shutdownServers(servers []*node.Server) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	for _, server := range servers {
		_ = server.Shutdown(ctx)
	}
}

func shutdownMinerServers(servers []*miner.Server) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	for _, server := range servers {
		_ = server.Shutdown(ctx)
	}
}

func short(hash string) string {
	if len(hash) <= 14 {
		return hash
	}
	return hash[:14]
}
