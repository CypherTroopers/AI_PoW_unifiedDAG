package miner

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aipow"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aiseal"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/storage"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/zkml"
)

type Service struct {
	ID            string
	PrivateKey    ed25519.PrivateKey
	Difficulty    uint8
	MaxNonce      uint64
	ManifestRoot  string
	Runner        aipow.Runner
	ZKMLProver    *zkml.Prover
	ProofProvider ProofProvider

	proofStore *storage.Store
}

type ProofProvider interface {
	Create(context.Context, protocol.MiningRequest, string) (aiseal.Proof, []byte, error)
}

type FileProofProvider struct {
	DAGPath         string
	SidecarPath     string
	Manifest        aiseal.Manifest
	PoWSamples      int
	TensorSamples   int
	Target          string
	MaxSealAttempts uint64
	nonce           atomic.Uint64
}

func (p *FileProofProvider) Create(ctx context.Context, request protocol.MiningRequest, minerID string) (aiseal.Proof, []byte, error) {
	attempts := p.MaxSealAttempts
	if attempts == 0 {
		attempts = 1024
	}
	for attempt := uint64(0); attempt < attempts; attempt++ {
		proof, data, err := aiseal.ProveFiles(ctx, aiseal.ProveConfig{
			DAGPath: p.DAGPath, SidecarPath: p.SidecarPath, Manifest: p.Manifest,
			BlockHash: request.ParentHash, Miner: minerID, Epoch: request.Height, Nonce: p.nonce.Add(1) - 1,
			PoWSamples: p.PoWSamples, TensorSamples: p.TensorSamples,
		})
		if err != nil {
			return aiseal.Proof{}, nil, err
		}
		if p.Target == "" {
			return proof, data, nil
		}
		meets, err := aiseal.WorkMeetsTarget(proof.Seal.WorkHash, p.Target)
		if err != nil {
			return aiseal.Proof{}, nil, err
		}
		if meets {
			return proof, data, nil
		}
	}
	return aiseal.Proof{}, nil, fmt.Errorf("no AISeal nonce met target in %d attempts", attempts)
}

func New(id string, privateKey ed25519.PrivateKey, difficulty uint8, maxNonce uint64, manifestRoot string) (*Service, error) {
	if id == "" || len(privateKey) != ed25519.PrivateKeySize {
		return nil, errors.New("miner id and Ed25519 private key are required")
	}
	if maxNonce == 0 {
		maxNonce = 5_000_000
	}
	proofStore, err := storage.New("", 16<<20)
	if err != nil {
		return nil, err
	}
	return &Service{
		ID: id, PrivateKey: append(ed25519.PrivateKey(nil), privateKey...), Difficulty: difficulty,
		MaxNonce: maxNonce, ManifestRoot: manifestRoot, Runner: aipow.DummyRunner{ManifestRoot: manifestRoot}, proofStore: proofStore,
	}, nil
}

func (s *Service) SetProofProvider(provider ProofProvider) { s.ProofProvider = provider }

func (s *Service) SetRunner(runner aipow.Runner) {
	if runner != nil {
		s.Runner = runner
	}
}

func (s *Service) SetZKMLProver(prover *zkml.Prover) { s.ZKMLProver = prover }

func (s *Service) SetProofStore(store *storage.Store) {
	if store != nil {
		s.proofStore = store
	}
}

func (s *Service) Mine(ctx context.Context, request protocol.MiningRequest) (protocol.MiningResponse, error) {
	if request.Height == 0 || request.ParentHash == "" {
		return protocol.MiningResponse{}, errors.New("height and parent hash are required")
	}
	if err := protocol.ValidateAIJob(request.Job); err != nil {
		return protocol.MiningResponse{}, err
	}
	result, err := s.Runner.Run(ctx, request.Job)
	if err != nil {
		return protocol.MiningResponse{}, err
	}
	aiDigest := result.AIDigest
	aiProofHash := result.AIDigest
	var bundle *protocol.AIProofBundle
	if s.ProofProvider != nil {
		proof, proofBytes, err := s.ProofProvider.Create(ctx, request, s.ID)
		if err != nil {
			return protocol.MiningResponse{}, fmt.Errorf("create AISeal proof: %w", err)
		}
		aiDigest = proof.Seal.AIDigest
		aiProofHash = proof.Seal.ProofHash
		contentHash, err := s.proofStore.Put(proofBytes)
		if err != nil {
			return protocol.MiningResponse{}, err
		}
		bundle = &protocol.AIProofBundle{
			MinerID: s.ID, ProofHash: proof.Seal.ProofHash, ContentHash: contentHash,
			Proof: append(json.RawMessage(nil), proofBytes...),
		}
	}
	var zkBundle *protocol.ZKMLProofBundle
	zkProofHash := ""
	if s.ZKMLProver != nil {
		input, model := zkml.Derive(request.Job.Model, request.Job.Prompt)
		proof, err := s.ZKMLProver.Prove(input, model)
		if err != nil {
			return protocol.MiningResponse{}, fmt.Errorf("create zkML proof: %w", err)
		}
		if result.Output != "zkml-linear-v1:"+proof.Output {
			return protocol.MiningResponse{}, errors.New("inference result is not bound to zkML public output")
		}
		zkProofHash = zkml.ProofHash(proof)
		zkBundle = &protocol.ZKMLProofBundle{
			WorkerID: s.ID, Scheme: proof.Scheme, CircuitID: proof.CircuitID,
			ModelCommitment: proof.ModelCommitment, InputCommitment: proof.InputCommitment,
			Output: proof.Output, Proof: append([]byte(nil), proof.Proof...), ProofHash: zkProofHash,
		}
	}
	ticket, err := aipow.Mine(ctx, protocol.PoWTicket{
		Height: request.Height, ParentHash: request.ParentHash, JobID: request.Job.ID,
		ResultHash: protocol.HashJSON(result), AIDigest: aiDigest, MinerID: s.ID,
		Difficulty: s.Difficulty, ManifestRoot: s.ManifestRoot, AIProofHash: aiProofHash, ZKMLProofHash: zkProofHash,
	}, s.MaxNonce)
	if err != nil {
		return protocol.MiningResponse{}, err
	}
	ticket.Signature = base64.StdEncoding.EncodeToString(ed25519.Sign(s.PrivateKey, protocol.TicketSigningBytes(ticket)))
	attestation := protocol.InferenceAttestation{
		WorkerID: s.ID, Height: request.Height, ParentHash: request.ParentHash,
		JobID: request.Job.ID, ResultHash: protocol.HashJSON(result),
		ModelHash: result.ModelHash, InputHash: result.InputHash, OutputHash: result.OutputHash,
		ZKMLProofHash: zkProofHash,
	}
	attestation.Signature = base64.StdEncoding.EncodeToString(ed25519.Sign(s.PrivateKey, protocol.InferenceSigningBytes(attestation)))
	return protocol.MiningResponse{
		Ticket: ticket, Result: result, Attestation: attestation, AIProof: bundle, ZKMLProof: zkBundle,
	}, nil
}

func (s *Service) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "minerId": s.ID, "role": "ai-miner-sidecar"})
	})
	mux.HandleFunc("GET /v1/proofs/{hash}", func(w http.ResponseWriter, r *http.Request) {
		hash := r.PathValue("hash")
		proof, ok := s.proofStore.Get(hash)
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "proof not found"})
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(proof)
	})
	mux.HandleFunc("POST /v1/mine", func(w http.ResponseWriter, r *http.Request) {
		var request protocol.MiningRequest
		decoder := json.NewDecoder(io.LimitReader(r.Body, 8<<20))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
			return
		}
		response, err := s.Mine(r.Context(), request)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, response)
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mux.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

type Server struct {
	Service    *Service
	HTTPServer *http.Server
	Listener   net.Listener
	URL        string
}

func StartServer(service *Service, address string) (*Server, error) {
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return nil, err
	}
	host := listener.Addr().String()
	if strings.HasPrefix(host, "[::]") {
		host = "127.0.0.1" + strings.TrimPrefix(host, "[::]")
	}
	httpServer := &http.Server{
		Handler: service.Handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second,
		WriteTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second,
	}
	running := &Server{Service: service, HTTPServer: httpServer, Listener: listener, URL: "http://" + host}
	go func() { _ = httpServer.Serve(listener) }()
	return running, nil
}

func (s *Server) Shutdown(ctx context.Context) error { return s.HTTPServer.Shutdown(ctx) }
