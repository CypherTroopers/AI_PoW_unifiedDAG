package node

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

func (n *Node) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", n.handleHealth)
	mux.HandleFunc("GET /v1/status", n.handleStatus)
	mux.HandleFunc("GET /v1/blocks", n.handleBlocks)
	mux.HandleFunc("GET /v1/objects/{hash}", n.handleObject)
	mux.HandleFunc("POST /v1/hotstuff/proposal", n.handleProposal)
	mux.HandleFunc("POST /v1/hotstuff/qc", n.handleQC)
	mux.HandleFunc("POST /v1/propose", n.handlePropose)
	mux.HandleFunc("POST /v1/round", n.handleRound)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		defer func() {
			if recovered := recover(); recovered != nil {
				writeError(w, http.StatusInternalServerError, fmt.Errorf("panic: %v", recovered))
			}
		}()
		mux.ServeHTTP(w, r)
	})
}

func (n *Node) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "nodeId": n.id})
}

func (n *Node) handleStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, n.Status())
}

func (n *Node) handleBlocks(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, n.Blocks())
}

func (n *Node) handleObject(w http.ResponseWriter, r *http.Request) {
	object, ok := n.Object(r.PathValue("hash"))
	if !ok {
		writeError(w, http.StatusNotFound, errors.New("object not found"))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(object)
}

func (n *Node) handleProposal(w http.ResponseWriter, r *http.Request) {
	var request protocol.ProposalRequest
	if !decodeRequest(w, r, &request) {
		return
	}
	vote, err := n.OnProposal(request.Block)
	if err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, http.StatusOK, vote)
}

func (n *Node) handleQC(w http.ResponseWriter, r *http.Request) {
	var request protocol.QCRequest
	if !decodeRequest(w, r, &request) {
		return
	}
	vote, finalized, err := n.OnQC(request.Block, request.QC)
	if err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, http.StatusOK, qcNetworkResponse{Vote: vote, Finalized: finalized})
}

func (n *Node) handlePropose(w http.ResponseWriter, r *http.Request) {
	var request protocol.ProposeRequest
	if !decodeRequest(w, r, &request) {
		return
	}
	response, err := n.Propose(r.Context(), request)
	if err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func (n *Node) handleRound(w http.ResponseWriter, r *http.Request) {
	var request protocol.RoundRequest
	if !decodeRequest(w, r, &request) {
		return
	}
	response, err := n.RunRound(r.Context(), request)
	if err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

func decodeRequest(w http.ResponseWriter, r *http.Request, target any) bool {
	decoder := json.NewDecoder(io.LimitReader(r.Body, 8<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

type Server struct {
	Node       *Node
	HTTPServer *http.Server
	Listener   net.Listener
	URL        string
}

func StartServer(n *Node, address string) (*Server, error) {
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return nil, err
	}
	host := listener.Addr().String()
	if strings.HasPrefix(host, "[::]") {
		host = "127.0.0.1" + strings.TrimPrefix(host, "[::]")
	}
	server := &http.Server{
		Handler: n.Handler(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second,
		WriteTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second,
	}
	running := &Server{Node: n, HTTPServer: server, Listener: listener, URL: "http://" + host}
	go func() {
		if serveErr := server.Serve(listener); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			// The CLI observes failures through health/status requests. Avoid a process-wide panic here.
			return
		}
	}()
	return running, nil
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.HTTPServer.Shutdown(ctx)
}
