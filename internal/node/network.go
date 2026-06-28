package node

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/engine"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
)

type qcNetworkResponse struct {
	Vote      *protocol.Vote `json:"vote,omitempty"`
	Finalized bool           `json:"finalized"`
}

func (n *Node) RunRound(ctx context.Context, request protocol.RoundRequest) (protocol.RoundResponse, error) {
	n.roundMu.Lock()
	defer n.roundMu.Unlock()

	if request.Job.ID == "" {
		request.Job = protocol.NewAIJob(request.Job.Model, request.Job.Prompt)
	}
	if err := protocol.ValidateAIJob(request.Job); err != nil {
		return protocol.RoundResponse{}, err
	}
	height, tip := n.replica.HeightAndTip()
	peers := n.Peers()
	miners := n.Miners()
	if len(peers) < n.validators.Quorum() {
		return protocol.RoundResponse{}, fmt.Errorf("not enough configured peers: got=%d need=%d", len(peers), n.validators.Quorum())
	}
	if len(miners) < n.validators.Quorum() {
		return protocol.RoundResponse{}, fmt.Errorf("not enough configured AI miner sidecars: got=%d need=%d", len(miners), n.validators.Quorum())
	}

	miningRequest := protocol.MiningRequest{Height: height + 1, ParentHash: tip, Job: request.Job}
	tickets, result, proofs, resultQC, zkmlProofs := n.collectTickets(ctx, miners, miningRequest)
	if len(tickets) < n.validators.Quorum() {
		return protocol.RoundResponse{}, fmt.Errorf("mining quorum not reached: got=%d need=%d", len(tickets), n.validators.Quorum())
	}
	if n.zkmlVerifier == nil {
		if err := n.runner.Verify(ctx, request.Job, result); err != nil {
			return protocol.RoundResponse{}, err
		}
	} else if err := n.validateResultCommitment(request.Job, result); err != nil {
		return protocol.RoundResponse{}, err
	}
	winner, err := n.validateCompetition(height+1, tip, request.Job, result, tickets, proofs)
	if err != nil {
		return protocol.RoundResponse{}, err
	}
	if err := n.validateResultCertificate(height+1, tip, request.Job, result, tickets, resultQC, zkmlProofs); err != nil {
		return protocol.RoundResponse{}, err
	}
	winnerURL, ok := peers[winner.MinerID]
	if !ok {
		return protocol.RoundResponse{}, fmt.Errorf("winner %s has no peer URL", winner.MinerID)
	}

	var response protocol.RoundResponse
	err = n.postJSON(ctx, winnerURL+"/v1/propose", protocol.ProposeRequest{
		Job: request.Job, Result: result, EVM: request.EVM, Tickets: tickets, AIProofs: proofs,
		ResultQC: resultQC, ZKMLProofs: zkmlProofs,
	}, &response)
	if err != nil {
		return protocol.RoundResponse{}, fmt.Errorf("winning proposer %s: %w", winner.MinerID, err)
	}
	return response, nil
}

func (n *Node) Propose(ctx context.Context, request protocol.ProposeRequest) (protocol.RoundResponse, error) {
	n.proposeMu.Lock()
	defer n.proposeMu.Unlock()

	if err := protocol.ValidateAIJob(request.Job); err != nil {
		return protocol.RoundResponse{}, err
	}
	height, tip := n.replica.HeightAndTip()
	if n.zkmlVerifier == nil {
		if err := n.runner.Verify(ctx, request.Job, request.Result); err != nil {
			return protocol.RoundResponse{}, err
		}
	} else if err := n.validateResultCommitment(request.Job, request.Result); err != nil {
		return protocol.RoundResponse{}, err
	}
	result := request.Result
	winner, err := n.validateCompetition(height+1, tip, request.Job, result, request.Tickets, request.AIProofs)
	if err != nil {
		return protocol.RoundResponse{}, err
	}
	if err := n.validateResultCertificate(height+1, tip, request.Job, result, request.Tickets, request.ResultQC, request.ZKMLProofs); err != nil {
		return protocol.RoundResponse{}, err
	}
	if winner.MinerID != n.id {
		return protocol.RoundResponse{}, fmt.Errorf("node %s is not selected proposer; winner=%s", n.id, winner.MinerID)
	}

	receipt, err := n.evm.Execute(ctx, tip, request.EVM)
	if err != nil {
		return protocol.RoundResponse{}, err
	}
	timestamp := time.Now().UTC().Truncate(time.Second)
	prevRandao := protocol.HashJSON(struct {
		Parent string `json:"parent"`
		JobID  string `json:"jobId"`
		Height uint64 `json:"height"`
	}{tip, request.Job.ID, height + 1})
	beaconRoot := protocol.HashJSON(struct {
		Domain string `json:"domain"`
		Parent string `json:"parent"`
		Height uint64 `json:"height"`
	}{"AIDCHAIN_PARENT_BEACON_ROOT_V1", tip, height + 1})
	identityHash := protocol.HashBytes([]byte(n.id))
	feeRecipient := "0x" + identityHash[len(identityHash)-40:]
	executionPayload, err := n.execution.Build(ctx, engine.BuildInput{
		Timestamp: uint64(timestamp.Unix()), PrevRandao: prevRandao, SuggestedFeeRecipient: feeRecipient,
		ParentBeaconBlockRoot: beaconRoot, SlotNumber: height + 1, TargetGasLimit: 30_000_000,
	})
	if err != nil {
		return protocol.RoundResponse{}, fmt.Errorf("build execution payload: %w", err)
	}
	tickets := append([]protocol.PoWTicket(nil), request.Tickets...)
	sort.Slice(tickets, func(i, j int) bool { return tickets[i].MinerID < tickets[j].MinerID })
	proofs := append([]protocol.AIProofBundle(nil), request.AIProofs...)
	sort.Slice(proofs, func(i, j int) bool { return proofs[i].MinerID < proofs[j].MinerID })
	resultQC := request.ResultQC
	resultQC.Attestations = append([]protocol.InferenceAttestation(nil), request.ResultQC.Attestations...)
	sort.Slice(resultQC.Attestations, func(i, j int) bool { return resultQC.Attestations[i].WorkerID < resultQC.Attestations[j].WorkerID })
	zkmlProofs := append([]protocol.ZKMLProofBundle(nil), request.ZKMLProofs...)
	sort.Slice(zkmlProofs, func(i, j int) bool { return zkmlProofs[i].WorkerID < zkmlProofs[j].WorkerID })
	block := protocol.Block{
		Header: protocol.BlockHeader{
			Version: protocol.ProtocolVersion, Height: height + 1, ParentHash: tip, ProposerID: n.id,
			Timestamp: timestamp.Format(time.RFC3339Nano), AIReceiptRoot: result.AIDigest,
			EVMStateRoot: executionPayload.StateRoot, EVMReceiptRoot: receipt.StateRoot,
			ExecutionHash: protocol.HashJSON(executionPayload), TicketsRoot: protocol.TicketsRoot(tickets), AIProofsRoot: protocol.AIProofsRoot(proofs),
			ResultQCRoot: protocol.ResultQCRoot(resultQC), ZKMLProofsRoot: protocol.ZKMLProofsRoot(zkmlProofs),
		},
		Job: request.Job, Result: result, EVMRequest: request.EVM, EVM: receipt, Execution: executionPayload,
		Winner: winner, Tickets: tickets, AIProofs: proofs, ResultQC: resultQC, ZKMLProofs: zkmlProofs,
	}
	blockHash := block.Hash()
	peers := n.Peers()

	prepareVotes := n.broadcastProposal(ctx, peers, block)
	prepareQC, err := n.validators.MakeQC(protocol.PhasePrepare, block.Header.Height, blockHash, prepareVotes)
	if err != nil {
		return protocol.RoundResponse{}, fmt.Errorf("prepare QC: %w", err)
	}
	preCommitVotes, _ := n.broadcastQC(ctx, peers, block, prepareQC, protocol.PhasePreCommit)
	preCommitQC, err := n.validators.MakeQC(protocol.PhasePreCommit, block.Header.Height, blockHash, preCommitVotes)
	if err != nil {
		return protocol.RoundResponse{}, fmt.Errorf("precommit QC: %w", err)
	}
	commitVotes, _ := n.broadcastQC(ctx, peers, block, preCommitQC, protocol.PhaseCommit)
	commitQC, err := n.validators.MakeQC(protocol.PhaseCommit, block.Header.Height, blockHash, commitVotes)
	if err != nil {
		return protocol.RoundResponse{}, fmt.Errorf("commit QC: %w", err)
	}
	_, finalized := n.broadcastQC(ctx, peers, block, commitQC, "")
	if finalized < n.validators.Quorum() {
		return protocol.RoundResponse{}, fmt.Errorf("finalization quorum not reached: got=%d need=%d", finalized, n.validators.Quorum())
	}
	return protocol.RoundResponse{Block: block, BlockHash: blockHash, CommitQC: commitQC}, nil
}

func (n *Node) collectTickets(ctx context.Context, miners map[string]string, request protocol.MiningRequest) ([]protocol.PoWTicket, protocol.AIResult, []protocol.AIProofBundle, protocol.ResultCertificate, []protocol.ZKMLProofBundle) {
	type result struct {
		id       string
		response protocol.MiningResponse
		err      error
	}
	results := make(chan result, len(miners))
	var wg sync.WaitGroup
	for id, url := range miners {
		wg.Add(1)
		go func(id, url string) {
			defer wg.Done()
			var response protocol.MiningResponse
			err := n.postJSON(ctx, url+"/v1/mine", request, &response)
			results <- result{id: id, response: response, err: err}
		}(id, url)
	}
	wg.Wait()
	close(results)

	responsesByResult := make(map[string][]protocol.MiningResponse)
	for result := range results {
		if result.err == nil && result.response.Ticket.MinerID == result.id {
			hash := protocol.HashJSON(result.response.Result)
			responsesByResult[hash] = append(responsesByResult[hash], result.response)
		}
	}
	var selected []protocol.MiningResponse
	selectedHash := ""
	for hash, responses := range responsesByResult {
		if len(responses) > len(selected) || (len(responses) == len(selected) && (selectedHash == "" || hash < selectedHash)) {
			selected = responses
			selectedHash = hash
		}
	}
	if len(selected) == 0 {
		return nil, protocol.AIResult{}, nil, protocol.ResultCertificate{}, nil
	}
	tickets := make([]protocol.PoWTicket, 0, len(selected))
	proofs := make([]protocol.AIProofBundle, 0, len(selected))
	attestations := make([]protocol.InferenceAttestation, 0, len(selected))
	zkmlProofs := make([]protocol.ZKMLProofBundle, 0, len(selected))
	for _, response := range selected {
		tickets = append(tickets, response.Ticket)
		attestations = append(attestations, response.Attestation)
		if response.AIProof != nil {
			proofs = append(proofs, *response.AIProof)
		}
		if response.ZKMLProof != nil {
			zkmlProofs = append(zkmlProofs, *response.ZKMLProof)
		}
	}
	certificate := protocol.ResultCertificate{ResultHash: selectedHash, Attestations: attestations}
	return tickets, selected[0].Result, proofs, certificate, zkmlProofs
}

func (n *Node) broadcastProposal(ctx context.Context, peers map[string]string, block protocol.Block) []protocol.Vote {
	type result struct {
		vote protocol.Vote
		err  error
	}
	results := make(chan result, len(peers))
	var wg sync.WaitGroup
	for _, url := range peers {
		wg.Add(1)
		go func(url string) {
			defer wg.Done()
			var vote protocol.Vote
			err := n.postJSON(ctx, url+"/v1/hotstuff/proposal", protocol.ProposalRequest{Block: block}, &vote)
			results <- result{vote: vote, err: err}
		}(url)
	}
	wg.Wait()
	close(results)

	votes := make([]protocol.Vote, 0, len(peers))
	for result := range results {
		if result.err == nil && result.vote.Phase == protocol.PhasePrepare {
			votes = append(votes, result.vote)
		}
	}
	return votes
}

func (n *Node) broadcastQC(ctx context.Context, peers map[string]string, block protocol.Block, qc protocol.QuorumCertificate, expected protocol.Phase) ([]protocol.Vote, int) {
	type result struct {
		response qcNetworkResponse
		err      error
	}
	results := make(chan result, len(peers))
	var wg sync.WaitGroup
	for _, url := range peers {
		wg.Add(1)
		go func(url string) {
			defer wg.Done()
			var response qcNetworkResponse
			err := n.postJSON(ctx, url+"/v1/hotstuff/qc", protocol.QCRequest{Block: block, QC: qc}, &response)
			results <- result{response: response, err: err}
		}(url)
	}
	wg.Wait()
	close(results)

	votes := make([]protocol.Vote, 0, len(peers))
	finalized := 0
	for result := range results {
		if result.err != nil {
			continue
		}
		if result.response.Finalized {
			finalized++
		}
		if result.response.Vote != nil && result.response.Vote.Phase == expected {
			votes = append(votes, *result.response.Vote)
		}
	}
	return votes, finalized
}

func (n *Node) postJSON(ctx context.Context, url string, input, output any) error {
	payload, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := n.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var apiError struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(body, &apiError) == nil && apiError.Error != "" {
			return errors.New(apiError.Error)
		}
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	if output == nil || len(body) == 0 {
		return nil
	}
	if err := json.Unmarshal(body, output); err != nil {
		return fmt.Errorf("decode %s: %w", url, err)
	}
	return nil
}
