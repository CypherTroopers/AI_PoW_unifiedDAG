package aipow

import (
	"context"
	"errors"
	"strconv"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/protocol"
	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/zkml"
)

type Runner interface {
	Run(context.Context, protocol.AIJob) (protocol.AIResult, error)
	Verify(context.Context, protocol.AIJob, protocol.AIResult) error
}

type QuantizedRunner struct {
	ManifestRoot string
}

func (r QuantizedRunner) Run(_ context.Context, job protocol.AIJob) (protocol.AIResult, error) {
	if job.ID == "" || job.Model == "" || job.Prompt == "" {
		return protocol.AIResult{}, errors.New("job id, model, and prompt are required")
	}
	input, model := zkml.Derive(job.Model, job.Prompt)
	output := "zkml-linear-v1:" + strconv.FormatUint(zkml.Evaluate(input, model), 10)
	modelHash := protocol.HashBytes([]byte(job.Model))
	inputHash := protocol.HashBytes([]byte(job.Prompt))
	outputHash := protocol.HashBytes([]byte(output))
	aiDigest := protocol.HashJSON(struct {
		Domain       string `json:"domain"`
		JobID        string `json:"jobId"`
		ModelHash    string `json:"modelHash"`
		InputHash    string `json:"inputHash"`
		OutputHash   string `json:"outputHash"`
		ManifestRoot string `json:"manifestRoot"`
	}{"AIDCHAIN_AI_RECEIPT_V1", job.ID, modelHash, inputHash, outputHash, r.ManifestRoot})
	return protocol.AIResult{
		JobID: job.ID, ModelHash: modelHash, InputHash: inputHash, Output: output,
		OutputHash: outputHash, AIDigest: aiDigest, ProofScheme: zkml.Scheme,
	}, nil
}

func (r QuantizedRunner) Verify(ctx context.Context, job protocol.AIJob, result protocol.AIResult) error {
	want, err := r.Run(ctx, job)
	if err != nil {
		return err
	}
	if protocol.HashJSON(want) != protocol.HashJSON(result) {
		return errors.New("quantized AI result does not match deterministic execution")
	}
	return nil
}
