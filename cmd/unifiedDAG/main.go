package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CypherTroopers/AI_PoW_unifiedDAG/internal/aiseal"
	"github.com/zeebo/blake3"
)

const (
	oneGiB = uint64(1024 * 1024 * 1024)

	defaultPageSize = uint64(4096)

	// v1/v2 unified AI-DAG header.
	//
	// Header layout:
	//   0..4     Magic
	//   4..6     Version
	//   6..8     PageType
	//   8..12    HeaderSize
	//   12..16   PageSize
	//   16..20   PayloadSize
	//   20..24   Flags
	//   24..32   PageIndex
	//   32..40   ModelOffset
	//   40..48   ModelSize
	//   48..56   ShardID
	//   56..64   ShardCount
	//   64..68   TensorID
	//   68..72   LayerID
	//   72..76   ShardID2
	//   76..80   QuantID
	//   80..112  PayloadHash BLAKE3-256
	//   112..144 PageHash / PageCommit BLAKE3-256
	//   144..160 Reserved
	//
	// PageHash is computed with bytes 112..144 zeroed.
	// This prevents self-referential page hashing.
	unifiedHeaderSize = uint64(160)

	hashSize       = 32
	pageHashOffset = 112
	pageHashEnd    = pageHashOffset + hashSize

	magicAIDG = "AIDG"

	manifestVersion = 2
	formatVersion   = uint16(2)

	hashAlgorithm = "BLAKE3-256"

	merkleVersion = 1

	sealVersion  = uint16(1)
	proofVersion = 1

	defaultSidecarExt     = ".sidecar"
	sidecarMagic          = "AIDGSC1!" // exactly 8 bytes
	sidecarVersion        = 1
	sidecarHeaderReserved = uint64(1 << 20) // 1 MiB reserved JSON header
)

const (
	PageTypeIndex  = uint16(0x0001)
	PageTypeModel  = uint16(0x0002)
	PageTypeFiller = uint16(0x0003)
)

const (
	treeAIDag  = "AIDG_MERKLE_TREE_AIDAG_V1"
	treeTensor = "AIDG_MERKLE_TREE_TENSOR_V1"
	treePoW    = "AIDG_MERKLE_TREE_POW_COMMIT_V1"

	domainHashBytes   = "AIDG_HASH_BYTES_V1"
	domainFileHash    = "AIDG_FILE_HASH_V1"
	domainPageCommit  = "AIDG_PAGE_COMMIT_V1"
	domainMerkleLeaf  = "AIDG_MERKLE_LEAF_V1"
	domainMerkleNode  = "AIDG_MERKLE_NODE_V1"
	domainMerkleEmpty = "AIDG_MERKLE_EMPTY_V1"
	domainFillerBlock = "AIDG_FILLER_BLOCK_V1"
	domainFillerMask  = "AIDG_FILLER_MASK_V1"
	domainManifest    = "AIDG_MANIFEST_ROOT_V1"

	domainSamplePow    = "AIDG_SAMPLE_POW_V1"
	domainSampleTensor = "AIDG_SAMPLE_TENSOR_V1"
	domainMixDigest    = "AIDG_MIX_DIGEST_V1"
	domainAIDigest     = "AIDG_AI_DIGEST_V1"
	domainProofHash    = "AIDG_PROOF_HASH_V1"
	domainSealWork     = "AIDG_SEAL_WORK_HASH_V1"
)

const (
	ProofSideLeft  = "left"
	ProofSideRight = "right"
)

type Config struct {
	Mode string

	OutPath     string
	DagPath     string
	MetaPath    string
	SidecarPath string
	ModelPath   string
	ExtractTo   string
	ProofPath   string

	SizeGB   uint64
	PageSize uint64
	Seed     string

	Workers int
	ChunkMB uint64

	Epoch         uint64
	Nonce         uint64
	BlockHash     string
	Miner         string
	Samples       int
	TensorSamples int
	Target        string

	Force  bool
	Verify bool
}

type Manifest struct {
	Version int    `json:"version"`
	Name    string `json:"name"`

	HashAlgorithm string `json:"hashAlgorithm"`
	MerkleVersion int    `json:"merkleVersion"`
	PageCommit    string `json:"pageCommit"`

	PageSize       uint64 `json:"pageSize"`
	HeaderSize     uint64 `json:"headerSize"`
	PayloadSize    uint64 `json:"payloadSize"`
	SizeGB         uint64 `json:"sizeGB"`
	TotalBytes     uint64 `json:"totalBytes"`
	TotalPages     uint64 `json:"totalPages"`
	Seed           string `json:"seed"`
	AIDagRoot      string `json:"aidagRoot"`
	TensorRoot     string `json:"tensorRoot"`
	PoWCommitRoot  string `json:"powCommitRoot"`
	ManifestRoot   string `json:"manifestRoot"`
	GenerationTime string `json:"generationTime"`

	AIDagLeafCount     uint64 `json:"aidagLeafCount"`
	TensorLeafCount    uint64 `json:"tensorLeafCount"`
	PoWCommitLeafCount uint64 `json:"powCommitLeafCount"`

	MerkleSidecar  string `json:"merkleSidecar"`
	SidecarVersion int    `json:"sidecarVersion"`

	ModelName      string `json:"modelName,omitempty"`
	ModelFormat    string `json:"modelFormat,omitempty"`
	ModelSize      uint64 `json:"modelSize,omitempty"`
	ModelHash      string `json:"modelHash,omitempty"`
	ModelStartPage uint64 `json:"modelStartPage,omitempty"`
	ModelPageCount uint64 `json:"modelPageCount,omitempty"`
	ModelEndPage   uint64 `json:"modelEndPage,omitempty"` // exclusive

	IndexPage     uint64 `json:"indexPage"`
	FillerStart   uint64 `json:"fillerStart,omitempty"`
	FillerEnd     uint64 `json:"fillerEnd,omitempty"` // exclusive
	UnifiedLayout string `json:"unifiedLayout"`
}

type PageHeader struct {
	Magic       [4]byte
	Version     uint16
	PageType    uint16
	HeaderSize  uint32
	PageSize    uint32
	PayloadSize uint32
	Flags       uint32

	PageIndex uint64

	// Model shard info.
	ModelOffset uint64
	ModelSize   uint64
	ShardID     uint64
	ShardCount  uint64

	// Reserved for future tensor-aware layout.
	TensorID uint32
	LayerID  uint32
	ShardID2 uint32
	QuantID  uint32

	PayloadHash [32]byte
	PageHash    [32]byte
	Reserved    [16]byte
}

type Job struct {
	PageIndex uint64
	PageType  uint16

	ModelOffset uint64
	ModelSize   uint64
	ShardID     uint64
	ShardCount  uint64
	Data        []byte
}

type Result struct {
	PageIndex uint64
	Data      []byte
	Err       error
}

type Roots struct {
	AIDagRoot     [32]byte
	TensorRoot    [32]byte
	PoWCommitRoot [32]byte

	AIDagLeafCount     uint64
	TensorLeafCount    uint64
	PoWCommitLeafCount uint64
}

type MerkleAccumulator struct {
	treeID string
	count  uint64
	stack  [64][32]byte
	filled [64]bool
}

type MerkleTree struct {
	treeID string
	levels [][][32]byte
}

type SidecarLevel struct {
	Level  int    `json:"level"`
	Count  uint64 `json:"count"`
	Offset uint64 `json:"offset"`
	Bytes  uint64 `json:"bytes"`
}

type SidecarManifest struct {
	Version       int    `json:"version"`
	HashAlgorithm string `json:"hashAlgorithm"`
	MerkleVersion int    `json:"merkleVersion"`
	CreatedAt     string `json:"createdAt"`

	PageSize       uint64 `json:"pageSize"`
	HeaderSize     uint64 `json:"headerSize"`
	PayloadSize    uint64 `json:"payloadSize"`
	TotalPages     uint64 `json:"totalPages"`
	ModelStartPage uint64 `json:"modelStartPage"`
	ModelPageCount uint64 `json:"modelPageCount"`

	ManifestRoot  string `json:"manifestRoot"`
	AIDagRoot     string `json:"aidagRoot"`
	TensorRoot    string `json:"tensorRoot"`
	PoWCommitRoot string `json:"powCommitRoot"`

	PoWLevels    []SidecarLevel `json:"powLevels"`
	TensorLevels []SidecarLevel `json:"tensorLevels"`
}

type MerkleSidecarReader struct {
	path string
	file *os.File
	meta SidecarManifest
}

type MerkleProofStep struct {
	Side string `json:"side"`
	Hash string `json:"hash"`
}

type AIDagPageProof struct {
	PageIndex  uint64            `json:"pageIndex"`
	Page       string            `json:"page"`
	PoWPath    []MerkleProofStep `json:"powPath"`
	TensorPath []MerkleProofStep `json:"tensorPath,omitempty"`
}

type AISeal struct {
	Version uint16 `json:"version"`
	Epoch   uint64 `json:"epoch"`

	BlockHash string `json:"blockHash"`
	Miner     string `json:"miner"`
	Nonce     uint64 `json:"nonce"`

	PowSampleCount    int `json:"powSampleCount"`
	TensorSampleCount int `json:"tensorSampleCount"`

	ManifestRoot string `json:"manifestRoot"`
	AIDagRoot    string `json:"aidagRoot"`
	TensorRoot   string `json:"tensorRoot"`
	PoWRoot      string `json:"powCommitRoot"`
	ModelHash    string `json:"modelHash"`

	MixDigest string `json:"mixDigest"`
	AIDigest  string `json:"aiDigest"`
	ProofHash string `json:"proofHash"`
	WorkHash  string `json:"workHash"`
}

type AISealProof struct {
	Version int              `json:"version"`
	Seal    AISeal           `json:"seal"`
	Pages   []AIDagPageProof `json:"pages"`
}

type ProofTrees struct {
	Sidecar *MerkleSidecarReader
}

type pageProofMaterial struct {
	PageIndex   uint64
	Header      PageHeader
	Page        []byte
	Payload     []byte
	PayloadHash [32]byte
	PageCommit  [32]byte
	PoWLeaf     [32]byte
	TensorLeaf  [32]byte
}

func main() {
	cfg := Config{}

	flag.StringVar(&cfg.Mode, "mode", "gen", "mode: gen, verify, extract, info, prove, verify-proof")
	flag.StringVar(&cfg.OutPath, "out", "./unified-aidag-128g.bin", "output unified AI-DAG path for gen mode")
	flag.StringVar(&cfg.DagPath, "dag", "", "AI-DAG path for verify/extract/info/prove mode")
	flag.StringVar(&cfg.MetaPath, "meta", "", "metadata path; default is <dag>.meta or <out>.meta")
	flag.StringVar(&cfg.SidecarPath, "sidecar", "", "Merkle sidecar tree path; default is <dag>.sidecar or <out>.sidecar")
	flag.StringVar(&cfg.ModelPath, "model", "", "model file path to embed into unified AI-DAG")
	flag.StringVar(&cfg.ExtractTo, "extract-out", "./extracted-model.bin", "output model path for extract mode")
	flag.StringVar(&cfg.ProofPath, "proof", "./aiseal-proof.json", "AISeal proof JSON path for prove/verify-proof mode")

	flag.Uint64Var(&cfg.SizeGB, "size-gb", 128, "AI-DAG size in GiB")
	flag.Uint64Var(&cfg.PageSize, "page-size", defaultPageSize, "AI-DAG page size")
	flag.StringVar(&cfg.Seed, "seed", "colossusx-unified-aidag-v1", "deterministic seed")
	flag.IntVar(&cfg.Workers, "workers", runtime.NumCPU(), "worker count")
	flag.Uint64Var(&cfg.ChunkMB, "chunk-mb", 64, "model read chunk size in MiB")

	flag.Uint64Var(&cfg.Epoch, "epoch", 0, "AI-DAG epoch for AISeal proof")
	flag.Uint64Var(&cfg.Nonce, "nonce", 0, "nonce for AISeal proof")
	flag.StringVar(&cfg.BlockHash, "block-hash", "", "previous block/header hash used as AISeal challenge seed")
	flag.StringVar(&cfg.Miner, "miner", "", "miner address or miner id used as AISeal challenge seed")
	flag.IntVar(&cfg.Samples, "samples", 64, "PoW page sample count for AISeal proof")
	flag.IntVar(&cfg.TensorSamples, "tensor-samples", 8, "model/tensor page sample count for AISeal proof")
	flag.StringVar(&cfg.Target, "target", "", "optional 32-byte target hex; verify workHash <= target")

	flag.BoolVar(&cfg.Force, "force", false, "overwrite output")
	flag.BoolVar(&cfg.Verify, "verify", false, "verify after generation")
	flag.Parse()

	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(cfg Config) error {
	switch cfg.Mode {
	case "gen":
		return runGenerate(cfg)
	case "verify":
		return runVerify(cfg)
	case "extract":
		return runExtract(cfg)
	case "info":
		return runInfo(cfg)
	case "prove":
		return runProve(cfg)
	case "verify-proof":
		return runVerifyProof(cfg)
	default:
		return fmt.Errorf("unknown mode: %s", cfg.Mode)
	}
}

func runGenerate(cfg Config) error {
	if err := validateGenerateConfig(&cfg); err != nil {
		return err
	}

	if cfg.MetaPath == "" {
		cfg.MetaPath = cfg.OutPath + ".meta"
	}
	if cfg.SidecarPath == "" {
		cfg.SidecarPath = cfg.OutPath + defaultSidecarExt
	}

	totalBytes := cfg.SizeGB * oneGiB
	totalPages := totalBytes / cfg.PageSize
	payloadSize := cfg.PageSize - unifiedHeaderSize

	if totalPages > uint64(maxInt()) {
		return fmt.Errorf("too many pages for strict page tracking: totalPages=%d", totalPages)
	}

	modelHash, modelSize, err := hashFile(cfg.ModelPath)
	if err != nil {
		return fmt.Errorf("hash model: %w", err)
	}

	modelPageCount := roundUp(modelSize, payloadSize) / payloadSize
	modelStartPage := uint64(1)
	modelEndPage := modelStartPage + modelPageCount

	if modelEndPage > totalPages {
		return fmt.Errorf("model does not fit: modelEndPage=%d totalPages=%d", modelEndPage, totalPages)
	}

	manifest := Manifest{
		Version:       manifestVersion,
		Name:          "ColossusX Unified PoW+AI-DAG v1",
		HashAlgorithm: hashAlgorithm,
		MerkleVersion: merkleVersion,
		PageCommit:    "BLAKE3-256(page with PageHash bytes 112..144 zeroed)",

		PageSize:    cfg.PageSize,
		HeaderSize:  unifiedHeaderSize,
		PayloadSize: payloadSize,
		SizeGB:      cfg.SizeGB,
		TotalBytes:  totalBytes,
		TotalPages:  totalPages,
		Seed:        cfg.Seed,

		AIDagLeafCount:     totalPages,
		TensorLeafCount:    modelPageCount,
		PoWCommitLeafCount: totalPages,

		SidecarVersion: sidecarVersion,
		MerkleSidecar:  cfg.SidecarPath,

		ModelSize:      modelSize,
		ModelHash:      "0x" + fmtHash(modelHash),
		ModelStartPage: modelStartPage,
		ModelPageCount: modelPageCount,
		ModelEndPage:   modelEndPage,

		IndexPage:   0,
		FillerStart: modelEndPage,
		FillerEnd:   totalPages,
	}

	fmt.Println("===== ColossusX Unified PoW+AI-DAG v1 Generator =====")
	fmt.Printf("output       : %s\n", cfg.OutPath)
	fmt.Printf("metadata     : %s\n", cfg.MetaPath)
	fmt.Printf("sidecar      : %s\n", cfg.SidecarPath)
	fmt.Printf("hash         : %s\n", manifest.HashAlgorithm)
	fmt.Printf("merkle       : version %d\n", manifest.MerkleVersion)
	fmt.Printf("size         : %d GiB\n", cfg.SizeGB)
	fmt.Printf("page size    : %d\n", cfg.PageSize)
	fmt.Printf("header size  : %d\n", unifiedHeaderSize)
	fmt.Printf("payload size : %d\n", payloadSize)
	fmt.Printf("total pages  : %d\n", totalPages)
	fmt.Printf("model        : %s\n", cfg.ModelPath)
	fmt.Printf("model size   : %d bytes\n", modelSize)
	fmt.Printf("model hash   : 0x%s\n", fmtHash(modelHash))
	fmt.Printf("model pages  : %d..%d exclusive\n", modelStartPage, modelEndPage)
	fmt.Printf("filler pages : %d..%d exclusive\n", modelEndPage, totalPages)
	fmt.Printf("workers      : %d\n", cfg.Workers)
	fmt.Println("=====================================================")

	if err := preflightAtomicTargets(cfg.Force, cfg.OutPath, cfg.MetaPath, cfg.SidecarPath); err != nil {
		return err
	}

	if err := ensureParentDir(cfg.OutPath); err != nil {
		return err
	}
	if err := ensureParentDir(cfg.MetaPath); err != nil {
		return err
	}
	if err := ensureParentDir(cfg.SidecarPath); err != nil {
		return err
	}

	tmpDagPath := tempPathFor(cfg.OutPath)
	tmpSidecarPath := tempPathFor(cfg.SidecarPath)
	defer os.Remove(tmpDagPath)
	defer os.Remove(tmpSidecarPath)

	out, err := os.OpenFile(tmpDagPath, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0644)
	if err != nil {
		return err
	}

	if err := out.Truncate(int64(totalBytes)); err != nil {
		_ = out.Close()
		return fmt.Errorf("truncate: %w", err)
	}

	start := time.Now()
	var written uint64
	var writtenPages uint64
	seenPages := make([]uint8, int(totalPages))

	jobs := make(chan Job, cfg.Workers*8)
	results := make(chan Result, cfg.Workers*8)

	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go recoveredPageWorker(&wg, jobs, results, cfg, manifest)
	}

	producerErr := make(chan error, 1)
	go func() {
		defer close(producerErr)

		jobs <- Job{PageIndex: 0, PageType: PageTypeIndex}

		if err := enqueueModelJobs(jobs, cfg, manifest); err != nil {
			producerErr <- err
			close(jobs)
			wg.Wait()
			close(results)
			return
		}

		for p := manifest.FillerStart; p < manifest.FillerEnd; p++ {
			jobs <- Job{PageIndex: p, PageType: PageTypeFiller}
		}

		close(jobs)
		wg.Wait()
		close(results)
	}()

	progressDone := make(chan struct{})
	go printProgressLoop(progressDone, start, &written, totalBytes)

	var firstErr error
	for res := range results {
		if res.Err != nil {
			if firstErr == nil {
				firstErr = res.Err
			}
			continue
		}

		if res.PageIndex >= totalPages {
			if firstErr == nil {
				firstErr = fmt.Errorf("worker returned out-of-range page: %d >= %d", res.PageIndex, totalPages)
			}
			continue
		}
		if seenPages[int(res.PageIndex)] != 0 {
			if firstErr == nil {
				firstErr = fmt.Errorf("duplicate page result: %d", res.PageIndex)
			}
			continue
		}

		if err := strictValidateBuiltPage(res.Data, res.PageIndex, cfg.PageSize); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}

		offset := res.PageIndex * cfg.PageSize
		n, err := out.WriteAt(res.Data, int64(offset))
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		if uint64(n) != cfg.PageSize {
			if firstErr == nil {
				firstErr = io.ErrShortWrite
			}
			continue
		}

		seenPages[int(res.PageIndex)] = 1
		atomic.AddUint64(&written, cfg.PageSize)
		atomic.AddUint64(&writtenPages, 1)
	}
	close(progressDone)

	if err := <-producerErr; err != nil {
		_ = out.Close()
		return err
	}
	if firstErr != nil {
		_ = out.Close()
		return firstErr
	}
	if writtenPages != totalPages {
		_ = out.Close()
		missing := firstMissingPage(seenPages)
		return fmt.Errorf("strict page commit failed: writtenPages=%d totalPages=%d firstMissing=%d", writtenPages, totalPages, missing)
	}
	if written != totalBytes {
		_ = out.Close()
		return fmt.Errorf("strict byte commit failed: written=%d total=%d", written, totalBytes)
	}

	if err := out.Sync(); err != nil {
		_ = out.Close()
		return fmt.Errorf("sync dag: %w", err)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("close dag: %w", err)
	}

	fmt.Println()
	fmt.Println("computing BLAKE3 Merkle roots from generated AI-DAG...")
	roots, err := computeRootsFromFile(tmpDagPath, cfg.PageSize)
	if err != nil {
		return err
	}
	if err := validateRootsNonEmpty(roots, manifest); err != nil {
		return err
	}

	manifest.AIDagRoot = "0x" + fmtHash(roots.AIDagRoot)
	manifest.TensorRoot = "0x" + fmtHash(roots.TensorRoot)
	manifest.PoWCommitRoot = "0x" + fmtHash(roots.PoWCommitRoot)

	manifest.AIDagLeafCount = roots.AIDagLeafCount
	manifest.TensorLeafCount = roots.TensorLeafCount
	manifest.PoWCommitLeafCount = roots.PoWCommitLeafCount

	manifest.ManifestRoot = "0x" + fmtHash(hashManifestRoot(manifest))

	// Local-only metadata is intentionally added after root calculation.
	manifest.GenerationTime = time.Now().UTC().Format(time.RFC3339)
	manifest.ModelName = filepath.Base(cfg.ModelPath)
	manifest.ModelFormat = guessModelFormat(cfg.ModelPath)
	manifest.UnifiedLayout = "page0=index; page1..modelEnd=model shards; remaining pages=deterministic filler; every page is PoW-mixable; roots are BLAKE3 Merkle roots; PoW/Tensor proof paths are stored in mandatory Merkle sidecar"

	if err := validateManifestBasics(manifest); err != nil {
		return err
	}

	fmt.Println("building mandatory Merkle sidecar tree...")
	if _, err := buildMerkleSidecarTempFromDAG(tmpDagPath, tmpSidecarPath, manifest); err != nil {
		return err
	}

	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}

	if err := commitAtomicPath(tmpDagPath, cfg.OutPath, cfg.Force); err != nil {
		return fmt.Errorf("commit dag: %w", err)
	}
	if err := commitAtomicPath(tmpSidecarPath, cfg.SidecarPath, cfg.Force); err != nil {
		return fmt.Errorf("commit sidecar: %w", err)
	}
	if err := writeFileAtomic(cfg.MetaPath, manifestBytes, 0644, cfg.Force); err != nil {
		return fmt.Errorf("commit manifest: %w", err)
	}

	fmt.Println("Unified AI-DAG generated successfully")
	fmt.Printf("AIDagRoot          : %s\n", manifest.AIDagRoot)
	fmt.Printf("TensorRoot         : %s\n", manifest.TensorRoot)
	fmt.Printf("PoWCommitRoot      : %s\n", manifest.PoWCommitRoot)
	fmt.Printf("ManifestRoot       : %s\n", manifest.ManifestRoot)
	fmt.Printf("AIDagLeafCount     : %d\n", manifest.AIDagLeafCount)
	fmt.Printf("TensorLeafCount    : %d\n", manifest.TensorLeafCount)
	fmt.Printf("PoWCommitLeafCount : %d\n", manifest.PoWCommitLeafCount)
	fmt.Printf("output             : %s\n", cfg.OutPath)
	fmt.Printf("metadata           : %s\n", cfg.MetaPath)
	fmt.Printf("sidecar            : %s\n", cfg.SidecarPath)

	if cfg.Verify {
		return verifyDagAgainstManifest(cfg.OutPath, cfg.MetaPath, cfg.SidecarPath)
	}

	return nil
}

func validateGenerateConfig(cfg *Config) error {
	if cfg.OutPath == "" {
		return errors.New("--out is required")
	}
	if cfg.ModelPath == "" {
		return errors.New("--model is required for unified AI-DAG generation")
	}
	if cfg.PageSize < 1024 {
		return errors.New("page size too small")
	}
	if cfg.PageSize%64 != 0 {
		return errors.New("page size must be multiple of 64")
	}
	if cfg.PageSize <= unifiedHeaderSize {
		return errors.New("page size must be larger than unified header size")
	}
	if cfg.SizeGB == 0 {
		return errors.New("--size-gb must be > 0")
	}
	totalBytes := cfg.SizeGB * oneGiB
	if totalBytes%cfg.PageSize != 0 {
		return errors.New("total size must be divisible by page size")
	}
	if cfg.Workers <= 0 {
		cfg.Workers = runtime.NumCPU()
	}
	if cfg.ChunkMB == 0 {
		cfg.ChunkMB = 64
	}
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return fmt.Errorf("model not found: %s", cfg.ModelPath)
	}
	return nil
}

func recoveredPageWorker(wg *sync.WaitGroup, jobs <-chan Job, results chan<- Result, cfg Config, manifest Manifest) {
	defer wg.Done()

	for job := range jobs {
		func(job Job) {
			defer func() {
				if r := recover(); r != nil {
					results <- Result{
						PageIndex: job.PageIndex,
						Err:       fmt.Errorf("worker panic while building page %d: %v\n%s", job.PageIndex, r, debug.Stack()),
					}
				}
			}()

			page, err := buildUnifiedPage(cfg, manifest, job)
			results <- Result{PageIndex: job.PageIndex, Data: page, Err: err}
		}(job)
	}
}

func enqueueModelJobs(jobs chan<- Job, cfg Config, manifest Manifest) error {
	f, err := os.Open(cfg.ModelPath)
	if err != nil {
		return err
	}
	defer f.Close()

	payloadSize := manifest.PayloadSize
	buf := make([]byte, payloadSize)

	var modelOffset uint64
	var shardID uint64

	for modelOffset < manifest.ModelSize {
		clear(buf)

		remaining := manifest.ModelSize - modelOffset
		readSize := payloadSize
		if remaining < readSize {
			readSize = remaining
		}

		n, err := io.ReadFull(f, buf[:readSize])
		if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
			return err
		}
		if uint64(n) != readSize {
			return fmt.Errorf("short model read: got=%d want=%d", n, readSize)
		}

		payload := make([]byte, payloadSize)
		copy(payload, buf)

		jobs <- Job{
			PageIndex:   manifest.ModelStartPage + shardID,
			PageType:    PageTypeModel,
			ModelOffset: modelOffset,
			ModelSize:   readSize,
			ShardID:     shardID,
			ShardCount:  manifest.ModelPageCount,
			Data:        payload,
		}

		modelOffset += readSize
		shardID++
	}

	return nil
}

func buildUnifiedPage(cfg Config, manifest Manifest, job Job) ([]byte, error) {
	page := make([]byte, cfg.PageSize)
	payload := page[unifiedHeaderSize:]

	switch job.PageType {
	case PageTypeIndex:
		indexPayload, err := buildIndexPayload(manifest, len(payload))
		if err != nil {
			return nil, err
		}
		copy(payload, indexPayload)

	case PageTypeModel:
		if uint64(len(job.Data)) != manifest.PayloadSize {
			return nil, fmt.Errorf("invalid model payload size: got=%d want=%d", len(job.Data), manifest.PayloadSize)
		}
		copy(payload, job.Data)

	case PageTypeFiller:
		fillDeterministicPayload(payload, cfg.Seed, job.PageIndex)

	default:
		return nil, fmt.Errorf("unknown page type: %d", job.PageType)
	}

	payloadHash := hashBytes(payload)

	header := PageHeader{}
	copy(header.Magic[:], []byte(magicAIDG))
	header.Version = formatVersion
	header.PageType = job.PageType
	header.HeaderSize = uint32(unifiedHeaderSize)
	header.PageSize = uint32(cfg.PageSize)
	header.PayloadSize = uint32(len(payload))
	header.PageIndex = job.PageIndex
	header.ModelOffset = job.ModelOffset
	header.ModelSize = job.ModelSize
	header.ShardID = job.ShardID
	header.ShardCount = job.ShardCount
	header.PayloadHash = payloadHash

	// Write header with PageHash = zero.
	writeHeader(page[:unifiedHeaderSize], header)

	// Compute stable page commitment with PageHash bytes zeroed.
	pageCommit := pageCommitHash(page)
	header.PageHash = pageCommit

	// Write final header.
	writeHeader(page[:unifiedHeaderSize], header)

	if err := strictValidateBuiltPage(page, job.PageIndex, cfg.PageSize); err != nil {
		return nil, err
	}

	return page, nil
}

func strictValidateBuiltPage(page []byte, pageIndex uint64, pageSize uint64) error {
	hdr, payloadHash, pageCommit, err := verifyPageBytes(page, pageIndex, pageSize)
	if err != nil {
		return err
	}
	if isZeroHash(payloadHash) {
		return fmt.Errorf("strict page commit failed: zero payload hash at page %d", pageIndex)
	}
	if isZeroHash(pageCommit) {
		return fmt.Errorf("strict page commit failed: zero page commit at page %d", pageIndex)
	}
	if hdr.PayloadHash != payloadHash {
		return fmt.Errorf("strict page commit failed: header payload hash mismatch at page %d", pageIndex)
	}
	if hdr.PageHash != pageCommit {
		return fmt.Errorf("strict page commit failed: header page commit mismatch at page %d", pageIndex)
	}
	return nil
}

func buildIndexPayload(manifest Manifest, max int) ([]byte, error) {
	tmp := canonicalManifestForIndex(manifest)

	b, err := json.Marshal(tmp)
	if err != nil {
		return nil, err
	}
	if len(b) > max {
		return nil, fmt.Errorf("manifest too large for index payload: %d > %d", len(b), max)
	}

	out := make([]byte, max)
	copy(out, b)
	return out, nil
}

func canonicalManifestForIndex(m Manifest) Manifest {
	m.AIDagRoot = ""
	m.TensorRoot = ""
	m.PoWCommitRoot = ""
	m.ManifestRoot = ""
	m.GenerationTime = ""
	m.ModelName = ""
	m.ModelFormat = ""
	m.UnifiedLayout = ""
	m.MerkleSidecar = ""
	m.SidecarVersion = 0
	return m
}

func canonicalManifestForRoot(m Manifest) Manifest {
	m.ManifestRoot = ""
	m.GenerationTime = ""
	m.ModelName = ""
	m.ModelFormat = ""
	m.UnifiedLayout = ""
	return m
}

func writeHeader(dst []byte, h PageHeader) {
	clear(dst)

	copy(dst[0:4], h.Magic[:])
	binary.LittleEndian.PutUint16(dst[4:6], h.Version)
	binary.LittleEndian.PutUint16(dst[6:8], h.PageType)
	binary.LittleEndian.PutUint32(dst[8:12], h.HeaderSize)
	binary.LittleEndian.PutUint32(dst[12:16], h.PageSize)
	binary.LittleEndian.PutUint32(dst[16:20], h.PayloadSize)
	binary.LittleEndian.PutUint32(dst[20:24], h.Flags)

	binary.LittleEndian.PutUint64(dst[24:32], h.PageIndex)

	binary.LittleEndian.PutUint64(dst[32:40], h.ModelOffset)
	binary.LittleEndian.PutUint64(dst[40:48], h.ModelSize)
	binary.LittleEndian.PutUint64(dst[48:56], h.ShardID)
	binary.LittleEndian.PutUint64(dst[56:64], h.ShardCount)

	binary.LittleEndian.PutUint32(dst[64:68], h.TensorID)
	binary.LittleEndian.PutUint32(dst[68:72], h.LayerID)
	binary.LittleEndian.PutUint32(dst[72:76], h.ShardID2)
	binary.LittleEndian.PutUint32(dst[76:80], h.QuantID)

	copy(dst[80:112], h.PayloadHash[:])
	copy(dst[112:144], h.PageHash[:])
	copy(dst[144:160], h.Reserved[:])
}

func readHeader(src []byte) (PageHeader, error) {
	var h PageHeader

	if len(src) < int(unifiedHeaderSize) {
		return h, errors.New("page too small")
	}

	copy(h.Magic[:], src[0:4])
	if string(h.Magic[:]) != magicAIDG {
		return h, errors.New("bad page magic")
	}

	h.Version = binary.LittleEndian.Uint16(src[4:6])
	h.PageType = binary.LittleEndian.Uint16(src[6:8])
	h.HeaderSize = binary.LittleEndian.Uint32(src[8:12])
	h.PageSize = binary.LittleEndian.Uint32(src[12:16])
	h.PayloadSize = binary.LittleEndian.Uint32(src[16:20])
	h.Flags = binary.LittleEndian.Uint32(src[20:24])
	h.PageIndex = binary.LittleEndian.Uint64(src[24:32])

	h.ModelOffset = binary.LittleEndian.Uint64(src[32:40])
	h.ModelSize = binary.LittleEndian.Uint64(src[40:48])
	h.ShardID = binary.LittleEndian.Uint64(src[48:56])
	h.ShardCount = binary.LittleEndian.Uint64(src[56:64])

	h.TensorID = binary.LittleEndian.Uint32(src[64:68])
	h.LayerID = binary.LittleEndian.Uint32(src[68:72])
	h.ShardID2 = binary.LittleEndian.Uint32(src[72:76])
	h.QuantID = binary.LittleEndian.Uint32(src[76:80])

	copy(h.PayloadHash[:], src[80:112])
	copy(h.PageHash[:], src[112:144])
	copy(h.Reserved[:], src[144:160])

	return h, nil
}

func validatePageHeader(h PageHeader, actualPageIndex uint64, pageSize uint64) error {
	if string(h.Magic[:]) != magicAIDG {
		return errors.New("bad page magic")
	}
	if h.Version != formatVersion {
		return fmt.Errorf("unsupported page version: got=%d want=%d", h.Version, formatVersion)
	}
	if uint64(h.HeaderSize) != unifiedHeaderSize {
		return fmt.Errorf("bad header size: got=%d want=%d", h.HeaderSize, unifiedHeaderSize)
	}
	if uint64(h.PageSize) != pageSize {
		return fmt.Errorf("bad page size: got=%d want=%d", h.PageSize, pageSize)
	}
	if uint64(h.PayloadSize) != pageSize-unifiedHeaderSize {
		return fmt.Errorf("bad payload size: got=%d want=%d", h.PayloadSize, pageSize-unifiedHeaderSize)
	}
	if h.PageIndex != actualPageIndex {
		return fmt.Errorf("page index mismatch: header=%d actual=%d", h.PageIndex, actualPageIndex)
	}
	switch h.PageType {
	case PageTypeIndex, PageTypeModel, PageTypeFiller:
		return nil
	default:
		return fmt.Errorf("unknown page type: %d", h.PageType)
	}
}

func fillDeterministicPayload(dst []byte, seed string, pageIndex uint64) {
	var counter uint64

	for off := 0; off < len(dst); off += hashSize {
		sum := blake3HashTagged(domainFillerBlock, func(w io.Writer) {
			writeString(w, seed)
			writeUint64(w, pageIndex)
			writeUint64(w, counter)
		})

		n := copy(dst[off:], sum[:])
		if n < hashSize {
			break
		}
		counter++
	}

	mask := blake3HashTagged(domainFillerMask, func(w io.Writer) {
		writeBytes(w, dst)
	})

	for i := range dst {
		dst[i] ^= mask[i%hashSize]
	}
}

func computeRootsFromFile(path string, pageSize uint64) (Roots, error) {
	f, err := os.Open(path)
	if err != nil {
		return Roots{}, err
	}
	defer f.Close()

	page := make([]byte, pageSize)
	var pageIndex uint64

	aidag := NewMerkleAccumulator(treeAIDag)
	tensor := NewMerkleAccumulator(treeTensor)
	powc := NewMerkleAccumulator(treePoW)

	for {
		n, err := io.ReadFull(f, page)
		if err == io.EOF {
			break
		}
		if err == io.ErrUnexpectedEOF {
			return Roots{}, errors.New("dag size is not page aligned")
		}
		if err != nil {
			return Roots{}, err
		}
		if uint64(n) != pageSize {
			return Roots{}, io.ErrShortBuffer
		}

		hdr, payloadHash, pageCommit, err := verifyPageBytes(page, pageIndex, pageSize)
		if err != nil {
			return Roots{}, err
		}

		aidagLeaf := merkleLeafHash(treeAIDag, hdr, payloadHash, pageCommit)
		aidag.AddLeaf(aidagLeaf)

		if hdr.PageType == PageTypeModel {
			tensorLeaf := merkleLeafHash(treeTensor, hdr, payloadHash, pageCommit)
			tensor.AddLeaf(tensorLeaf)
		}

		powLeaf := merkleLeafHash(treePoW, hdr, payloadHash, pageCommit)
		powc.AddLeaf(powLeaf)

		pageIndex++
		if pageIndex%1000000 == 0 {
			fmt.Printf("  root progress: page %d\n", pageIndex)
		}
	}

	return Roots{
		AIDagRoot:          aidag.Root(),
		TensorRoot:         tensor.Root(),
		PoWCommitRoot:      powc.Root(),
		AIDagLeafCount:     aidag.Count(),
		TensorLeafCount:    tensor.Count(),
		PoWCommitLeafCount: powc.Count(),
	}, nil
}

func verifyPageBytes(page []byte, pageIndex uint64, pageSize uint64) (PageHeader, [32]byte, [32]byte, error) {
	var zero [32]byte

	if uint64(len(page)) != pageSize {
		return PageHeader{}, zero, zero, fmt.Errorf("bad page buffer size at page %d: got=%d want=%d", pageIndex, len(page), pageSize)
	}
	if pageSize < unifiedHeaderSize {
		return PageHeader{}, zero, zero, fmt.Errorf("bad page size: %d", pageSize)
	}

	hdr, err := readHeader(page[:unifiedHeaderSize])
	if err != nil {
		return PageHeader{}, zero, zero, fmt.Errorf("bad page %d: %w", pageIndex, err)
	}
	if err := validatePageHeader(hdr, pageIndex, pageSize); err != nil {
		return PageHeader{}, zero, zero, fmt.Errorf("bad page %d: %w", pageIndex, err)
	}

	payload := page[unifiedHeaderSize:]
	payloadHash := hashBytes(payload)
	if payloadHash != hdr.PayloadHash {
		return PageHeader{}, zero, zero, fmt.Errorf("payload hash mismatch at page %d", pageIndex)
	}

	pageCommit := pageCommitHash(page)
	if pageCommit != hdr.PageHash {
		return PageHeader{}, zero, zero, fmt.Errorf("page commit mismatch at page %d", pageIndex)
	}

	return hdr, payloadHash, pageCommit, nil
}

func NewMerkleAccumulator(treeID string) *MerkleAccumulator {
	return &MerkleAccumulator{treeID: treeID}
}

func (m *MerkleAccumulator) Count() uint64 { return m.count }

func (m *MerkleAccumulator) AddLeaf(leaf [32]byte) {
	node := leaf
	c := m.count

	level := 0
	for c&1 == 1 {
		left := m.stack[level]
		node = merkleNodeHash(m.treeID, left, node)
		m.filled[level] = false

		c >>= 1
		level++
	}

	m.stack[level] = node
	m.filled[level] = true
	m.count++
}

func (m *MerkleAccumulator) Root() [32]byte {
	if m.count == 0 {
		return merkleEmptyHash(m.treeID)
	}

	var root [32]byte
	have := false

	for level := 0; level < len(m.stack); level++ {
		if !m.filled[level] {
			continue
		}

		if !have {
			root = m.stack[level]
			have = true
			continue
		}

		root = merkleNodeHash(m.treeID, m.stack[level], root)
	}

	return root
}

func NewMerkleTree(treeID string, leaves [][32]byte) *MerkleTree {
	levels := make([][][32]byte, 0, 64)
	if len(leaves) == 0 {
		return &MerkleTree{treeID: treeID, levels: levels}
	}

	cur := make([][32]byte, len(leaves))
	copy(cur, leaves)
	levels = append(levels, cur)

	for len(cur) > 1 {
		next := make([][32]byte, (len(cur)+1)/2)
		for i := 0; i < len(cur); i += 2 {
			if i+1 < len(cur) {
				next[i/2] = merkleNodeHash(treeID, cur[i], cur[i+1])
			} else {
				next[i/2] = cur[i]
			}
		}
		levels = append(levels, next)
		cur = next
	}

	return &MerkleTree{treeID: treeID, levels: levels}
}

func (t *MerkleTree) Count() uint64 {
	if len(t.levels) == 0 {
		return 0
	}
	return uint64(len(t.levels[0]))
}

func (t *MerkleTree) Root() [32]byte {
	if len(t.levels) == 0 {
		return merkleEmptyHash(t.treeID)
	}
	return t.levels[len(t.levels)-1][0]
}

func (t *MerkleTree) Proof(leafIndex uint64) ([]MerkleProofStep, error) {
	if len(t.levels) == 0 {
		return nil, errors.New("cannot prove empty tree")
	}
	if leafIndex >= uint64(len(t.levels[0])) {
		return nil, fmt.Errorf("leaf index out of range: %d >= %d", leafIndex, len(t.levels[0]))
	}

	idx := int(leafIndex)
	proof := make([]MerkleProofStep, 0, len(t.levels)-1)

	for level := 0; level < len(t.levels)-1; level++ {
		nodes := t.levels[level]
		if idx%2 == 0 {
			if idx+1 < len(nodes) {
				proof = append(proof, MerkleProofStep{Side: ProofSideRight, Hash: "0x" + fmtHash(nodes[idx+1])})
			}
		} else {
			proof = append(proof, MerkleProofStep{Side: ProofSideLeft, Hash: "0x" + fmtHash(nodes[idx-1])})
		}
		idx /= 2
	}

	return proof, nil
}

func verifyMerkleProof(treeID string, leaf [32]byte, leafIndex uint64, leafCount uint64, proof []MerkleProofStep, expectedRoot [32]byte) error {
	if leafCount == 0 {
		return errors.New("cannot verify proof for empty tree")
	}
	if leafIndex >= leafCount {
		return fmt.Errorf("leaf index out of range: %d >= %d", leafIndex, leafCount)
	}

	node := leaf
	idx := leafIndex
	count := leafCount
	proofPos := 0

	for count > 1 {
		if idx%2 == 0 {
			if idx+1 < count {
				if proofPos >= len(proof) {
					return errors.New("merkle proof too short")
				}
				step := proof[proofPos]
				proofPos++
				if step.Side != ProofSideRight {
					return fmt.Errorf("bad merkle side at proof step %d: got=%s want=%s", proofPos-1, step.Side, ProofSideRight)
				}
				sibling, err := decodeHex32(step.Hash)
				if err != nil {
					return fmt.Errorf("bad merkle sibling at proof step %d: %w", proofPos-1, err)
				}
				node = merkleNodeHash(treeID, node, sibling)
			}
		} else {
			if proofPos >= len(proof) {
				return errors.New("merkle proof too short")
			}
			step := proof[proofPos]
			proofPos++
			if step.Side != ProofSideLeft {
				return fmt.Errorf("bad merkle side at proof step %d: got=%s want=%s", proofPos-1, step.Side, ProofSideLeft)
			}
			sibling, err := decodeHex32(step.Hash)
			if err != nil {
				return fmt.Errorf("bad merkle sibling at proof step %d: %w", proofPos-1, err)
			}
			node = merkleNodeHash(treeID, sibling, node)
		}

		idx /= 2
		count = (count + 1) / 2
	}

	if proofPos != len(proof) {
		return fmt.Errorf("merkle proof has unused steps: used=%d total=%d", proofPos, len(proof))
	}
	if node != expectedRoot {
		return fmt.Errorf("merkle root mismatch: got=0x%s want=0x%s", fmtHash(node), fmtHash(expectedRoot))
	}
	return nil
}

func merkleLeafHash(treeID string, hdr PageHeader, payloadHash [32]byte, pageCommit [32]byte) [32]byte {
	return blake3HashTagged(domainMerkleLeaf, func(w io.Writer) {
		writeString(w, treeID)

		writeUint16(w, hdr.Version)
		writeUint16(w, hdr.PageType)
		writeUint32(w, hdr.HeaderSize)
		writeUint32(w, hdr.PageSize)
		writeUint32(w, hdr.PayloadSize)
		writeUint32(w, hdr.Flags)

		writeUint64(w, hdr.PageIndex)

		writeUint64(w, hdr.ModelOffset)
		writeUint64(w, hdr.ModelSize)
		writeUint64(w, hdr.ShardID)
		writeUint64(w, hdr.ShardCount)

		writeUint32(w, hdr.TensorID)
		writeUint32(w, hdr.LayerID)
		writeUint32(w, hdr.ShardID2)
		writeUint32(w, hdr.QuantID)

		writeFixed32(w, payloadHash)
		writeFixed32(w, pageCommit)
	})
}

func merkleNodeHash(treeID string, left [32]byte, right [32]byte) [32]byte {
	return blake3HashTagged(domainMerkleNode, func(w io.Writer) {
		writeString(w, treeID)
		writeFixed32(w, left)
		writeFixed32(w, right)
	})
}

func merkleEmptyHash(treeID string) [32]byte {
	return blake3HashTagged(domainMerkleEmpty, func(w io.Writer) {
		writeString(w, treeID)
	})
}

func runVerify(cfg Config) error {
	dag := cfg.DagPath
	if dag == "" {
		dag = cfg.OutPath
	}

	meta := cfg.MetaPath
	if meta == "" {
		meta = dag + ".meta"
	}

	sidecar := cfg.SidecarPath
	if sidecar == "" {
		sidecar = dag + defaultSidecarExt
	}

	return verifyDagAgainstManifest(dag, meta, sidecar)
}

func verifyDagAgainstManifest(dagPath, metaPath, sidecarPath string) error {
	manifest, err := readManifest(metaPath)
	if err != nil {
		return err
	}

	if err := validateManifestBasics(manifest); err != nil {
		return err
	}

	gotManifestRoot := "0x" + fmtHash(hashManifestRoot(manifest))
	if !equalHexString(manifest.ManifestRoot, gotManifestRoot) {
		return fmt.Errorf("ManifestRoot mismatch: meta=%s got=%s", manifest.ManifestRoot, gotManifestRoot)
	}

	roots, err := computeRootsFromFile(dagPath, manifest.PageSize)
	if err != nil {
		return err
	}
	if err := validateRootsNonEmpty(roots, manifest); err != nil {
		return err
	}

	gotAIDag := "0x" + fmtHash(roots.AIDagRoot)
	gotTensor := "0x" + fmtHash(roots.TensorRoot)
	gotPoW := "0x" + fmtHash(roots.PoWCommitRoot)

	if !equalHexString(manifest.AIDagRoot, gotAIDag) {
		return fmt.Errorf("AIDagRoot mismatch: meta=%s got=%s", manifest.AIDagRoot, gotAIDag)
	}
	if !equalHexString(manifest.TensorRoot, gotTensor) {
		return fmt.Errorf("TensorRoot mismatch: meta=%s got=%s", manifest.TensorRoot, gotTensor)
	}
	if !equalHexString(manifest.PoWCommitRoot, gotPoW) {
		return fmt.Errorf("PoWCommitRoot mismatch: meta=%s got=%s", manifest.PoWCommitRoot, gotPoW)
	}

	if manifest.AIDagLeafCount != roots.AIDagLeafCount {
		return fmt.Errorf("AIDagLeafCount mismatch: meta=%d got=%d", manifest.AIDagLeafCount, roots.AIDagLeafCount)
	}
	if manifest.TensorLeafCount != roots.TensorLeafCount {
		return fmt.Errorf("TensorLeafCount mismatch: meta=%d got=%d", manifest.TensorLeafCount, roots.TensorLeafCount)
	}
	if manifest.PoWCommitLeafCount != roots.PoWCommitLeafCount {
		return fmt.Errorf("PoWCommitLeafCount mismatch: meta=%d got=%d", manifest.PoWCommitLeafCount, roots.PoWCommitLeafCount)
	}

	if sidecarPath == "" {
		return errors.New("sidecar path is required in production verify mode")
	}
	sidecar, err := openMerkleSidecar(sidecarPath)
	if err != nil {
		return err
	}
	defer sidecar.Close()
	if err := sidecar.validateAgainstManifest(manifest); err != nil {
		return err
	}

	fmt.Println("verify OK")
	fmt.Printf("AIDagRoot          : %s\n", gotAIDag)
	fmt.Printf("TensorRoot         : %s\n", gotTensor)
	fmt.Printf("PoWCommitRoot      : %s\n", gotPoW)
	fmt.Printf("AIDagLeafCount     : %d\n", roots.AIDagLeafCount)
	fmt.Printf("TensorLeafCount    : %d\n", roots.TensorLeafCount)
	fmt.Printf("PoWCommitLeafCount : %d\n", roots.PoWCommitLeafCount)
	fmt.Printf("sidecar            : %s\n", sidecarPath)

	return nil
}

func validateManifestBasics(manifest Manifest) error {
	if manifest.Version != 0 && manifest.Version != manifestVersion {
		return fmt.Errorf("unsupported manifest version: meta=%d want=%d", manifest.Version, manifestVersion)
	}
	if manifest.HashAlgorithm != hashAlgorithm {
		return fmt.Errorf("unsupported hash algorithm: meta=%s want=%s", manifest.HashAlgorithm, hashAlgorithm)
	}
	if manifest.MerkleVersion != merkleVersion {
		return fmt.Errorf("unsupported merkle version: meta=%d want=%d", manifest.MerkleVersion, merkleVersion)
	}
	if manifest.HeaderSize != unifiedHeaderSize {
		return fmt.Errorf("unsupported header size: meta=%d want=%d", manifest.HeaderSize, unifiedHeaderSize)
	}
	if manifest.PageSize <= manifest.HeaderSize {
		return fmt.Errorf("invalid page/header size: page=%d header=%d", manifest.PageSize, manifest.HeaderSize)
	}
	if manifest.PayloadSize != manifest.PageSize-manifest.HeaderSize {
		return fmt.Errorf("invalid payload size: got=%d want=%d", manifest.PayloadSize, manifest.PageSize-manifest.HeaderSize)
	}
	if manifest.TotalPages == 0 {
		return errors.New("manifest totalPages is zero")
	}
	if manifest.TotalBytes != manifest.TotalPages*manifest.PageSize {
		return fmt.Errorf("manifest totalBytes mismatch: got=%d want=%d", manifest.TotalBytes, manifest.TotalPages*manifest.PageSize)
	}
	if manifest.ModelPageCount == 0 || manifest.ModelEndPage <= manifest.ModelStartPage {
		return errors.New("manifest has no valid model page range")
	}
	if manifest.ModelStartPage == 0 {
		return errors.New("model must not start at index page 0")
	}
	if manifest.ModelEndPage > manifest.TotalPages {
		return fmt.Errorf("model page range out of DAG: modelEndPage=%d totalPages=%d", manifest.ModelEndPage, manifest.TotalPages)
	}
	if manifest.ModelPageCount != manifest.ModelEndPage-manifest.ModelStartPage {
		return fmt.Errorf("modelPageCount mismatch: got=%d want=%d", manifest.ModelPageCount, manifest.ModelEndPage-manifest.ModelStartPage)
	}
	if manifest.AIDagLeafCount != manifest.TotalPages {
		return fmt.Errorf("aidagLeafCount mismatch: got=%d want=%d", manifest.AIDagLeafCount, manifest.TotalPages)
	}
	if manifest.PoWCommitLeafCount != manifest.TotalPages {
		return fmt.Errorf("powCommitLeafCount mismatch: got=%d want=%d", manifest.PoWCommitLeafCount, manifest.TotalPages)
	}
	if manifest.TensorLeafCount != manifest.ModelPageCount {
		return fmt.Errorf("tensorLeafCount mismatch: got=%d want=%d", manifest.TensorLeafCount, manifest.ModelPageCount)
	}
	for _, c := range []struct {
		name string
		val  string
	}{
		{"AIDagRoot", manifest.AIDagRoot},
		{"TensorRoot", manifest.TensorRoot},
		{"PoWCommitRoot", manifest.PoWCommitRoot},
		{"ManifestRoot", manifest.ManifestRoot},
		{"ModelHash", manifest.ModelHash},
	} {
		h, err := decodeHex32(c.val)
		if err != nil {
			return fmt.Errorf("bad/empty %s: %w", c.name, err)
		}
		if isZeroHash(h) {
			return fmt.Errorf("%s is zero", c.name)
		}
	}
	return nil
}

func runExtract(cfg Config) error {
	dag := cfg.DagPath
	if dag == "" {
		return errors.New("--dag is required for extract mode")
	}

	meta := cfg.MetaPath
	if meta == "" {
		meta = dag + ".meta"
	}

	manifest, err := readManifest(meta)
	if err != nil {
		return err
	}
	if err := validateManifestBasics(manifest); err != nil {
		return err
	}

	return extractModel(dag, cfg.ExtractTo, manifest)
}

func extractModel(dagPath, outPath string, manifest Manifest) error {
	if manifest.ModelSize == 0 || manifest.ModelPageCount == 0 {
		return errors.New("manifest has no model")
	}
	if manifest.HeaderSize != unifiedHeaderSize {
		return fmt.Errorf("unsupported header size: meta=%d want=%d", manifest.HeaderSize, unifiedHeaderSize)
	}

	in, err := os.Open(dagPath)
	if err != nil {
		return err
	}
	defer in.Close()

	if err := ensureParentDir(outPath); err != nil {
		return err
	}

	tmpOut := tempPathFor(outPath)
	defer os.Remove(tmpOut)

	out, err := os.OpenFile(tmpOut, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	page := make([]byte, manifest.PageSize)
	var written uint64

	for p := manifest.ModelStartPage; p < manifest.ModelEndPage; p++ {
		n, err := in.ReadAt(page, int64(p*manifest.PageSize))
		if err != nil {
			_ = out.Close()
			return err
		}
		if uint64(n) != manifest.PageSize {
			_ = out.Close()
			return io.ErrUnexpectedEOF
		}

		hdr, _, _, err := verifyPageBytes(page, p, manifest.PageSize)
		if err != nil {
			_ = out.Close()
			return err
		}
		if hdr.PageType != PageTypeModel {
			_ = out.Close()
			return fmt.Errorf("expected model page at %d, got type %d", p, hdr.PageType)
		}

		payload := page[manifest.HeaderSize:]

		toWrite := manifest.ModelSize - written
		if toWrite > uint64(len(payload)) {
			toWrite = uint64(len(payload))
		}
		if toWrite == 0 {
			break
		}

		nw, err := out.Write(payload[:toWrite])
		if err != nil {
			_ = out.Close()
			return err
		}
		if uint64(nw) != toWrite {
			_ = out.Close()
			return io.ErrShortWrite
		}

		written += toWrite
	}

	if written != manifest.ModelSize {
		_ = out.Close()
		return fmt.Errorf("extract size mismatch: wrote=%d want=%d", written, manifest.ModelSize)
	}

	if err := out.Sync(); err != nil {
		_ = out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}

	hash, size, err := hashFile(tmpOut)
	if err != nil {
		return err
	}

	got := "0x" + fmtHash(hash)
	if size != manifest.ModelSize {
		return fmt.Errorf("extracted size mismatch: got=%d want=%d", size, manifest.ModelSize)
	}
	if !equalHexString(manifest.ModelHash, got) {
		return fmt.Errorf("extracted model hash mismatch: got=%s want=%s", got, manifest.ModelHash)
	}

	if err := commitAtomicPath(tmpOut, outPath, true); err != nil {
		return err
	}

	fmt.Println("extract OK")
	fmt.Printf("model out  : %s\n", outPath)
	fmt.Printf("model size : %d\n", size)
	fmt.Printf("model hash : %s\n", got)

	return nil
}

func runInfo(cfg Config) error {
	dag := cfg.DagPath
	if dag == "" {
		dag = cfg.OutPath
	}

	meta := cfg.MetaPath
	if meta == "" {
		meta = dag + ".meta"
	}

	manifest, err := readManifest(meta)
	if err != nil {
		return err
	}

	b, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}

	fmt.Println(string(b))
	return nil
}

func runProve(cfg Config) error {
	if cfg.DagPath == "" {
		return errors.New("--dag is required for prove mode")
	}
	if cfg.ProofPath == "" {
		return errors.New("--proof is required for prove mode")
	}
	if cfg.BlockHash == "" {
		return errors.New("--block-hash is required for prove mode")
	}
	if cfg.Miner == "" {
		return errors.New("--miner is required for prove mode")
	}
	if cfg.Samples <= 0 {
		return errors.New("--samples must be > 0")
	}
	if cfg.TensorSamples < 0 {
		return errors.New("--tensor-samples must be >= 0")
	}

	meta := cfg.MetaPath
	if meta == "" {
		meta = cfg.DagPath + ".meta"
	}
	sidecarPath := cfg.SidecarPath
	if sidecarPath == "" {
		sidecarPath = cfg.DagPath + defaultSidecarExt
	}

	manifest, err := readManifest(meta)
	if err != nil {
		return err
	}
	if err := validateManifestBasics(manifest); err != nil {
		return err
	}
	gotManifestRoot := "0x" + fmtHash(hashManifestRoot(manifest))
	if !equalHexString(manifest.ManifestRoot, gotManifestRoot) {
		return fmt.Errorf("ManifestRoot mismatch: meta=%s got=%s", manifest.ManifestRoot, gotManifestRoot)
	}

	fmt.Println("opening mandatory Merkle sidecar tree for proof generation...")
	sidecar, err := openMerkleSidecar(sidecarPath)
	if err != nil {
		return err
	}
	defer sidecar.Close()
	if err := sidecar.validateAgainstManifest(manifest); err != nil {
		return err
	}

	trees := ProofTrees{Sidecar: sidecar}

	powSampleIndices, tensorSamplePages, requiredPages, err := deriveRequiredProofPages(manifest, cfg.BlockHash, cfg.Miner, cfg.Epoch, cfg.Nonce, cfg.Samples, cfg.TensorSamples)
	if err != nil {
		return err
	}

	proofPages := make([]AIDagPageProof, 0, len(requiredPages))
	materials := make([]pageProofMaterial, 0, len(requiredPages))
	for _, pageIndex := range requiredPages {
		mat, err := readProofMaterial(cfg.DagPath, manifest, pageIndex)
		if err != nil {
			return err
		}

		powPath, err := trees.Sidecar.Proof(treePoW, pageIndex)
		if err != nil {
			return err
		}

		pageProof := AIDagPageProof{
			PageIndex: pageIndex,
			Page:      "0x" + hex.EncodeToString(mat.Page),
			PoWPath:   powPath,
		}

		if isTensorRequiredPage(pageIndex, tensorSamplePages) {
			if mat.Header.PageType != PageTypeModel {
				return fmt.Errorf("required tensor sample page %d is not a model page", pageIndex)
			}
			tensorIndex := pageIndex - manifest.ModelStartPage
			tensorPath, err := trees.Sidecar.Proof(treeTensor, tensorIndex)
			if err != nil {
				return err
			}
			pageProof.TensorPath = tensorPath
		}

		proofPages = append(proofPages, pageProof)
		materials = append(materials, mat)
	}

	seal := AISeal{
		Version:           sealVersion,
		Epoch:             cfg.Epoch,
		BlockHash:         normalizeArbitraryHexOrString(cfg.BlockHash),
		Miner:             strings.ToLower(strings.TrimSpace(cfg.Miner)),
		Nonce:             cfg.Nonce,
		PowSampleCount:    cfg.Samples,
		TensorSampleCount: cfg.TensorSamples,
		ManifestRoot:      normalizeHexString(manifest.ManifestRoot),
		AIDagRoot:         normalizeHexString(manifest.AIDagRoot),
		TensorRoot:        normalizeHexString(manifest.TensorRoot),
		PoWRoot:           normalizeHexString(manifest.PoWCommitRoot),
		ModelHash:         normalizeHexString(manifest.ModelHash),
	}

	mixDigest := computeMixDigest(seal, manifest, powSampleIndices, tensorSamplePages, materials)
	aiDigest := computeAIDigest(seal, manifest, tensorSamplePages, materials)
	seal.MixDigest = "0x" + fmtHash(mixDigest)
	seal.AIDigest = "0x" + fmtHash(aiDigest)
	workHash := computeSealWorkHash(seal)
	seal.WorkHash = "0x" + fmtHash(workHash)

	proof := AISealProof{
		Version: proofVersion,
		Seal:    seal,
		Pages:   proofPages,
	}
	proofHash := computeProofHash(proof)
	proof.Seal.ProofHash = "0x" + fmtHash(proofHash)

	if cfg.Target != "" {
		target, err := decodeHex32(cfg.Target)
		if err != nil {
			return fmt.Errorf("bad --target: %w", err)
		}
		if !hashMeetsTarget(workHash, target) {
			return fmt.Errorf("work hash does not meet target: work=%s target=%s", proof.Seal.WorkHash, normalizeHexString(cfg.Target))
		}
	}

	b, err := json.MarshalIndent(proof, "", "  ")
	if err != nil {
		return err
	}
	if err := writeFileAtomic(cfg.ProofPath, b, 0644, cfg.Force); err != nil {
		return err
	}

	fmt.Println("AISeal proof generated successfully")
	fmt.Printf("proof              : %s\n", cfg.ProofPath)
	fmt.Printf("sidecar            : %s\n", sidecarPath)
	fmt.Printf("pow samples        : %d\n", len(powSampleIndices))
	fmt.Printf("tensor samples     : %d\n", len(tensorSamplePages))
	fmt.Printf("unique proof pages : %d\n", len(proofPages))
	fmt.Printf("mixDigest          : %s\n", proof.Seal.MixDigest)
	fmt.Printf("aiDigest           : %s\n", proof.Seal.AIDigest)
	fmt.Printf("workHash           : %s\n", proof.Seal.WorkHash)
	fmt.Printf("proofHash          : %s\n", proof.Seal.ProofHash)

	return nil
}

func runVerifyProof(cfg Config) error {
	if cfg.ProofPath == "" {
		return errors.New("--proof is required for verify-proof mode")
	}

	meta := cfg.MetaPath
	if meta == "" {
		if cfg.DagPath != "" {
			meta = cfg.DagPath + ".meta"
		} else {
			return errors.New("--meta is required for verify-proof mode when --dag is not provided")
		}
	}

	manifestBytes, err := os.ReadFile(meta)
	if err != nil {
		return err
	}
	manifest, err := aiseal.LoadManifestBytes(manifestBytes)
	if err != nil {
		return err
	}
	proofBytes, err := os.ReadFile(cfg.ProofPath)
	if err != nil {
		return err
	}
	result, proof, err := aiseal.VerifyBytes(manifest, proofBytes, cfg.Target, aiseal.DefaultLimits())
	if err != nil {
		return err
	}

	fmt.Println("AISeal light verify OK")
	fmt.Printf("proof              : %s\n", cfg.ProofPath)
	fmt.Printf("mixDigest          : %s\n", proof.Seal.MixDigest)
	fmt.Printf("aiDigest           : %s\n", proof.Seal.AIDigest)
	fmt.Printf("workHash           : %s\n", result.WorkHash)
	fmt.Printf("proofHash          : %s\n", result.ProofHash)
	fmt.Printf("verified pages     : %d\n", len(proof.Pages))

	return nil
}

func buildMerkleSidecarTempFromDAG(dagPath, sidecarTmpPath string, manifest Manifest) (SidecarManifest, error) {
	if err := validateManifestBasics(manifest); err != nil {
		return SidecarManifest{}, err
	}
	if err := ensureParentDir(sidecarTmpPath); err != nil {
		return SidecarManifest{}, err
	}

	f, err := os.OpenFile(sidecarTmpPath, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0644)
	if err != nil {
		return SidecarManifest{}, err
	}
	defer f.Close()

	if _, err := f.Write(make([]byte, sidecarHeaderReserved)); err != nil {
		return SidecarManifest{}, fmt.Errorf("reserve sidecar header: %w", err)
	}

	dag, err := os.Open(dagPath)
	if err != nil {
		return SidecarManifest{}, err
	}
	defer dag.Close()

	powLevel0 := SidecarLevel{Level: 0, Count: manifest.TotalPages, Offset: sidecarHeaderReserved, Bytes: manifest.TotalPages * hashSize}
	powWriter := bufio.NewWriterSize(f, 4*1024*1024)
	page := make([]byte, manifest.PageSize)

	for pageIndex := uint64(0); pageIndex < manifest.TotalPages; pageIndex++ {
		n, err := dag.ReadAt(page, int64(pageIndex*manifest.PageSize))
		if err != nil {
			return SidecarManifest{}, fmt.Errorf("sidecar read page %d: %w", pageIndex, err)
		}
		if uint64(n) != manifest.PageSize {
			return SidecarManifest{}, io.ErrUnexpectedEOF
		}
		hdr, payloadHash, pageCommit, err := verifyPageBytes(page, pageIndex, manifest.PageSize)
		if err != nil {
			return SidecarManifest{}, err
		}
		leaf := merkleLeafHash(treePoW, hdr, payloadHash, pageCommit)
		if _, err := powWriter.Write(leaf[:]); err != nil {
			return SidecarManifest{}, err
		}
		if (pageIndex+1)%1000000 == 0 {
			fmt.Printf("  sidecar pow leaves: page %d\n", pageIndex+1)
		}
	}
	if err := powWriter.Flush(); err != nil {
		return SidecarManifest{}, err
	}

	tensorOffset, err := currentOffset(f)
	if err != nil {
		return SidecarManifest{}, err
	}
	tensorLevel0 := SidecarLevel{Level: 0, Count: manifest.ModelPageCount, Offset: tensorOffset, Bytes: manifest.ModelPageCount * hashSize}
	tensorWriter := bufio.NewWriterSize(f, 4*1024*1024)

	for pageIndex := manifest.ModelStartPage; pageIndex < manifest.ModelEndPage; pageIndex++ {
		n, err := dag.ReadAt(page, int64(pageIndex*manifest.PageSize))
		if err != nil {
			return SidecarManifest{}, fmt.Errorf("sidecar read tensor page %d: %w", pageIndex, err)
		}
		if uint64(n) != manifest.PageSize {
			return SidecarManifest{}, io.ErrUnexpectedEOF
		}
		hdr, payloadHash, pageCommit, err := verifyPageBytes(page, pageIndex, manifest.PageSize)
		if err != nil {
			return SidecarManifest{}, err
		}
		if hdr.PageType != PageTypeModel {
			return SidecarManifest{}, fmt.Errorf("sidecar expected model page at %d, got type %d", pageIndex, hdr.PageType)
		}
		if hdr.ShardID != pageIndex-manifest.ModelStartPage {
			return SidecarManifest{}, fmt.Errorf("sidecar model shard mismatch at page %d: shard=%d expected=%d", pageIndex, hdr.ShardID, pageIndex-manifest.ModelStartPage)
		}
		leaf := merkleLeafHash(treeTensor, hdr, payloadHash, pageCommit)
		if _, err := tensorWriter.Write(leaf[:]); err != nil {
			return SidecarManifest{}, err
		}
	}
	if err := tensorWriter.Flush(); err != nil {
		return SidecarManifest{}, err
	}

	powLevels, err := appendSidecarUpperLevels(f, treePoW, []SidecarLevel{powLevel0})
	if err != nil {
		return SidecarManifest{}, err
	}
	tensorLevels, err := appendSidecarUpperLevels(f, treeTensor, []SidecarLevel{tensorLevel0})
	if err != nil {
		return SidecarManifest{}, err
	}

	powRoot, err := readSidecarRoot(f, powLevels)
	if err != nil {
		return SidecarManifest{}, err
	}
	tensorRoot, err := readSidecarRoot(f, tensorLevels)
	if err != nil {
		return SidecarManifest{}, err
	}
	manifestPowRoot, err := decodeHex32(manifest.PoWCommitRoot)
	if err != nil {
		return SidecarManifest{}, err
	}
	manifestTensorRoot, err := decodeHex32(manifest.TensorRoot)
	if err != nil {
		return SidecarManifest{}, err
	}
	if powRoot != manifestPowRoot {
		return SidecarManifest{}, fmt.Errorf("sidecar PoW root mismatch: got=0x%s want=%s", fmtHash(powRoot), manifest.PoWCommitRoot)
	}
	if tensorRoot != manifestTensorRoot {
		return SidecarManifest{}, fmt.Errorf("sidecar Tensor root mismatch: got=0x%s want=%s", fmtHash(tensorRoot), manifest.TensorRoot)
	}

	sm := SidecarManifest{
		Version:        sidecarVersion,
		HashAlgorithm:  hashAlgorithm,
		MerkleVersion:  merkleVersion,
		CreatedAt:      time.Now().UTC().Format(time.RFC3339),
		PageSize:       manifest.PageSize,
		HeaderSize:     manifest.HeaderSize,
		PayloadSize:    manifest.PayloadSize,
		TotalPages:     manifest.TotalPages,
		ModelStartPage: manifest.ModelStartPage,
		ModelPageCount: manifest.ModelPageCount,
		ManifestRoot:   normalizeHexString(manifest.ManifestRoot),
		AIDagRoot:      normalizeHexString(manifest.AIDagRoot),
		TensorRoot:     normalizeHexString(manifest.TensorRoot),
		PoWCommitRoot:  normalizeHexString(manifest.PoWCommitRoot),
		PoWLevels:      powLevels,
		TensorLevels:   tensorLevels,
	}

	if err := writeSidecarHeader(f, sm); err != nil {
		return SidecarManifest{}, err
	}
	if err := f.Sync(); err != nil {
		return SidecarManifest{}, err
	}

	fmt.Printf("  sidecar pow levels    : %d\n", len(powLevels))
	fmt.Printf("  sidecar tensor levels : %d\n", len(tensorLevels))
	fmt.Printf("  sidecar pow root      : 0x%s\n", fmtHash(powRoot))
	fmt.Printf("  sidecar tensor root   : 0x%s\n", fmtHash(tensorRoot))

	return sm, nil
}

func appendSidecarUpperLevels(f *os.File, treeID string, levels []SidecarLevel) ([]SidecarLevel, error) {
	for levels[len(levels)-1].Count > 1 {
		prev := levels[len(levels)-1]
		if prev.Bytes != prev.Count*hashSize {
			return nil, fmt.Errorf("bad sidecar level bytes: level=%d bytes=%d count=%d", prev.Level, prev.Bytes, prev.Count)
		}

		nextCount := (prev.Count + 1) / 2
		nextOffset, err := currentOffset(f)
		if err != nil {
			return nil, err
		}
		next := SidecarLevel{Level: prev.Level + 1, Count: nextCount, Offset: nextOffset, Bytes: nextCount * hashSize}

		section := io.NewSectionReader(f, int64(prev.Offset), int64(prev.Bytes))
		writer := bufio.NewWriterSize(f, 4*1024*1024)

		for i := uint64(0); i < prev.Count; i += 2 {
			left, err := readHashFromReader(section)
			if err != nil {
				return nil, err
			}
			parent := left
			if i+1 < prev.Count {
				right, err := readHashFromReader(section)
				if err != nil {
					return nil, err
				}
				parent = merkleNodeHash(treeID, left, right)
			}
			if _, err := writer.Write(parent[:]); err != nil {
				return nil, err
			}
		}
		if err := writer.Flush(); err != nil {
			return nil, err
		}

		levels = append(levels, next)
		fmt.Printf("  sidecar %s level %d count %d\n", treeID, next.Level, next.Count)
	}
	return levels, nil
}

func readSidecarRoot(f *os.File, levels []SidecarLevel) ([32]byte, error) {
	var out [32]byte
	if len(levels) == 0 {
		return out, errors.New("empty sidecar levels")
	}
	last := levels[len(levels)-1]
	if last.Count != 1 || last.Bytes != hashSize {
		return out, fmt.Errorf("bad sidecar root level: level=%d count=%d bytes=%d", last.Level, last.Count, last.Bytes)
	}
	_, err := f.ReadAt(out[:], int64(last.Offset))
	return out, err
}

func writeSidecarHeader(f *os.File, sm SidecarManifest) error {
	b, err := json.Marshal(sm)
	if err != nil {
		return err
	}
	if uint64(len(b))+16 > sidecarHeaderReserved {
		return fmt.Errorf("sidecar header too large: %d > reserved %d", len(b)+16, sidecarHeaderReserved)
	}

	buf := make([]byte, sidecarHeaderReserved)
	copy(buf[0:8], []byte(sidecarMagic))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(len(b)))
	copy(buf[16:], b)
	_, err = f.WriteAt(buf, 0)
	return err
}

func openMerkleSidecar(path string) (*MerkleSidecarReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open sidecar: %w", err)
	}

	prefix := make([]byte, 16)
	if _, err := f.ReadAt(prefix, 0); err != nil {
		_ = f.Close()
		return nil, err
	}
	if string(prefix[0:8]) != sidecarMagic {
		_ = f.Close()
		return nil, fmt.Errorf("bad sidecar magic: %q", string(prefix[0:8]))
	}
	headerLen := binary.LittleEndian.Uint64(prefix[8:16])
	if headerLen == 0 || headerLen+16 > sidecarHeaderReserved {
		_ = f.Close()
		return nil, fmt.Errorf("bad sidecar header length: %d", headerLen)
	}
	header := make([]byte, headerLen)
	if _, err := f.ReadAt(header, 16); err != nil {
		_ = f.Close()
		return nil, err
	}

	var sm SidecarManifest
	if err := json.Unmarshal(header, &sm); err != nil {
		_ = f.Close()
		return nil, err
	}

	r := &MerkleSidecarReader{path: path, file: f, meta: sm}
	if err := r.validateSelf(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return r, nil
}

func (r *MerkleSidecarReader) Close() error {
	if r == nil || r.file == nil {
		return nil
	}
	return r.file.Close()
}

func (r *MerkleSidecarReader) validateSelf() error {
	if r.meta.Version != sidecarVersion {
		return fmt.Errorf("unsupported sidecar version: got=%d want=%d", r.meta.Version, sidecarVersion)
	}
	if r.meta.HashAlgorithm != hashAlgorithm {
		return fmt.Errorf("unsupported sidecar hash algorithm: got=%s want=%s", r.meta.HashAlgorithm, hashAlgorithm)
	}
	if r.meta.MerkleVersion != merkleVersion {
		return fmt.Errorf("unsupported sidecar merkle version: got=%d want=%d", r.meta.MerkleVersion, merkleVersion)
	}
	if r.meta.PageSize <= unifiedHeaderSize {
		return fmt.Errorf("bad sidecar page size: %d", r.meta.PageSize)
	}
	for name, levels := range map[string][]SidecarLevel{
		treePoW:    r.meta.PoWLevels,
		treeTensor: r.meta.TensorLevels,
	} {
		if err := validateSidecarLevels(name, levels); err != nil {
			return err
		}
	}
	return nil
}

func validateSidecarLevels(treeID string, levels []SidecarLevel) error {
	if len(levels) == 0 {
		return fmt.Errorf("sidecar %s has no levels", treeID)
	}
	for i, level := range levels {
		if level.Level != i {
			return fmt.Errorf("sidecar %s level index mismatch: got=%d want=%d", treeID, level.Level, i)
		}
		if level.Count == 0 {
			return fmt.Errorf("sidecar %s level %d count is zero", treeID, i)
		}
		if level.Bytes != level.Count*hashSize {
			return fmt.Errorf("sidecar %s level %d bytes mismatch: got=%d want=%d", treeID, i, level.Bytes, level.Count*hashSize)
		}
		if i > 0 {
			want := (levels[i-1].Count + 1) / 2
			if level.Count != want {
				return fmt.Errorf("sidecar %s level %d count mismatch: got=%d want=%d", treeID, i, level.Count, want)
			}
		}
	}
	last := levels[len(levels)-1]
	if last.Count != 1 {
		return fmt.Errorf("sidecar %s root level count is %d", treeID, last.Count)
	}
	return nil
}

func (r *MerkleSidecarReader) validateAgainstManifest(manifest Manifest) error {
	if r.meta.PageSize != manifest.PageSize {
		return fmt.Errorf("sidecar pageSize mismatch: sidecar=%d manifest=%d", r.meta.PageSize, manifest.PageSize)
	}
	if r.meta.HeaderSize != manifest.HeaderSize {
		return fmt.Errorf("sidecar headerSize mismatch: sidecar=%d manifest=%d", r.meta.HeaderSize, manifest.HeaderSize)
	}
	if r.meta.PayloadSize != manifest.PayloadSize {
		return fmt.Errorf("sidecar payloadSize mismatch: sidecar=%d manifest=%d", r.meta.PayloadSize, manifest.PayloadSize)
	}
	if r.meta.TotalPages != manifest.TotalPages {
		return fmt.Errorf("sidecar totalPages mismatch: sidecar=%d manifest=%d", r.meta.TotalPages, manifest.TotalPages)
	}
	if r.meta.ModelStartPage != manifest.ModelStartPage || r.meta.ModelPageCount != manifest.ModelPageCount {
		return fmt.Errorf("sidecar model range mismatch: sidecar=%d/%d manifest=%d/%d", r.meta.ModelStartPage, r.meta.ModelPageCount, manifest.ModelStartPage, manifest.ModelPageCount)
	}
	checks := []struct {
		name string
		got  string
		want string
	}{
		{"ManifestRoot", r.meta.ManifestRoot, manifest.ManifestRoot},
		{"AIDagRoot", r.meta.AIDagRoot, manifest.AIDagRoot},
		{"TensorRoot", r.meta.TensorRoot, manifest.TensorRoot},
		{"PoWCommitRoot", r.meta.PoWCommitRoot, manifest.PoWCommitRoot},
	}
	for _, c := range checks {
		if !equalHexString(c.got, c.want) {
			return fmt.Errorf("sidecar %s mismatch: sidecar=%s manifest=%s", c.name, c.got, c.want)
		}
	}
	if len(r.meta.PoWLevels) == 0 || r.meta.PoWLevels[0].Count != manifest.PoWCommitLeafCount {
		return fmt.Errorf("sidecar PoW leaf count mismatch: sidecar=%d manifest=%d", firstLevelCount(r.meta.PoWLevels), manifest.PoWCommitLeafCount)
	}
	if len(r.meta.TensorLevels) == 0 || r.meta.TensorLevels[0].Count != manifest.TensorLeafCount {
		return fmt.Errorf("sidecar Tensor leaf count mismatch: sidecar=%d manifest=%d", firstLevelCount(r.meta.TensorLevels), manifest.TensorLeafCount)
	}
	return nil
}

func (r *MerkleSidecarReader) Proof(treeID string, leafIndex uint64) ([]MerkleProofStep, error) {
	levels, err := r.levels(treeID)
	if err != nil {
		return nil, err
	}
	if len(levels) == 0 {
		return nil, errors.New("cannot prove empty sidecar tree")
	}
	if leafIndex >= levels[0].Count {
		return nil, fmt.Errorf("leaf index out of range: %d >= %d", leafIndex, levels[0].Count)
	}

	idx := leafIndex
	proof := make([]MerkleProofStep, 0, len(levels)-1)
	for levelIdx := 0; levelIdx < len(levels)-1; levelIdx++ {
		level := levels[levelIdx]
		if idx%2 == 0 {
			if idx+1 < level.Count {
				sibling, err := r.readLevelHash(level, idx+1)
				if err != nil {
					return nil, err
				}
				proof = append(proof, MerkleProofStep{Side: ProofSideRight, Hash: "0x" + fmtHash(sibling)})
			}
		} else {
			sibling, err := r.readLevelHash(level, idx-1)
			if err != nil {
				return nil, err
			}
			proof = append(proof, MerkleProofStep{Side: ProofSideLeft, Hash: "0x" + fmtHash(sibling)})
		}
		idx /= 2
	}
	return proof, nil
}

func (r *MerkleSidecarReader) levels(treeID string) ([]SidecarLevel, error) {
	switch treeID {
	case treePoW:
		return r.meta.PoWLevels, nil
	case treeTensor:
		return r.meta.TensorLevels, nil
	default:
		return nil, fmt.Errorf("unknown sidecar tree: %s", treeID)
	}
}

func (r *MerkleSidecarReader) readLevelHash(level SidecarLevel, index uint64) ([32]byte, error) {
	var out [32]byte
	if index >= level.Count {
		return out, fmt.Errorf("sidecar level %d index out of range: %d >= %d", level.Level, index, level.Count)
	}
	_, err := r.file.ReadAt(out[:], int64(level.Offset+index*hashSize))
	return out, err
}

func buildProofTreesFromDAG(dagPath string, manifest Manifest) (ProofTrees, error) {
	sidecarPath := dagPath + defaultSidecarExt
	sidecar, err := openMerkleSidecar(sidecarPath)
	if err != nil {
		return ProofTrees{}, err
	}
	if err := sidecar.validateAgainstManifest(manifest); err != nil {
		_ = sidecar.Close()
		return ProofTrees{}, err
	}
	return ProofTrees{Sidecar: sidecar}, nil
}

func compareProofTreeRootsWithManifest(trees ProofTrees, manifest Manifest) error {
	if trees.Sidecar == nil {
		return errors.New("missing sidecar proof tree")
	}
	return trees.Sidecar.validateAgainstManifest(manifest)
}

func deriveRequiredProofPages(manifest Manifest, blockHash string, miner string, epoch uint64, nonce uint64, powSamples int, tensorSamples int) ([]uint64, []uint64, []uint64, error) {
	if powSamples <= 0 {
		return nil, nil, nil, errors.New("pow sample count must be > 0")
	}
	if tensorSamples < 0 {
		return nil, nil, nil, errors.New("tensor sample count must be >= 0")
	}
	if uint64(powSamples) > manifest.TotalPages {
		return nil, nil, nil, fmt.Errorf("pow sample count too large: samples=%d totalPages=%d", powSamples, manifest.TotalPages)
	}
	if uint64(tensorSamples) > manifest.ModelPageCount {
		return nil, nil, nil, fmt.Errorf("tensor sample count too large: samples=%d modelPages=%d", tensorSamples, manifest.ModelPageCount)
	}

	powIndices := sampleUniqueIndices(domainSamplePow, manifest, blockHash, miner, epoch, nonce, uint64(powSamples), 0, manifest.TotalPages)
	tensorIndices := sampleUniqueIndices(domainSampleTensor, manifest, blockHash, miner, epoch, nonce, uint64(tensorSamples), manifest.ModelStartPage, manifest.ModelPageCount)

	set := make(map[uint64]struct{}, len(powIndices)+len(tensorIndices))
	for _, p := range powIndices {
		set[p] = struct{}{}
	}
	for _, p := range tensorIndices {
		set[p] = struct{}{}
	}

	required := make([]uint64, 0, len(set))
	for p := range set {
		required = append(required, p)
	}
	sort.Slice(required, func(i, j int) bool { return required[i] < required[j] })

	return powIndices, tensorIndices, required, nil
}

func sampleUniqueIndices(domain string, manifest Manifest, blockHash string, miner string, epoch uint64, nonce uint64, count uint64, start uint64, span uint64) []uint64 {
	if count == 0 || span == 0 {
		return nil
	}

	seed := blake3HashTagged(domain, func(w io.Writer) {
		writeString(w, normalizeArbitraryHexOrString(blockHash))
		writeString(w, strings.ToLower(strings.TrimSpace(miner)))
		writeUint64(w, epoch)
		writeUint64(w, nonce)
		writeString(w, normalizeHexString(manifest.ManifestRoot))
		writeString(w, normalizeHexString(manifest.PoWCommitRoot))
		writeString(w, normalizeHexString(manifest.TensorRoot))
		writeUint64(w, manifest.TotalPages)
		writeUint64(w, manifest.ModelStartPage)
		writeUint64(w, manifest.ModelPageCount)
		writeUint64(w, count)
		writeUint64(w, start)
		writeUint64(w, span)
	})

	out := make([]uint64, 0, count)
	seen := make(map[uint64]struct{}, count)
	var counter uint64

	for uint64(len(out)) < count {
		h := blake3HashTagged(domain, func(w io.Writer) {
			writeFixed32(w, seed)
			writeUint64(w, counter)
		})
		idx := start + binary.LittleEndian.Uint64(h[:8])%span
		counter++

		if _, ok := seen[idx]; ok {
			continue
		}
		seen[idx] = struct{}{}
		out = append(out, idx)
	}

	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func isTensorRequiredPage(pageIndex uint64, tensorSamplePages []uint64) bool {
	i := sort.Search(len(tensorSamplePages), func(i int) bool { return tensorSamplePages[i] >= pageIndex })
	return i < len(tensorSamplePages) && tensorSamplePages[i] == pageIndex
}

func readProofMaterial(dagPath string, manifest Manifest, pageIndex uint64) (pageProofMaterial, error) {
	if pageIndex >= manifest.TotalPages {
		return pageProofMaterial{}, fmt.Errorf("page index out of range: %d >= %d", pageIndex, manifest.TotalPages)
	}

	f, err := os.Open(dagPath)
	if err != nil {
		return pageProofMaterial{}, err
	}
	defer f.Close()

	page := make([]byte, manifest.PageSize)
	n, err := f.ReadAt(page, int64(pageIndex*manifest.PageSize))
	if err != nil {
		return pageProofMaterial{}, err
	}
	if uint64(n) != manifest.PageSize {
		return pageProofMaterial{}, io.ErrUnexpectedEOF
	}

	hdr, payloadHash, pageCommit, err := verifyPageBytes(page, pageIndex, manifest.PageSize)
	if err != nil {
		return pageProofMaterial{}, err
	}

	payload := make([]byte, manifest.PayloadSize)
	copy(payload, page[manifest.HeaderSize:])

	mat := pageProofMaterial{
		PageIndex:   pageIndex,
		Header:      hdr,
		Page:        page,
		Payload:     payload,
		PayloadHash: payloadHash,
		PageCommit:  pageCommit,
		PoWLeaf:     merkleLeafHash(treePoW, hdr, payloadHash, pageCommit),
	}
	if hdr.PageType == PageTypeModel {
		mat.TensorLeaf = merkleLeafHash(treeTensor, hdr, payloadHash, pageCommit)
	}

	return mat, nil
}

func computeMixDigest(seal AISeal, manifest Manifest, powSampleIndices []uint64, tensorSamplePages []uint64, materials []pageProofMaterial) [32]byte {
	matByIndex := make(map[uint64]pageProofMaterial, len(materials))
	for _, mat := range materials {
		matByIndex[mat.PageIndex] = mat
	}

	return blake3HashTagged(domainMixDigest, func(w io.Writer) {
		writeSealChallengeFields(w, seal)
		writeString(w, normalizeHexString(manifest.ManifestRoot))
		writeUint64(w, manifest.TotalPages)
		writeUint64(w, manifest.ModelStartPage)
		writeUint64(w, manifest.ModelPageCount)

		writeUint64(w, uint64(len(powSampleIndices)))
		for _, pageIndex := range powSampleIndices {
			mat, ok := matByIndex[pageIndex]
			if !ok {
				continue
			}
			writeUint64(w, pageIndex)
			writeUint16(w, mat.Header.PageType)
			writeFixed32(w, mat.PayloadHash)
			writeFixed32(w, mat.PageCommit)
			writeFixed32(w, mat.PoWLeaf)
		}

		writeUint64(w, uint64(len(tensorSamplePages)))
		for _, pageIndex := range tensorSamplePages {
			mat, ok := matByIndex[pageIndex]
			if !ok {
				continue
			}
			writeUint64(w, pageIndex)
			writeUint64(w, mat.Header.ShardID)
			writeUint64(w, mat.Header.ModelOffset)
			writeUint64(w, mat.Header.ModelSize)
			writeFixed32(w, mat.PayloadHash)
			writeFixed32(w, mat.PageCommit)
			writeFixed32(w, mat.TensorLeaf)
		}
	})
}

func computeAIDigest(seal AISeal, manifest Manifest, tensorSamplePages []uint64, materials []pageProofMaterial) [32]byte {
	matByIndex := make(map[uint64]pageProofMaterial, len(materials))
	for _, mat := range materials {
		matByIndex[mat.PageIndex] = mat
	}

	return blake3HashTagged(domainAIDigest, func(w io.Writer) {
		writeSealChallengeFields(w, seal)
		writeString(w, normalizeHexString(manifest.ModelHash))
		writeUint64(w, manifest.ModelSize)
		writeUint64(w, manifest.ModelPageCount)
		writeUint64(w, uint64(len(tensorSamplePages)))

		for _, pageIndex := range tensorSamplePages {
			mat, ok := matByIndex[pageIndex]
			if !ok {
				continue
			}
			writeUint64(w, pageIndex)
			writeUint64(w, mat.Header.ShardID)
			writeUint64(w, mat.Header.ModelOffset)
			writeUint64(w, mat.Header.ModelSize)
			writeFixed32(w, mat.PayloadHash)
			writeFixed32(w, mat.PageCommit)

			challenge := lightweightTensorPageChallenge(seal, pageIndex, mat.Payload)
			writeFixed32(w, challenge)
		}
	})
}

func lightweightTensorPageChallenge(seal AISeal, pageIndex uint64, payload []byte) [32]byte {
	return blake3HashTagged(domainAIDigest, func(w io.Writer) {
		writeSealChallengeFields(w, seal)
		writeUint64(w, pageIndex)
		writeUint64(w, uint64(len(payload)))

		if len(payload) == 0 {
			return
		}

		seed := blake3HashTagged(domainAIDigest, func(sw io.Writer) {
			writeSealChallengeFields(sw, seal)
			writeUint64(sw, pageIndex)
		})

		for i := uint64(0); i < 32; i++ {
			h := blake3HashTagged(domainAIDigest, func(hw io.Writer) {
				writeFixed32(hw, seed)
				writeUint64(hw, i)
			})
			off := int(binary.LittleEndian.Uint64(h[:8]) % uint64(len(payload)))
			end := off + 32
			if end <= len(payload) {
				writeBytes(w, payload[off:end])
			} else {
				wrap := end - len(payload)
				buf := make([]byte, 0, 32)
				buf = append(buf, payload[off:]...)
				buf = append(buf, payload[:wrap]...)
				writeBytes(w, buf)
			}
		}
	})
}

func writeSealChallengeFields(w io.Writer, seal AISeal) {
	writeUint16(w, seal.Version)
	writeUint64(w, seal.Epoch)
	writeString(w, normalizeArbitraryHexOrString(seal.BlockHash))
	writeString(w, strings.ToLower(strings.TrimSpace(seal.Miner)))
	writeUint64(w, seal.Nonce)
	writeUint64(w, uint64(seal.PowSampleCount))
	writeUint64(w, uint64(seal.TensorSampleCount))
	writeString(w, normalizeHexString(seal.ManifestRoot))
	writeString(w, normalizeHexString(seal.AIDagRoot))
	writeString(w, normalizeHexString(seal.TensorRoot))
	writeString(w, normalizeHexString(seal.PoWRoot))
	writeString(w, normalizeHexString(seal.ModelHash))
}

func computeSealWorkHash(seal AISeal) [32]byte {
	return blake3HashTagged(domainSealWork, func(w io.Writer) {
		writeSealChallengeFields(w, seal)
		writeString(w, normalizeHexString(seal.MixDigest))
		writeString(w, normalizeHexString(seal.AIDigest))
	})
}

func computeProofHash(proof AISealProof) [32]byte {
	proof.Seal.ProofHash = ""
	proof.Pages = canonicalProofPagesForHash(proof.Pages)

	return blake3HashTagged(domainProofHash, func(w io.Writer) {
		writeUint64(w, uint64(proof.Version))
		writeSealChallengeFields(w, proof.Seal)
		writeString(w, normalizeHexString(proof.Seal.MixDigest))
		writeString(w, normalizeHexString(proof.Seal.AIDigest))
		writeString(w, normalizeHexString(proof.Seal.WorkHash))
		writeUint64(w, uint64(len(proof.Pages)))
		for _, p := range proof.Pages {
			writeUint64(w, p.PageIndex)
			writeString(w, normalizeHexString(p.Page))
			writeProofPath(w, p.PoWPath)
			writeProofPath(w, p.TensorPath)
		}
	})
}

func writeProofPath(w io.Writer, path []MerkleProofStep) {
	writeUint64(w, uint64(len(path)))
	for _, step := range path {
		writeString(w, step.Side)
		writeString(w, normalizeHexString(step.Hash))
	}
}

func canonicalProofPagesForHash(in []AIDagPageProof) []AIDagPageProof {
	out := make([]AIDagPageProof, len(in))
	copy(out, in)
	sort.Slice(out, func(i, j int) bool { return out[i].PageIndex < out[j].PageIndex })
	return out
}

func readAISealProof(path string) (AISealProof, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return AISealProof{}, err
	}
	var proof AISealProof
	if err := json.Unmarshal(b, &proof); err != nil {
		return AISealProof{}, err
	}
	return proof, nil
}

func verifyAISealProof(manifest Manifest, proof AISealProof) ([32]byte, [32]byte, error) {
	var zero [32]byte

	if proof.Version != proofVersion {
		return zero, zero, fmt.Errorf("unsupported proof version: got=%d want=%d", proof.Version, proofVersion)
	}
	if proof.Seal.Version != sealVersion {
		return zero, zero, fmt.Errorf("unsupported seal version: got=%d want=%d", proof.Seal.Version, sealVersion)
	}
	if proof.Seal.PowSampleCount <= 0 {
		return zero, zero, errors.New("seal pow sample count must be > 0")
	}
	if proof.Seal.TensorSampleCount < 0 {
		return zero, zero, errors.New("seal tensor sample count must be >= 0")
	}

	if err := verifySealMatchesManifest(proof.Seal, manifest); err != nil {
		return zero, zero, err
	}

	powRoot, err := decodeHex32(manifest.PoWCommitRoot)
	if err != nil {
		return zero, zero, fmt.Errorf("bad manifest PoWCommitRoot: %w", err)
	}
	tensorRoot, err := decodeHex32(manifest.TensorRoot)
	if err != nil {
		return zero, zero, fmt.Errorf("bad manifest TensorRoot: %w", err)
	}

	powSamples, tensorSamples, requiredPages, err := deriveRequiredProofPages(manifest, proof.Seal.BlockHash, proof.Seal.Miner, proof.Seal.Epoch, proof.Seal.Nonce, proof.Seal.PowSampleCount, proof.Seal.TensorSampleCount)
	if err != nil {
		return zero, zero, err
	}

	if len(proof.Pages) != len(requiredPages) {
		return zero, zero, fmt.Errorf("proof page count mismatch: got=%d want=%d", len(proof.Pages), len(requiredPages))
	}

	pageProofByIndex := make(map[uint64]AIDagPageProof, len(proof.Pages))
	for _, p := range proof.Pages {
		if _, exists := pageProofByIndex[p.PageIndex]; exists {
			return zero, zero, fmt.Errorf("duplicate proof page: %d", p.PageIndex)
		}
		pageProofByIndex[p.PageIndex] = p
	}

	materials := make([]pageProofMaterial, 0, len(requiredPages))
	for _, pageIndex := range requiredPages {
		p, ok := pageProofByIndex[pageIndex]
		if !ok {
			return zero, zero, fmt.Errorf("missing proof page: %d", pageIndex)
		}

		mat, err := materialFromPageProof(p, manifest)
		if err != nil {
			return zero, zero, err
		}

		if err := verifyMerkleProof(treePoW, mat.PoWLeaf, pageIndex, manifest.PoWCommitLeafCount, p.PoWPath, powRoot); err != nil {
			return zero, zero, fmt.Errorf("PoW proof failed for page %d: %w", pageIndex, err)
		}

		if isTensorRequiredPage(pageIndex, tensorSamples) {
			if mat.Header.PageType != PageTypeModel {
				return zero, zero, fmt.Errorf("tensor sample page %d is not a model page", pageIndex)
			}
			if len(p.TensorPath) == 0 && manifest.TensorLeafCount > 1 {
				return zero, zero, fmt.Errorf("missing tensor path for page %d", pageIndex)
			}
			tensorIndex := pageIndex - manifest.ModelStartPage
			if mat.Header.ShardID != tensorIndex {
				return zero, zero, fmt.Errorf("model shard mismatch at page %d: shard=%d expected=%d", pageIndex, mat.Header.ShardID, tensorIndex)
			}
			if err := verifyMerkleProof(treeTensor, mat.TensorLeaf, tensorIndex, manifest.TensorLeafCount, p.TensorPath, tensorRoot); err != nil {
				return zero, zero, fmt.Errorf("Tensor proof failed for page %d: %w", pageIndex, err)
			}
		} else if len(p.TensorPath) != 0 {
			return zero, zero, fmt.Errorf("unexpected tensor path for non-required page %d", pageIndex)
		}

		materials = append(materials, mat)
	}

	mixDigest := computeMixDigest(proof.Seal, manifest, powSamples, tensorSamples, materials)
	if !equalHexString(proof.Seal.MixDigest, "0x"+fmtHash(mixDigest)) {
		return zero, zero, fmt.Errorf("mixDigest mismatch: proof=%s got=0x%s", proof.Seal.MixDigest, fmtHash(mixDigest))
	}

	aiDigest := computeAIDigest(proof.Seal, manifest, tensorSamples, materials)
	if !equalHexString(proof.Seal.AIDigest, "0x"+fmtHash(aiDigest)) {
		return zero, zero, fmt.Errorf("aiDigest mismatch: proof=%s got=0x%s", proof.Seal.AIDigest, fmtHash(aiDigest))
	}

	workHash := computeSealWorkHash(proof.Seal)
	if proof.Seal.WorkHash != "" && !equalHexString(proof.Seal.WorkHash, "0x"+fmtHash(workHash)) {
		return zero, zero, fmt.Errorf("workHash mismatch: proof=%s got=0x%s", proof.Seal.WorkHash, fmtHash(workHash))
	}

	proofHash := computeProofHash(proof)
	if !equalHexString(proof.Seal.ProofHash, "0x"+fmtHash(proofHash)) {
		return zero, zero, fmt.Errorf("proofHash mismatch: proof=%s got=0x%s", proof.Seal.ProofHash, fmtHash(proofHash))
	}

	return workHash, proofHash, nil
}

func verifySealMatchesManifest(seal AISeal, manifest Manifest) error {
	checks := []struct {
		name string
		got  string
		want string
	}{
		{"ManifestRoot", seal.ManifestRoot, manifest.ManifestRoot},
		{"AIDagRoot", seal.AIDagRoot, manifest.AIDagRoot},
		{"TensorRoot", seal.TensorRoot, manifest.TensorRoot},
		{"PoWCommitRoot", seal.PoWRoot, manifest.PoWCommitRoot},
		{"ModelHash", seal.ModelHash, manifest.ModelHash},
	}

	for _, c := range checks {
		if !equalHexString(c.got, c.want) {
			return fmt.Errorf("seal %s mismatch: seal=%s manifest=%s", c.name, c.got, c.want)
		}
	}
	return nil
}

func materialFromPageProof(p AIDagPageProof, manifest Manifest) (pageProofMaterial, error) {
	page, err := decodeHexBytes(p.Page)
	if err != nil {
		return pageProofMaterial{}, fmt.Errorf("bad page hex for page %d: %w", p.PageIndex, err)
	}
	if uint64(len(page)) != manifest.PageSize {
		return pageProofMaterial{}, fmt.Errorf("bad page size for page %d: got=%d want=%d", p.PageIndex, len(page), manifest.PageSize)
	}

	hdr, payloadHash, pageCommit, err := verifyPageBytes(page, p.PageIndex, manifest.PageSize)
	if err != nil {
		return pageProofMaterial{}, err
	}

	payload := make([]byte, manifest.PayloadSize)
	copy(payload, page[manifest.HeaderSize:])

	mat := pageProofMaterial{
		PageIndex:   p.PageIndex,
		Header:      hdr,
		Page:        page,
		Payload:     payload,
		PayloadHash: payloadHash,
		PageCommit:  pageCommit,
		PoWLeaf:     merkleLeafHash(treePoW, hdr, payloadHash, pageCommit),
	}
	if hdr.PageType == PageTypeModel {
		mat.TensorLeaf = merkleLeafHash(treeTensor, hdr, payloadHash, pageCommit)
	}

	return mat, nil
}

func readManifest(path string) (Manifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, err
	}

	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return Manifest{}, err
	}

	return m, nil
}

func writeManifest(path string, manifest Manifest) error {
	b, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(path, b, 0644, true)
}

func hashFile(path string) ([32]byte, uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return [32]byte{}, 0, err
	}
	defer f.Close()

	h := blake3.New()
	writeString(h, domainFileHash)

	n, err := io.Copy(h, f)
	if err != nil {
		return [32]byte{}, 0, err
	}

	return finalizeHash(h), uint64(n), nil
}

func hashBytes(b []byte) [32]byte {
	return blake3HashTagged(domainHashBytes, func(w io.Writer) {
		writeBytes(w, b)
	})
}

func pageCommitHash(page []byte) [32]byte {
	return blake3HashTagged(domainPageCommit, func(w io.Writer) {
		if len(page) < pageHashEnd {
			writeBytes(w, page)
			return
		}

		writeBytes(w, page[:pageHashOffset])

		var zero [32]byte
		writeFixed32(w, zero)

		writeBytes(w, page[pageHashEnd:])
	})
}

func hashManifestRoot(m Manifest) [32]byte {
	m = canonicalManifestForRoot(m)
	b, _ := json.Marshal(m)

	return blake3HashTagged(domainManifest, func(w io.Writer) {
		writeBytes(w, b)
	})
}

func blake3HashTagged(domain string, writeFn func(io.Writer)) [32]byte {
	h := blake3.New()
	writeString(h, domain)

	if writeFn != nil {
		writeFn(h)
	}

	return finalizeHash(h)
}

func finalizeHash(h interface{ Sum([]byte) []byte }) [32]byte {
	sum := h.Sum(nil)

	var out [32]byte
	copy(out[:], sum)

	return out
}

func writeString(w io.Writer, s string) { writeBytes(w, []byte(s)) }

func writeBytes(w io.Writer, b []byte) {
	writeUint64(w, uint64(len(b)))
	_, _ = w.Write(b)
}

func writeFixed32(w io.Writer, h [32]byte) { _, _ = w.Write(h[:]) }

func writeUint16(w io.Writer, v uint16) {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	_, _ = w.Write(b[:])
}

func writeUint32(w io.Writer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, _ = w.Write(b[:])
}

func writeUint64(w io.Writer, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	_, _ = w.Write(b[:])
}

func fmtHash(h [32]byte) string { return fmt.Sprintf("%x", h[:]) }

func decodeHex32(s string) ([32]byte, error) {
	var out [32]byte
	b, err := decodeHexBytes(s)
	if err != nil {
		return out, err
	}
	if len(b) != 32 {
		return out, fmt.Errorf("expected 32 bytes, got %d", len(b))
	}
	copy(out[:], b)
	return out, nil
}

func decodeHexBytes(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		s = s[2:]
	}
	if s == "" {
		return nil, errors.New("empty hex string")
	}
	if len(s)%2 != 0 {
		return nil, errors.New("odd-length hex string")
	}
	return hex.DecodeString(s)
}

func normalizeHexString(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return ""
	}
	if strings.HasPrefix(s, "0x") {
		return s
	}
	return "0x" + s
}

func normalizeArbitraryHexOrString(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return ""
	}
	if strings.HasPrefix(s, "0x") || isHexString(s) {
		return normalizeHexString(s)
	}
	return s
}

func isHexString(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

func equalHexString(a, b string) bool { return normalizeHexString(a) == normalizeHexString(b) }

func hashMeetsTarget(hash [32]byte, target [32]byte) bool {
	return bytes.Compare(hash[:], target[:]) <= 0
}

func roundUp(v, align uint64) uint64 {
	if align == 0 {
		return v
	}
	if v%align == 0 {
		return v
	}
	return v + align - (v % align)
}

func guessModelFormat(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".gguf":
		return "gguf"
	case ".safetensors":
		return "safetensors"
	case ".bin":
		return "bin"
	default:
		return "unknown"
	}
}

func printProgressLoop(done <-chan struct{}, start time.Time, written *uint64, total uint64) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			printProgress(start, atomic.LoadUint64(written), total)
		case <-done:
			printProgress(start, atomic.LoadUint64(written), total)
			fmt.Println()
			return
		}
	}
}

func printProgress(start time.Time, written, total uint64) {
	elapsed := time.Since(start).Seconds()
	if elapsed <= 0 {
		elapsed = 1
	}

	percent := float64(written) / float64(total) * 100
	mbs := float64(written) / 1024 / 1024 / elapsed

	eta := 0.0
	if mbs > 0 && written < total {
		eta = float64(total-written) / 1024 / 1024 / mbs
	}

	fmt.Printf(
		"\rprogress: %.2f%% written: %.2f GiB / %.2f GiB speed: %.2f MiB/s eta: %.0fs",
		percent,
		float64(written)/1024/1024/1024,
		float64(total)/1024/1024/1024,
		mbs,
		eta,
	)
}

func validateRootsNonEmpty(roots Roots, manifest Manifest) error {
	checks := []struct {
		name      string
		root      [32]byte
		count     uint64
		emptyHash [32]byte
	}{
		{"AIDagRoot", roots.AIDagRoot, roots.AIDagLeafCount, merkleEmptyHash(treeAIDag)},
		{"TensorRoot", roots.TensorRoot, roots.TensorLeafCount, merkleEmptyHash(treeTensor)},
		{"PoWCommitRoot", roots.PoWCommitRoot, roots.PoWCommitLeafCount, merkleEmptyHash(treePoW)},
	}
	for _, c := range checks {
		if c.count == 0 {
			return fmt.Errorf("%s leaf count is zero", c.name)
		}
		if isZeroHash(c.root) {
			return fmt.Errorf("%s is zero", c.name)
		}
		if c.root == c.emptyHash {
			return fmt.Errorf("%s equals empty-tree hash despite count=%d", c.name, c.count)
		}
	}
	if roots.AIDagLeafCount != manifest.TotalPages {
		return fmt.Errorf("AIDagLeafCount mismatch: got=%d want=%d", roots.AIDagLeafCount, manifest.TotalPages)
	}
	if roots.PoWCommitLeafCount != manifest.TotalPages {
		return fmt.Errorf("PoWCommitLeafCount mismatch: got=%d want=%d", roots.PoWCommitLeafCount, manifest.TotalPages)
	}
	if roots.TensorLeafCount != manifest.ModelPageCount {
		return fmt.Errorf("TensorLeafCount mismatch: got=%d want=%d", roots.TensorLeafCount, manifest.ModelPageCount)
	}
	return nil
}

func isZeroHash(h [32]byte) bool {
	var z [32]byte
	return h == z
}

func readHashFromReader(r io.Reader) ([32]byte, error) {
	var h [32]byte
	_, err := io.ReadFull(r, h[:])
	return h, err
}

func currentOffset(f *os.File) (uint64, error) {
	off, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}
	return uint64(off), nil
}

func firstLevelCount(levels []SidecarLevel) uint64 {
	if len(levels) == 0 {
		return 0
	}
	return levels[0].Count
}

func firstMissingPage(seen []uint8) uint64 {
	for i, v := range seen {
		if v == 0 {
			return uint64(i)
		}
	}
	return ^uint64(0)
}

func maxInt() int { return int(^uint(0) >> 1) }

func preflightAtomicTargets(force bool, paths ...string) error {
	if force {
		return nil
	}
	for _, p := range paths {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err == nil {
			return fmt.Errorf("output exists: %s ; use --force", p)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func ensureParentDir(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0755)
}

func tempPathFor(path string) string {
	dir := filepath.Dir(path)
	base := filepath.Base(path)
	return filepath.Join(dir, fmt.Sprintf(".%s.tmp.%d.%d", base, os.Getpid(), time.Now().UnixNano()))
}

func commitAtomicPath(tmpPath, finalPath string, force bool) error {
	if !force {
		if _, err := os.Stat(finalPath); err == nil {
			return fmt.Errorf("target exists: %s ; use --force", finalPath)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	} else {
		if err := os.Remove(finalPath); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}

	if err := os.Rename(tmpPath, finalPath); err != nil {
		return err
	}
	return fsyncDir(filepath.Dir(finalPath))
}

func writeFileAtomic(path string, data []byte, perm os.FileMode, force bool) error {
	if err := ensureParentDir(path); err != nil {
		return err
	}
	if !force {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("target exists: %s ; use --force", path)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}

	tmp := tempPathFor(path)
	defer os.Remove(tmp)

	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, perm)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return commitAtomicPath(tmp, path, force)
}

func fsyncDir(dir string) error {
	if dir == "" {
		dir = "."
	}
	d, err := os.Open(dir)
	if err != nil {
		return nil // best effort; some platforms/filesystems reject directory open/sync
	}
	defer d.Close()
	_ = d.Sync()
	return nil
}
