package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

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
)

const (
	PageTypeIndex  = uint16(0x0001)
	PageTypeModel  = uint16(0x0002)
	PageTypeFiller = uint16(0x0003)
)

const (
	treeAIDag = "AIDG_MERKLE_TREE_AIDAG_V1"
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
)

type Config struct {
	Mode string

	OutPath   string
	DagPath   string
	MetaPath  string
	ModelPath string
	ExtractTo string

	SizeGB   uint64
	PageSize uint64
	Seed     string

	Workers int
	ChunkMB uint64

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
	Data         []byte
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

func main() {
	cfg := Config{}

	flag.StringVar(&cfg.Mode, "mode", "gen", "mode: gen, verify, extract, info")
	flag.StringVar(&cfg.OutPath, "out", "./unified-aidag-128g.bin", "output unified AI-DAG path for gen mode")
	flag.StringVar(&cfg.DagPath, "dag", "", "AI-DAG path for verify/extract/info mode")
	flag.StringVar(&cfg.MetaPath, "meta", "", "metadata path; default is <dag>.meta or <out>.meta")
	flag.StringVar(&cfg.ModelPath, "model", "", "model file path to embed into unified AI-DAG")
	flag.StringVar(&cfg.ExtractTo, "extract-out", "./extracted-model.bin", "output model path for extract mode")

	flag.Uint64Var(&cfg.SizeGB, "size-gb", 128, "AI-DAG size in GiB")
	flag.Uint64Var(&cfg.PageSize, "page-size", defaultPageSize, "AI-DAG page size")
	flag.StringVar(&cfg.Seed, "seed", "colossusx-unified-aidag-v1", "deterministic seed")
	flag.IntVar(&cfg.Workers, "workers", runtime.NumCPU(), "worker count")
	flag.Uint64Var(&cfg.ChunkMB, "chunk-mb", 64, "model read chunk size in MiB")

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

	totalBytes := cfg.SizeGB * oneGiB
	totalPages := totalBytes / cfg.PageSize
	payloadSize := cfg.PageSize - unifiedHeaderSize

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

		PageSize:       cfg.PageSize,
		HeaderSize:     unifiedHeaderSize,
		PayloadSize:    payloadSize,
		SizeGB:         cfg.SizeGB,
		TotalBytes:     totalBytes,
		TotalPages:     totalPages,
		Seed:           cfg.Seed,
		GenerationTime: time.Now().UTC().Format(time.RFC3339),

		AIDagLeafCount:     totalPages,
		TensorLeafCount:    modelPageCount,
		PoWCommitLeafCount: totalPages,

		ModelName:      filepath.Base(cfg.ModelPath),
		ModelFormat:    guessModelFormat(cfg.ModelPath),
		ModelSize:      modelSize,
		ModelHash:      "0x" + fmtHash(modelHash),
		ModelStartPage: modelStartPage,
		ModelPageCount: modelPageCount,
		ModelEndPage:   modelEndPage,

		IndexPage:   0,
		FillerStart: modelEndPage,
		FillerEnd:   totalPages,

		UnifiedLayout: "page0=index; page1..modelEnd=model shards; remaining pages=deterministic filler; every page is PoW-mixable; roots are BLAKE3 Merkle roots",
	}

	fmt.Println("===== ColossusX Unified PoW+AI-DAG v1 Generator =====")
	fmt.Printf("output       : %s\n", cfg.OutPath)
	fmt.Printf("metadata     : %s\n", cfg.MetaPath)
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

	if dir := filepath.Dir(cfg.OutPath); dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	if _, err := os.Stat(cfg.OutPath); err == nil && !cfg.Force {
		return fmt.Errorf("output exists: %s ; use --force", cfg.OutPath)
	}

	out, err := os.OpenFile(cfg.OutPath, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer out.Close()

	if err := out.Truncate(int64(totalBytes)); err != nil {
		return fmt.Errorf("truncate: %w", err)
	}

	start := time.Now()
	var written uint64

	jobs := make(chan Job, cfg.Workers*8)
	results := make(chan Result, cfg.Workers*8)

	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				page, err := buildUnifiedPage(cfg, manifest, job)
				results <- Result{PageIndex: job.PageIndex, Data: page, Err: err}
			}
		}()
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
		atomic.AddUint64(&written, cfg.PageSize)
	}
	close(progressDone)

	if err := <-producerErr; err != nil {
		return err
	}
	if firstErr != nil {
		return firstErr
	}

	if err := out.Sync(); err != nil {
		return fmt.Errorf("sync: %w", err)
	}

	fmt.Println()
	fmt.Println("computing BLAKE3 Merkle roots from generated AI-DAG...")
	roots, err := computeRootsFromFile(cfg.OutPath, cfg.PageSize)
	if err != nil {
		return err
	}

	manifest.AIDagRoot = "0x" + fmtHash(roots.AIDagRoot)
	manifest.TensorRoot = "0x" + fmtHash(roots.TensorRoot)
	manifest.PoWCommitRoot = "0x" + fmtHash(roots.PoWCommitRoot)

	manifest.AIDagLeafCount = roots.AIDagLeafCount
	manifest.TensorLeafCount = roots.TensorLeafCount
	manifest.PoWCommitLeafCount = roots.PoWCommitLeafCount

	manifest.ManifestRoot = "0x" + fmtHash(hashManifestRoot(manifest))

	if err := writeManifest(cfg.MetaPath, manifest); err != nil {
		return err
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

	if cfg.Verify {
		return verifyDagAgainstManifest(cfg.OutPath, cfg.MetaPath)
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
			Data:         payload,
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

	// Sanity check. pageCommitHash ignores the PageHash slot, so this must stay stable.
	if got := pageCommitHash(page); got != pageCommit {
		return nil, fmt.Errorf("unstable page commit at page %d", job.PageIndex)
	}

	return page, nil
}

func buildIndexPayload(manifest Manifest, max int) ([]byte, error) {
	tmp := manifest
	tmp.AIDagRoot = ""
	tmp.TensorRoot = ""
	tmp.PoWCommitRoot = ""
	tmp.ManifestRoot = ""

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

		hdr, err := readHeader(page[:unifiedHeaderSize])
		if err != nil {
			return Roots{}, fmt.Errorf("bad page %d: %w", pageIndex, err)
		}
		if err := validatePageHeader(hdr, pageIndex, pageSize); err != nil {
			return Roots{}, fmt.Errorf("bad page %d: %w", pageIndex, err)
		}

		payload := page[unifiedHeaderSize:]
		payloadHash := hashBytes(payload)
		if payloadHash != hdr.PayloadHash {
			return Roots{}, fmt.Errorf("payload hash mismatch at page %d", pageIndex)
		}

		pageCommit := pageCommitHash(page)
		if pageCommit != hdr.PageHash {
			return Roots{}, fmt.Errorf("page commit mismatch at page %d", pageIndex)
		}

		aidagLeaf := merkleLeafHash(treeAIDag, hdr, payloadHash, pageCommit)
		aidag.AddLeaf(aidagLeaf)

		if hdr.PageType == PageTypeModel {
			tensorLeaf := merkleLeafHash(treeTensor, hdr, payloadHash, pageCommit)
			tensor.AddLeaf(tensorLeaf)
		}

		// Every page, including model shard pages and index page, is PoW-mixable.
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

func NewMerkleAccumulator(treeID string) *MerkleAccumulator {
	return &MerkleAccumulator{
		treeID: treeID,
	}
}

func (m *MerkleAccumulator) Count() uint64 {
	return m.count
}

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

	// Canonical non-duplicating Merkle root:
	// split by largest power-of-two subtree.
	//
	// Example:
	//   3 leaves => H(H(0,1), 2)
	//   5 leaves => H(H(H(0,1), H(2,3)), 4)
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

	return verifyDagAgainstManifest(dag, meta)
}

func verifyDagAgainstManifest(dagPath, metaPath string) error {
	manifest, err := readManifest(metaPath)
	if err != nil {
		return err
	}

	if manifest.HashAlgorithm != "" && manifest.HashAlgorithm != hashAlgorithm {
		return fmt.Errorf("unsupported hash algorithm: meta=%s want=%s", manifest.HashAlgorithm, hashAlgorithm)
	}
	if manifest.MerkleVersion != 0 && manifest.MerkleVersion != merkleVersion {
		return fmt.Errorf("unsupported merkle version: meta=%d want=%d", manifest.MerkleVersion, merkleVersion)
	}
	if manifest.HeaderSize != unifiedHeaderSize {
		return fmt.Errorf("unsupported header size: meta=%d want=%d", manifest.HeaderSize, unifiedHeaderSize)
	}

	if manifest.ManifestRoot != "" {
		gotManifestRoot := "0x" + fmtHash(hashManifestRoot(manifest))
		if manifest.ManifestRoot != gotManifestRoot {
			return fmt.Errorf("ManifestRoot mismatch: meta=%s got=%s", manifest.ManifestRoot, gotManifestRoot)
		}
	}

	roots, err := computeRootsFromFile(dagPath, manifest.PageSize)
	if err != nil {
		return err
	}

	gotAIDag := "0x" + fmtHash(roots.AIDagRoot)
	gotTensor := "0x" + fmtHash(roots.TensorRoot)
	gotPoW := "0x" + fmtHash(roots.PoWCommitRoot)

	if manifest.AIDagRoot != "" && manifest.AIDagRoot != gotAIDag {
		return fmt.Errorf("AIDagRoot mismatch: meta=%s got=%s", manifest.AIDagRoot, gotAIDag)
	}
	if manifest.TensorRoot != "" && manifest.TensorRoot != gotTensor {
		return fmt.Errorf("TensorRoot mismatch: meta=%s got=%s", manifest.TensorRoot, gotTensor)
	}
	if manifest.PoWCommitRoot != "" && manifest.PoWCommitRoot != gotPoW {
		return fmt.Errorf("PoWCommitRoot mismatch: meta=%s got=%s", manifest.PoWCommitRoot, gotPoW)
	}

	if manifest.AIDagLeafCount != 0 && manifest.AIDagLeafCount != roots.AIDagLeafCount {
		return fmt.Errorf("AIDagLeafCount mismatch: meta=%d got=%d", manifest.AIDagLeafCount, roots.AIDagLeafCount)
	}
	if manifest.TensorLeafCount != roots.TensorLeafCount {
		return fmt.Errorf("TensorLeafCount mismatch: meta=%d got=%d", manifest.TensorLeafCount, roots.TensorLeafCount)
	}
	if manifest.PoWCommitLeafCount != 0 && manifest.PoWCommitLeafCount != roots.PoWCommitLeafCount {
		return fmt.Errorf("PoWCommitLeafCount mismatch: meta=%d got=%d", manifest.PoWCommitLeafCount, roots.PoWCommitLeafCount)
	}

	fmt.Println("verify OK")
	fmt.Printf("AIDagRoot          : %s\n", gotAIDag)
	fmt.Printf("TensorRoot         : %s\n", gotTensor)
	fmt.Printf("PoWCommitRoot      : %s\n", gotPoW)
	fmt.Printf("AIDagLeafCount     : %d\n", roots.AIDagLeafCount)
	fmt.Printf("TensorLeafCount    : %d\n", roots.TensorLeafCount)
	fmt.Printf("PoWCommitLeafCount : %d\n", roots.PoWCommitLeafCount)

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

	if dir := filepath.Dir(outPath); dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	out, err := os.OpenFile(outPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer out.Close()

	page := make([]byte, manifest.PageSize)
	var written uint64

	for p := manifest.ModelStartPage; p < manifest.ModelEndPage; p++ {
		n, err := in.ReadAt(page, int64(p*manifest.PageSize))
		if err != nil {
			return err
		}
		if uint64(n) != manifest.PageSize {
			return io.ErrUnexpectedEOF
		}

		hdr, err := readHeader(page[:manifest.HeaderSize])
		if err != nil {
			return err
		}
		if err := validatePageHeader(hdr, p, manifest.PageSize); err != nil {
			return err
		}
		if hdr.PageType != PageTypeModel {
			return fmt.Errorf("expected model page at %d, got type %d", p, hdr.PageType)
		}

		payload := page[manifest.HeaderSize:]

		payloadHash := hashBytes(payload)
		if payloadHash != hdr.PayloadHash {
			return fmt.Errorf("payload hash mismatch at model page %d", p)
		}
		pageCommit := pageCommitHash(page)
		if pageCommit != hdr.PageHash {
			return fmt.Errorf("page commit mismatch at model page %d", p)
		}

		toWrite := manifest.ModelSize - written
		if toWrite > uint64(len(payload)) {
			toWrite = uint64(len(payload))
		}
		if toWrite == 0 {
			break
		}

		nw, err := out.Write(payload[:toWrite])
		if err != nil {
			return err
		}
		if uint64(nw) != toWrite {
			return io.ErrShortWrite
		}

		written += toWrite
	}

	if written != manifest.ModelSize {
		return fmt.Errorf("extract size mismatch: wrote=%d want=%d", written, manifest.ModelSize)
	}

	if err := out.Sync(); err != nil {
		return err
	}

	hash, size, err := hashFile(outPath)
	if err != nil {
		return err
	}

	got := "0x" + fmtHash(hash)
	if size != manifest.ModelSize {
		return fmt.Errorf("extracted size mismatch: got=%d want=%d", size, manifest.ModelSize)
	}
	if manifest.ModelHash != "" && manifest.ModelHash != got {
		return fmt.Errorf("extracted model hash mismatch: got=%s want=%s", got, manifest.ModelHash)
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

	return os.WriteFile(path, b, 0644)
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
	// ManifestRoot must commit AIDagRoot, TensorRoot, PoWCommitRoot,
	// and all layout metadata. Only ManifestRoot itself is cleared.
	m.ManifestRoot = ""

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

func finalizeHash(h interface {
	Sum([]byte) []byte
}) [32]byte {
	sum := h.Sum(nil)

	var out [32]byte
	copy(out[:], sum)

	return out
}

func writeString(w io.Writer, s string) {
	writeBytes(w, []byte(s))
}

func writeBytes(w io.Writer, b []byte) {
	writeUint64(w, uint64(len(b)))
	_, _ = w.Write(b)
}

func writeFixed32(w io.Writer, h [32]byte) {
	_, _ = w.Write(h[:])
}

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

func fmtHash(h [32]byte) string {
	return fmt.Sprintf("%x", h[:])
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
	switch filepath.Ext(path) {
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
