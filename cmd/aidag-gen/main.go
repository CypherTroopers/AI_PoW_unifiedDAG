package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// Page types are used only for non-model pages.
// Model region is raw bytes without per-page headers so that it can be mmap/read
// as a contiguous GGUF/safetensors/bin blob.
const (
	PageTypePoW   = uint8(0x00)
	PageTypeIndex = uint8(0x02)
)

const (
	defaultPageSize = 4096
	oneGiB         = uint64(1024 * 1024 * 1024)

	// Page header is used for Index and PoW pages only.
	// Model pages DO NOT use this header.
	pageHeaderSize = 16
)

type Config struct {
	OutPath       string
	SizeGB        uint64
	PageSize      uint64
	Seed          string
	Workers       int
	ChunkMB       uint64
	Force         bool
	Verify        bool
	IndexPath     string
	ModelPath     string
	ModelOffsetGB uint64 // where raw model bytes begin; default 32 GiB
}

type PageJob struct {
	StartPage uint64
	PageCount uint64
	Offset    uint64
	Size      uint64
	Type      uint8
	Data      []byte // used for raw model chunks
}

type PageResult struct {
	Job    PageJob
	Data   []byte
	Err    error
	DoneAt time.Time
}

type DAGMeta struct {
	Version    int    `json:"version"`
	Name       string `json:"name"`
	SizeGB     uint64 `json:"sizeGB"`
	TotalBytes uint64 `json:"totalBytes"`
	PageSize   uint64 `json:"pageSize"`
	TotalPages uint64 `json:"totalPages"`
	Seed       string `json:"seed"`

	// This is intentionally empty inside page 0 during root calculation,
	// and written only to the external .meta file after root calculation.
	AIDagRoot string `json:"aidagRoot"`

	ModelHash      string `json:"modelHash,omitempty"`
	ModelSize      uint64 `json:"modelSize,omitempty"`
	ModelName      string `json:"modelName,omitempty"`
	ModelFormat    string `json:"modelFormat,omitempty"`
	ModelOffset    uint64 `json:"modelOffset,omitempty"`
	ModelEndOffset uint64 `json:"modelEndOffset,omitempty"`
	ModelStartPage uint64 `json:"modelStartPage,omitempty"`
	ModelEndPage   uint64 `json:"modelEndPage,omitempty"` // exclusive

	IndexStartPage uint64 `json:"indexStartPage"`
	IndexEndPage   uint64 `json:"indexEndPage"`
	PoWStartPage   uint64 `json:"powStartPage"`
	PoWEndPage     uint64 `json:"powEndPage"`
	PoW2StartPage  uint64 `json:"pow2StartPage,omitempty"`
	PoW2EndPage    uint64 `json:"pow2EndPage,omitempty"`
}

func main() {
	var cfg Config

	flag.StringVar(&cfg.OutPath, "out", "./aidag-128g.bin", "output AI-DAG file path")
	flag.Uint64Var(&cfg.SizeGB, "size-gb", 128, "AI-DAG size in GiB")
	flag.Uint64Var(&cfg.PageSize, "page-size", defaultPageSize, "page size in bytes")
	flag.StringVar(&cfg.Seed, "seed", "colossusx-ai-dag-v1", "deterministic seed")
	flag.IntVar(&cfg.Workers, "workers", runtime.NumCPU(), "number of generator workers")
	flag.Uint64Var(&cfg.ChunkMB, "chunk-mb", 64, "write chunk size in MiB")
	flag.BoolVar(&cfg.Force, "force", false, "overwrite existing output file")
	flag.BoolVar(&cfg.Verify, "verify", false, "verify file root after generation")
	flag.StringVar(&cfg.IndexPath, "index", "", "optional output metadata/index file path")
	flag.StringVar(&cfg.ModelPath, "model", "", "path to LLM model file, preferably GGUF")
	flag.Uint64Var(&cfg.ModelOffsetGB, "model-offset-gb", 32, "raw model offset in GiB inside AI-DAG")
	flag.Parse()

	if err := validateConfig(&cfg); err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	if cfg.IndexPath == "" {
		cfg.IndexPath = cfg.OutPath + ".meta"
	}

	if err := generateAIDAG(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "generation failed: %v\n", err)
		os.Exit(1)
	}
}

func validateConfig(cfg *Config) error {
	if cfg.OutPath == "" {
		return fmt.Errorf("out path is required")
	}
	if cfg.SizeGB == 0 {
		return fmt.Errorf("size-gb must be > 0")
	}
	if cfg.PageSize == 0 || cfg.PageSize%64 != 0 {
		return fmt.Errorf("page-size must be non-zero and multiple of 64")
	}
	if cfg.Workers <= 0 {
		cfg.Workers = runtime.NumCPU()
	}
	if cfg.ChunkMB == 0 {
		cfg.ChunkMB = 64
	}

	totalSize := cfg.SizeGB * oneGiB
	if totalSize%cfg.PageSize != 0 {
		return fmt.Errorf("total size must be divisible by page size")
	}
	chunkSize := cfg.ChunkMB * 1024 * 1024
	if chunkSize%cfg.PageSize != 0 {
		return fmt.Errorf("chunk size must be divisible by page size")
	}

	if _, err := os.Stat(cfg.OutPath); err == nil && !cfg.Force {
		return fmt.Errorf("output file already exists: %s ; use --force to overwrite", cfg.OutPath)
	}

	modelOffset := cfg.ModelOffsetGB * oneGiB
	if modelOffset%cfg.PageSize != 0 {
		return fmt.Errorf("model offset must be page aligned")
	}
	if modelOffset >= totalSize {
		return fmt.Errorf("model offset %d is outside DAG size %d", modelOffset, totalSize)
	}

	if cfg.ModelPath != "" {
		fi, err := os.Stat(cfg.ModelPath)
		if err != nil {
			return fmt.Errorf("model file not found: %s", cfg.ModelPath)
		}
		if fi.Size() <= 0 {
			return fmt.Errorf("model file is empty: %s", cfg.ModelPath)
		}
		if modelOffset+uint64(fi.Size()) > totalSize {
			return fmt.Errorf("model does not fit: offset=%d size=%d total=%d", modelOffset, fi.Size(), totalSize)
		}
	}

	return nil
}

func generateAIDAG(cfg Config) error {
	totalSize := cfg.SizeGB * oneGiB
	totalPages := totalSize / cfg.PageSize
	modelOffset := cfg.ModelOffsetGB * oneGiB
	modelStartPage := modelOffset / cfg.PageSize

	fmt.Println("===== ColossusX AI-DAG Generator (Raw Model Region) =====")
	fmt.Printf("output        : %s\n", cfg.OutPath)
	fmt.Printf("metadata      : %s\n", cfg.IndexPath)
	fmt.Printf("size          : %d GiB\n", cfg.SizeGB)
	fmt.Printf("total bytes   : %d\n", totalSize)
	fmt.Printf("page size     : %d bytes\n", cfg.PageSize)
	fmt.Printf("total pages   : %d\n", totalPages)
	fmt.Printf("workers       : %d\n", cfg.Workers)
	fmt.Printf("seed          : %s\n", cfg.Seed)
	fmt.Printf("model offset  : %d bytes (%d GiB)\n", modelOffset, cfg.ModelOffsetGB)
	if cfg.ModelPath != "" {
		fmt.Printf("model         : %s\n", cfg.ModelPath)
	} else {
		fmt.Println("model         : (none, PoW only)")
	}
	fmt.Println("===========================================================")

	if dir := filepath.Dir(cfg.OutPath); dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}

	file, err := os.OpenFile(cfg.OutPath, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	if err := file.Truncate(int64(totalSize)); err != nil {
		return fmt.Errorf("truncate output file: %w", err)
	}

	var modelHash [32]byte
	var modelSize uint64
	var modelName string
	var modelEndOffset uint64
	var modelEndPage uint64

	if cfg.ModelPath != "" {
		fmt.Println("computing model hash...")
		modelHash, modelSize, err = hashFile(cfg.ModelPath)
		if err != nil {
			return fmt.Errorf("hash model: %w", err)
		}
		modelName = filepath.Base(cfg.ModelPath)
		modelEndOffset = modelOffset + modelSize
		modelEndPage = (modelEndOffset + cfg.PageSize - 1) / cfg.PageSize
		fmt.Printf("model hash    : %x\n", modelHash)
		fmt.Printf("model size    : %d bytes\n", modelSize)
		fmt.Printf("model pages   : %d .. %d exclusive\n", modelStartPage, modelEndPage)
	}

	meta := DAGMeta{
		Version:        2,
		Name:           "ColossusX AI-DAG",
		SizeGB:         cfg.SizeGB,
		TotalBytes:     totalSize,
		PageSize:       cfg.PageSize,
		TotalPages:     totalPages,
		Seed:           cfg.Seed,
		IndexStartPage: 0,
		IndexEndPage:   1,
		PoWStartPage:   1,
		PoWEndPage:     modelStartPage,
	}
	if cfg.ModelPath != "" {
		meta.ModelHash = fmt.Sprintf("0x%x", modelHash)
		meta.ModelSize = modelSize
		meta.ModelName = modelName
		meta.ModelFormat = guessModelFormat(modelName)
		meta.ModelOffset = modelOffset
		meta.ModelEndOffset = modelEndOffset
		meta.ModelStartPage = modelStartPage
		meta.ModelEndPage = modelEndPage
		meta.PoW2StartPage = modelEndPage
		meta.PoW2EndPage = totalPages
	}

	jobs := make(chan PageJob, cfg.Workers*4)
	results := make(chan PageResult, cfg.Workers*4)
	var written uint64
	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				var data []byte
				var err error
				switch job.Type {
				case PageTypeIndex:
					data, err = buildIndexPage(cfg.PageSize, meta)
				case PageTypePoW:
					data, err = buildPoWPages(cfg.Seed, job, cfg.PageSize)
				default:
					// Raw model chunk already has final bytes.
					data = job.Data
				}
				results <- PageResult{Job: job, Data: data, Err: err, DoneAt: time.Now()}
			}
		}()
	}

	errCh := make(chan error, 1)
	go func() {
		defer close(errCh)

		jobs <- PageJob{StartPage: 0, PageCount: 1, Offset: 0, Size: cfg.PageSize, Type: PageTypeIndex}

		// PoW before model region.
		if err := enqueuePoWJobs(jobs, cfg, 1, modelStartPage); err != nil {
			errCh <- err
			close(jobs)
			wg.Wait()
			close(results)
			return
		}

		// Raw model region. No page headers.
		if cfg.ModelPath != "" {
			if err := enqueueRawModelJobs(jobs, cfg, modelOffset, modelSize); err != nil {
				errCh <- err
				close(jobs)
				wg.Wait()
				close(results)
				return
			}
			// PoW after padded model page range.
			if err := enqueuePoWJobs(jobs, cfg, modelEndPage, totalPages); err != nil {
				errCh <- err
				close(jobs)
				wg.Wait()
				close(results)
				return
			}
		} else {
			if err := enqueuePoWJobs(jobs, cfg, modelStartPage, totalPages); err != nil {
				errCh <- err
				close(jobs)
				wg.Wait()
				close(results)
				return
			}
		}

		close(jobs)
		wg.Wait()
		close(results)
	}()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	done := make(chan struct{})
	var firstErr atomic.Value

	go func() {
		for {
			select {
			case <-ticker.C:
				printProgress(start, atomic.LoadUint64(&written), totalSize)
			case <-done:
				return
			}
		}
	}()

	for res := range results {
		if res.Err != nil {
			firstErr.Store(res.Err)
			continue
		}
		n, err := file.WriteAt(res.Data, int64(res.Job.Offset))
		if err != nil {
			firstErr.Store(err)
			continue
		}
		if uint64(n) != res.Job.Size {
			firstErr.Store(io.ErrShortWrite)
			continue
		}
		atomic.AddUint64(&written, res.Job.Size)
	}
	close(done)

	if err := <-errCh; err != nil {
		return err
	}
	if v := firstErr.Load(); v != nil {
		return v.(error)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync output file: %w", err)
	}

	printProgress(start, atomic.LoadUint64(&written), totalSize)
	fmt.Println()

	root, err := computeSyntheticRoot(cfg, meta)
	if err != nil {
		return err
	}
	meta.AIDagRoot = fmt.Sprintf("0x%x", root)
	if err := writeMetaFile(cfg.IndexPath, meta); err != nil {
		return err
	}

	fmt.Println("AI-DAG generated successfully")
	fmt.Printf("AIDagRoot     : %x\n", root)
	fmt.Printf("output        : %s\n", cfg.OutPath)
	fmt.Printf("metadata      : %s\n", cfg.IndexPath)
	if cfg.ModelPath != "" {
		fmt.Printf("modelOffset   : %d\n", meta.ModelOffset)
		fmt.Printf("modelSize     : %d\n", meta.ModelSize)
		fmt.Println("model region  : raw contiguous bytes; mmap friendly")
	}

	if cfg.Verify {
		fmt.Println("verifying AI-DAG root from file...")
		verifyRoot, err := computeFileRoot(cfg.OutPath, cfg.PageSize)
		if err != nil {
			return err
		}
		fmt.Printf("VerifyRoot    : %x\n", verifyRoot)
		if root != verifyRoot {
			return fmt.Errorf("verify failed: root mismatch")
		}
		fmt.Println("verify OK")
	}

	return nil
}

func enqueuePoWJobs(jobs chan<- PageJob, cfg Config, startPage, endPage uint64) error {
	if startPage >= endPage {
		return nil
	}
	chunkSize := cfg.ChunkMB * 1024 * 1024
	chunkPages := chunkSize / cfg.PageSize
	if chunkPages == 0 {
		chunkPages = 1
	}
	for page := startPage; page < endPage; page += chunkPages {
		count := chunkPages
		if page+count > endPage {
			count = endPage - page
		}
		jobs <- PageJob{StartPage: page, PageCount: count, Offset: page * cfg.PageSize, Size: count * cfg.PageSize, Type: PageTypePoW}
	}
	return nil
}

func enqueueRawModelJobs(jobs chan<- PageJob, cfg Config, modelOffset, modelSize uint64) error {
	f, err := os.Open(cfg.ModelPath)
	if err != nil {
		return err
	}
	defer f.Close()

	chunkSize := cfg.ChunkMB * 1024 * 1024
	if chunkSize%cfg.PageSize != 0 {
		return fmt.Errorf("chunk size must be page aligned")
	}

	var readOffset uint64
	for readOffset < modelSize {
		remaining := modelSize - readOffset
		payload := chunkSize
		if remaining < payload {
			payload = remaining
		}

		// Write full pages so the final partial model page is zero-padded.
		writeSize := roundUp(payload, cfg.PageSize)
		buf := make([]byte, writeSize)
		n, err := io.ReadFull(f, buf[:payload])
		if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
			return err
		}
		if uint64(n) != payload {
			return fmt.Errorf("short model read: got=%d want=%d", n, payload)
		}

		jobs <- PageJob{Offset: modelOffset + readOffset, Size: writeSize, Data: buf, Type: 0xff}
		readOffset += payload
	}
	return nil
}

func buildIndexPage(pageSize uint64, meta DAGMeta) ([]byte, error) {
	page := make([]byte, pageSize)
	page[0] = PageTypeIndex
	binary.LittleEndian.PutUint64(page[8:16], 0)

	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	dataArea := page[pageHeaderSize:]
	if uint64(len(metaJSON)) > uint64(len(dataArea)) {
		return nil, fmt.Errorf("meta JSON too large for index page (%d > %d)", len(metaJSON), len(dataArea))
	}
	copy(dataArea, metaJSON)
	return page, nil
}

func buildPoWPages(seed string, job PageJob, pageSize uint64) ([]byte, error) {
	total := job.PageCount * pageSize
	out := make([]byte, total)

	for p := uint64(0); p < job.PageCount; p++ {
		pageIndex := job.StartPage + p
		pageOffset := p * pageSize
		page := out[pageOffset : pageOffset+pageSize]
		page[0] = PageTypePoW
		binary.LittleEndian.PutUint64(page[8:16], pageIndex)
		fillPageData(page[pageHeaderSize:], seed, pageIndex)
	}
	return out, nil
}

func fillPageData(data []byte, seed string, pageIndex uint64) {
	var counter uint64
	var header [8]byte
	var block [32]byte

	for offset := 0; offset < len(data); offset += len(block) {
		h := sha256.New()
		binary.LittleEndian.PutUint64(header[:], pageIndex)
		h.Write([]byte(seed))
		h.Write(header[:])
		binary.LittleEndian.PutUint64(header[:], counter)
		h.Write(header[:])
		sum := h.Sum(nil)
		copy(block[:], sum)
		n := copy(data[offset:], block[:])
		if n < len(block) {
			break
		}
		counter++
	}

	pageHash := sha256.Sum256(data)
	for i := 0; i < len(data); i++ {
		data[i] ^= pageHash[i%len(pageHash)]
	}
}

func computeSyntheticRoot(cfg Config, meta DAGMeta) ([32]byte, error) {
	fmt.Println("computing AI-DAG root hash (this may take a while)...")
	h := sha256.New()
	var tmp [8]byte
	pageBuf := make([]byte, cfg.PageSize)

	var modelFile *os.File
	var err error
	if cfg.ModelPath != "" {
		modelFile, err = os.Open(cfg.ModelPath)
		if err != nil {
			return [32]byte{}, err
		}
		defer modelFile.Close()
	}

	var modelBytesRead uint64
	for pageIndex := uint64(0); pageIndex < meta.TotalPages; pageIndex++ {
		zero(pageBuf)

		switch {
		case pageIndex == 0:
			p, err := buildIndexPage(cfg.PageSize, meta)
			if err != nil {
				return [32]byte{}, err
			}
			copy(pageBuf, p)

		case cfg.ModelPath != "" && pageIndex >= meta.ModelStartPage && pageIndex < meta.ModelEndPage:
			remaining := meta.ModelSize - modelBytesRead
			toRead := cfg.PageSize
			if remaining < toRead {
				toRead = remaining
			}
			if toRead > 0 {
				n, err := io.ReadFull(modelFile, pageBuf[:toRead])
				if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
					return [32]byte{}, err
				}
				if uint64(n) != toRead {
					return [32]byte{}, fmt.Errorf("short model read during root: got=%d want=%d", n, toRead)
				}
				modelBytesRead += uint64(n)
			}

		default:
			pageBuf[0] = PageTypePoW
			binary.LittleEndian.PutUint64(pageBuf[8:16], pageIndex)
			fillPageData(pageBuf[pageHeaderSize:], cfg.Seed, pageIndex)
		}

		pageHash := sha256.Sum256(pageBuf)
		binary.LittleEndian.PutUint64(tmp[:], pageIndex)
		h.Write(tmp[:])
		h.Write(pageHash[:])

		if pageIndex%1000000 == 0 && pageIndex != 0 {
			fmt.Printf("  root progress: page %d / %d\n", pageIndex, meta.TotalPages)
		}
	}

	var root [32]byte
	copy(root[:], h.Sum(nil))
	return root, nil
}

func computeFileRoot(path string, pageSize uint64) ([32]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return [32]byte{}, err
	}
	defer file.Close()

	h := sha256.New()
	buf := make([]byte, pageSize)
	var pageIndex uint64
	var tmp [8]byte

	for {
		n, err := io.ReadFull(file, buf)
		if err == io.EOF {
			break
		}
		if err == io.ErrUnexpectedEOF {
			return [32]byte{}, fmt.Errorf("file size is not multiple of page size")
		}
		if err != nil {
			return [32]byte{}, err
		}
		if uint64(n) != pageSize {
			return [32]byte{}, io.ErrShortBuffer
		}
		pageHash := sha256.Sum256(buf)
		binary.LittleEndian.PutUint64(tmp[:], pageIndex)
		h.Write(tmp[:])
		h.Write(pageHash[:])
		pageIndex++
		if pageIndex%1000000 == 0 {
			fmt.Printf("verify progress: page %d\n", pageIndex)
		}
	}

	var root [32]byte
	copy(root[:], h.Sum(nil))
	return root, nil
}

func hashFile(path string) ([32]byte, uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return [32]byte{}, 0, err
	}
	defer f.Close()

	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return [32]byte{}, 0, err
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out, uint64(n), nil
}

func writeMetaFile(path string, meta DAGMeta) error {
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

func printProgress(start time.Time, written, total uint64) {
	elapsed := time.Since(start).Seconds()
	if elapsed <= 0 {
		elapsed = 1
	}
	percent := float64(written) / float64(total) * 100
	mbs := float64(written) / 1024 / 1024 / elapsed
	remaining := 0.0
	if mbs > 0 {
		remaining = float64(total-written) / 1024 / 1024 / mbs
	}

	fmt.Printf("\rprogress: %.2f%%  written: %.2f GiB / %.2f GiB  speed: %.2f MiB/s  eta: %.0fs",
		percent,
		float64(written)/1024/1024/1024,
		float64(total)/1024/1024/1024,
		mbs,
		remaining,
	)
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

func zero(b []byte) {
	for i := range b {
		b[i] = 0
	}
}

func guessModelFormat(name string) string {
	switch filepath.Ext(name) {
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
