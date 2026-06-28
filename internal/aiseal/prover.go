package aiseal

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

const (
	sidecarMagic          = "AIDGSC1!"
	sidecarVersion        = 1
	sidecarHeaderReserved = uint64(1 << 20)
	hashSize              = uint64(32)
)

type ProveConfig struct {
	DAGPath       string
	SidecarPath   string
	Manifest      Manifest
	BlockHash     string
	Miner         string
	Epoch         uint64
	Nonce         uint64
	PoWSamples    int
	TensorSamples int
	Target        string
}

func LoadManifestFile(path string) (Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, err
	}
	return LoadManifestBytes(data)
}

func ProveFiles(ctx context.Context, cfg ProveConfig) (Proof, []byte, error) {
	if cfg.DAGPath == "" || cfg.BlockHash == "" || cfg.Miner == "" {
		return Proof{}, nil, errors.New("DAG path, block hash, and miner are required")
	}
	if cfg.PoWSamples <= 0 || cfg.TensorSamples < 0 {
		return Proof{}, nil, errors.New("invalid proof sample counts")
	}
	if err := ValidateManifest(cfg.Manifest); err != nil {
		return Proof{}, nil, err
	}
	if cfg.SidecarPath == "" {
		cfg.SidecarPath = cfg.DAGPath + ".sidecar"
	}
	sidecar, err := openSidecar(cfg.SidecarPath)
	if err != nil {
		return Proof{}, nil, err
	}
	defer sidecar.Close()
	if err := sidecar.validateManifest(cfg.Manifest); err != nil {
		return Proof{}, nil, err
	}
	dag, err := os.Open(cfg.DAGPath)
	if err != nil {
		return Proof{}, nil, err
	}
	defer dag.Close()
	stat, err := dag.Stat()
	if err != nil {
		return Proof{}, nil, err
	}
	if uint64(stat.Size()) != cfg.Manifest.TotalBytes {
		return Proof{}, nil, errors.New("DAG size does not match manifest")
	}

	seal := Seal{
		Version: SealVersion, Epoch: cfg.Epoch, BlockHash: normalizeArbitrary(cfg.BlockHash),
		Miner: strings.ToLower(strings.TrimSpace(cfg.Miner)), Nonce: cfg.Nonce,
		PoWSampleCount: cfg.PoWSamples, TensorSampleCount: cfg.TensorSamples,
		ManifestRoot: normalizeHex(cfg.Manifest.ManifestRoot), AIDagRoot: normalizeHex(cfg.Manifest.AIDagRoot),
		TensorRoot: normalizeHex(cfg.Manifest.TensorRoot), PoWRoot: normalizeHex(cfg.Manifest.PoWCommitRoot),
		ModelHash: normalizeHex(cfg.Manifest.ModelHash),
	}
	powSamples, tensorSamples, required, err := deriveRequiredPages(cfg.Manifest, seal)
	if err != nil {
		return Proof{}, nil, err
	}
	pages := make([]PageProof, 0, len(required))
	materials := make([]pageMaterial, 0, len(required))
	for _, index := range required {
		select {
		case <-ctx.Done():
			return Proof{}, nil, ctx.Err()
		default:
		}
		material, err := readMaterialAt(dag, cfg.Manifest, index)
		if err != nil {
			return Proof{}, nil, err
		}
		powPath, err := sidecar.Proof(treePoW, index)
		if err != nil {
			return Proof{}, nil, err
		}
		page := PageProof{PageIndex: index, Page: "0x" + hex.EncodeToString(material.Page), PoWPath: powPath}
		if containsSorted(tensorSamples, index) {
			tensorPath, err := sidecar.Proof(treeTensor, index-cfg.Manifest.ModelStartPage)
			if err != nil {
				return Proof{}, nil, err
			}
			page.TensorPath = tensorPath
		}
		pages = append(pages, page)
		materials = append(materials, material)
	}
	seal.MixDigest = formatHash(computeMixDigest(seal, cfg.Manifest, powSamples, tensorSamples, materials))
	seal.AIDigest = formatHash(computeAIDigest(seal, cfg.Manifest, tensorSamples, materials))
	workHash := computeWorkHash(seal)
	seal.WorkHash = formatHash(workHash)
	if cfg.Target != "" {
		target, err := decodeHex32(cfg.Target)
		if err != nil || !hashMeetsTarget(workHash, target) {
			return Proof{}, nil, errors.New("generated AISeal does not meet target")
		}
	}
	proof := Proof{Version: ProofVersion, Seal: seal, Pages: pages}
	proof.Seal.ProofHash = formatHash(computeProofHash(proof))
	data, err := json.Marshal(proof)
	if err != nil {
		return Proof{}, nil, err
	}
	return proof, data, nil
}

func readMaterialAt(file *os.File, manifest Manifest, index uint64) (pageMaterial, error) {
	if index >= manifest.TotalPages {
		return pageMaterial{}, errors.New("proof page index outside DAG")
	}
	page := make([]byte, manifest.PageSize)
	n, err := file.ReadAt(page, int64(index*manifest.PageSize))
	if err != nil {
		return pageMaterial{}, err
	}
	if uint64(n) != manifest.PageSize {
		return pageMaterial{}, io.ErrUnexpectedEOF
	}
	header, payloadHash, pageCommit, err := verifyPage(page, index, manifest.PageSize)
	if err != nil {
		return pageMaterial{}, err
	}
	material := pageMaterial{
		PageIndex: index, Header: header, Page: page, Payload: append([]byte(nil), page[manifest.HeaderSize:]...),
		PayloadHash: payloadHash, PageCommit: pageCommit, PoWLeaf: merkleLeafHash(treePoW, header, payloadHash, pageCommit),
	}
	if header.PageType == PageTypeModel {
		material.TensorLeaf = merkleLeafHash(treeTensor, header, payloadHash, pageCommit)
	}
	return material, nil
}

type sidecarReader struct {
	file *os.File
	meta SidecarManifest
}

func openSidecar(path string) (*sidecarReader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	prefix := make([]byte, 16)
	if _, err := file.ReadAt(prefix, 0); err != nil {
		file.Close()
		return nil, err
	}
	if string(prefix[:8]) != sidecarMagic {
		file.Close()
		return nil, errors.New("bad sidecar magic")
	}
	headerLength := binary.LittleEndian.Uint64(prefix[8:16])
	if headerLength == 0 || headerLength+16 > sidecarHeaderReserved {
		file.Close()
		return nil, errors.New("bad sidecar header length")
	}
	header := make([]byte, headerLength)
	if _, err := file.ReadAt(header, 16); err != nil {
		file.Close()
		return nil, err
	}
	var meta SidecarManifest
	if err := json.Unmarshal(header, &meta); err != nil {
		file.Close()
		return nil, err
	}
	reader := &sidecarReader{file: file, meta: meta}
	if err := reader.validateSelf(); err != nil {
		file.Close()
		return nil, err
	}
	return reader, nil
}

func (r *sidecarReader) Close() error { return r.file.Close() }

func (r *sidecarReader) validateSelf() error {
	if r.meta.Version != sidecarVersion || r.meta.HashAlgorithm != HashAlgorithm || r.meta.MerkleVersion != MerkleVersion {
		return errors.New("unsupported sidecar format")
	}
	for name, levels := range map[string][]SidecarLevel{treePoW: r.meta.PoWLevels, treeTensor: r.meta.TensorLevels} {
		if len(levels) == 0 {
			return fmt.Errorf("sidecar %s has no levels", name)
		}
		for index, level := range levels {
			if level.Level != index || level.Count == 0 || level.Bytes != level.Count*hashSize {
				return fmt.Errorf("invalid sidecar %s level %d", name, index)
			}
			if index > 0 && level.Count != (levels[index-1].Count+1)/2 {
				return fmt.Errorf("invalid sidecar %s level count", name)
			}
		}
		if levels[len(levels)-1].Count != 1 {
			return fmt.Errorf("sidecar %s has no root", name)
		}
	}
	return nil
}

func (r *sidecarReader) validateManifest(manifest Manifest) error {
	if r.meta.PageSize != manifest.PageSize || r.meta.HeaderSize != manifest.HeaderSize ||
		r.meta.PayloadSize != manifest.PayloadSize || r.meta.TotalPages != manifest.TotalPages ||
		r.meta.ModelStartPage != manifest.ModelStartPage || r.meta.ModelPageCount != manifest.ModelPageCount {
		return errors.New("sidecar layout does not match manifest")
	}
	for _, check := range [][2]string{
		{r.meta.ManifestRoot, manifest.ManifestRoot}, {r.meta.AIDagRoot, manifest.AIDagRoot},
		{r.meta.TensorRoot, manifest.TensorRoot}, {r.meta.PoWCommitRoot, manifest.PoWCommitRoot},
	} {
		if !equalHex(check[0], check[1]) {
			return errors.New("sidecar root does not match manifest")
		}
	}
	if r.meta.PoWLevels[0].Count != manifest.PoWCommitLeafCount || r.meta.TensorLevels[0].Count != manifest.TensorLeafCount {
		return errors.New("sidecar leaf count does not match manifest")
	}
	return nil
}

func (r *sidecarReader) Proof(treeID string, leafIndex uint64) ([]MerkleProofStep, error) {
	var levels []SidecarLevel
	switch treeID {
	case treePoW:
		levels = r.meta.PoWLevels
	case treeTensor:
		levels = r.meta.TensorLevels
	default:
		return nil, errors.New("unknown sidecar tree")
	}
	if leafIndex >= levels[0].Count {
		return nil, errors.New("sidecar leaf index out of range")
	}
	index := leafIndex
	proof := make([]MerkleProofStep, 0, len(levels)-1)
	for levelIndex := 0; levelIndex < len(levels)-1; levelIndex++ {
		level := levels[levelIndex]
		if index%2 == 0 && index+1 < level.Count {
			sibling, err := r.readHash(level, index+1)
			if err != nil {
				return nil, err
			}
			proof = append(proof, MerkleProofStep{Side: proofSideRight, Hash: formatHash(sibling)})
		} else if index%2 == 1 {
			sibling, err := r.readHash(level, index-1)
			if err != nil {
				return nil, err
			}
			proof = append(proof, MerkleProofStep{Side: proofSideLeft, Hash: formatHash(sibling)})
		}
		index /= 2
	}
	return proof, nil
}

func (r *sidecarReader) readHash(level SidecarLevel, index uint64) ([32]byte, error) {
	var out [32]byte
	if index >= level.Count {
		return out, errors.New("sidecar hash index out of range")
	}
	_, err := r.file.ReadAt(out[:], int64(level.Offset+index*hashSize))
	return out, err
}

func sortedUnique(values []uint64) []uint64 {
	set := make(map[uint64]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	out := make([]uint64, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
