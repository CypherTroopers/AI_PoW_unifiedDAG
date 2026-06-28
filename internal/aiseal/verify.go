package aiseal

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

type ExpectedChallenge struct {
	BlockHash        string
	Miner            string
	Epoch            uint64
	ManifestRoot     string
	MinPoWSamples    int
	MinTensorSamples int
}

func LoadManifestBytes(data []byte) (Manifest, error) {
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, err
	}
	if err := ValidateManifest(manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func ValidateManifest(m Manifest) error {
	if m.Version != 0 && m.Version != ManifestVersion {
		return fmt.Errorf("unsupported manifest version: got=%d want=%d", m.Version, ManifestVersion)
	}
	if m.HashAlgorithm != HashAlgorithm || m.MerkleVersion != MerkleVersion {
		return errors.New("unsupported manifest hash or merkle version")
	}
	if m.HeaderSize != UnifiedHeaderSize || m.PageSize <= m.HeaderSize || m.PayloadSize != m.PageSize-m.HeaderSize {
		return errors.New("invalid manifest page layout")
	}
	if m.TotalPages == 0 || m.TotalBytes != m.TotalPages*m.PageSize {
		return errors.New("invalid manifest total page/byte count")
	}
	if m.ModelStartPage == 0 || m.ModelPageCount == 0 || m.ModelEndPage != m.ModelStartPage+m.ModelPageCount || m.ModelEndPage > m.TotalPages {
		return errors.New("invalid manifest model page range")
	}
	if m.AIDagLeafCount != m.TotalPages || m.PoWCommitLeafCount != m.TotalPages || m.TensorLeafCount != m.ModelPageCount {
		return errors.New("manifest leaf counts do not match layout")
	}
	for name, value := range map[string]string{
		"AIDagRoot": m.AIDagRoot, "TensorRoot": m.TensorRoot, "PoWCommitRoot": m.PoWCommitRoot,
		"ManifestRoot": m.ManifestRoot, "ModelHash": m.ModelHash,
	} {
		h, err := decodeHex32(value)
		if err != nil {
			return fmt.Errorf("invalid %s: %w", name, err)
		}
		if h == ([32]byte{}) {
			return fmt.Errorf("%s is zero", name)
		}
	}
	if !equalHex(m.ManifestRoot, formatHash(ManifestHash(m))) {
		return errors.New("manifest root does not match canonical manifest")
	}
	return nil
}

func VerifyBytes(manifest Manifest, proofBytes []byte, target string, limits Limits) (VerificationResult, Proof, error) {
	if limits.MaxProofBytes <= 0 {
		limits = DefaultLimits()
	}
	if len(proofBytes) > limits.MaxProofBytes {
		return VerificationResult{}, Proof{}, fmt.Errorf("AISeal proof is too large: %d > %d", len(proofBytes), limits.MaxProofBytes)
	}
	var proof Proof
	if err := json.Unmarshal(proofBytes, &proof); err != nil {
		return VerificationResult{}, Proof{}, err
	}
	result, err := Verify(manifest, proof, target, limits)
	return result, proof, err
}

func Verify(manifest Manifest, proof Proof, target string, limits Limits) (VerificationResult, error) {
	if err := ValidateManifest(manifest); err != nil {
		return VerificationResult{}, err
	}
	if limits.MaxPages <= 0 {
		limits = DefaultLimits()
	}
	if manifest.PageSize > limits.MaxPageSize || len(proof.Pages) > limits.MaxPages {
		return VerificationResult{}, errors.New("AISeal proof exceeds verifier resource limits")
	}
	if proof.Version != ProofVersion || proof.Seal.Version != SealVersion {
		return VerificationResult{}, errors.New("unsupported AISeal proof or seal version")
	}
	if proof.Seal.PoWSampleCount <= 0 || proof.Seal.TensorSampleCount < 0 {
		return VerificationResult{}, errors.New("invalid AISeal sample counts")
	}
	if err := verifySealMatchesManifest(proof.Seal, manifest); err != nil {
		return VerificationResult{}, err
	}
	powRoot, err := decodeHex32(manifest.PoWCommitRoot)
	if err != nil {
		return VerificationResult{}, err
	}
	tensorRoot, err := decodeHex32(manifest.TensorRoot)
	if err != nil {
		return VerificationResult{}, err
	}

	powSamples, tensorSamples, required, err := deriveRequiredPages(manifest, proof.Seal)
	if err != nil {
		return VerificationResult{}, err
	}
	if len(proof.Pages) != len(required) {
		return VerificationResult{}, fmt.Errorf("proof page count mismatch: got=%d want=%d", len(proof.Pages), len(required))
	}
	byIndex := make(map[uint64]PageProof, len(proof.Pages))
	for _, page := range proof.Pages {
		if _, exists := byIndex[page.PageIndex]; exists {
			return VerificationResult{}, fmt.Errorf("duplicate proof page %d", page.PageIndex)
		}
		byIndex[page.PageIndex] = page
	}

	materials := make([]pageMaterial, 0, len(required))
	for _, index := range required {
		page, ok := byIndex[index]
		if !ok {
			return VerificationResult{}, fmt.Errorf("missing proof page %d", index)
		}
		material, err := materialFromProof(page, manifest)
		if err != nil {
			return VerificationResult{}, err
		}
		if err := verifyMerkleProof(treePoW, material.PoWLeaf, index, manifest.PoWCommitLeafCount, page.PoWPath, powRoot); err != nil {
			return VerificationResult{}, fmt.Errorf("PoW merkle proof for page %d: %w", index, err)
		}
		if containsSorted(tensorSamples, index) {
			if material.Header.PageType != PageTypeModel {
				return VerificationResult{}, fmt.Errorf("tensor sample %d is not a model page", index)
			}
			tensorIndex := index - manifest.ModelStartPage
			if material.Header.ShardID != tensorIndex {
				return VerificationResult{}, fmt.Errorf("model shard mismatch at page %d", index)
			}
			if err := verifyMerkleProof(treeTensor, material.TensorLeaf, tensorIndex, manifest.TensorLeafCount, page.TensorPath, tensorRoot); err != nil {
				return VerificationResult{}, fmt.Errorf("tensor merkle proof for page %d: %w", index, err)
			}
		} else if len(page.TensorPath) != 0 {
			return VerificationResult{}, fmt.Errorf("unexpected tensor path for page %d", index)
		}
		materials = append(materials, material)
	}

	mixDigest := computeMixDigest(proof.Seal, manifest, powSamples, tensorSamples, materials)
	if !equalHex(proof.Seal.MixDigest, formatHash(mixDigest)) {
		return VerificationResult{}, errors.New("AISeal mixDigest mismatch")
	}
	aiDigest := computeAIDigest(proof.Seal, manifest, tensorSamples, materials)
	if !equalHex(proof.Seal.AIDigest, formatHash(aiDigest)) {
		return VerificationResult{}, errors.New("AISeal aiDigest mismatch")
	}
	workHash := computeWorkHash(proof.Seal)
	if proof.Seal.WorkHash != "" && !equalHex(proof.Seal.WorkHash, formatHash(workHash)) {
		return VerificationResult{}, errors.New("AISeal workHash mismatch")
	}
	proofHash := computeProofHash(proof)
	if !equalHex(proof.Seal.ProofHash, formatHash(proofHash)) {
		return VerificationResult{}, errors.New("AISeal proofHash mismatch")
	}
	if target != "" {
		targetHash, err := decodeHex32(target)
		if err != nil {
			return VerificationResult{}, fmt.Errorf("invalid target: %w", err)
		}
		if !hashMeetsTarget(workHash, targetHash) {
			return VerificationResult{}, errors.New("AISeal workHash does not meet target")
		}
	}
	return VerificationResult{
		WorkHash: formatHash(workHash), ProofHash: formatHash(proofHash), AIDigest: formatHash(aiDigest),
		MixDigest: formatHash(mixDigest), PageCount: len(proof.Pages),
	}, nil
}

func VerifyExpected(proof Proof, result VerificationResult, expected ExpectedChallenge) error {
	if expected.BlockHash != "" && normalizeArbitrary(proof.Seal.BlockHash) != normalizeArbitrary(expected.BlockHash) {
		return errors.New("AISeal block challenge mismatch")
	}
	if expected.Miner != "" && strings.ToLower(strings.TrimSpace(proof.Seal.Miner)) != strings.ToLower(strings.TrimSpace(expected.Miner)) {
		return errors.New("AISeal miner mismatch")
	}
	if expected.Epoch != 0 && proof.Seal.Epoch != expected.Epoch {
		return errors.New("AISeal epoch mismatch")
	}
	if expected.ManifestRoot != "" && !equalHex(proof.Seal.ManifestRoot, expected.ManifestRoot) {
		return errors.New("AISeal manifest mismatch")
	}
	if proof.Seal.PoWSampleCount < expected.MinPoWSamples || proof.Seal.TensorSampleCount < expected.MinTensorSamples {
		return errors.New("AISeal proof has insufficient samples")
	}
	if !equalHex(proof.Seal.ProofHash, result.ProofHash) || !equalHex(proof.Seal.AIDigest, result.AIDigest) {
		return errors.New("AISeal result binding mismatch")
	}
	return nil
}

func verifySealMatchesManifest(seal Seal, manifest Manifest) error {
	checks := [][2]string{
		{seal.ManifestRoot, manifest.ManifestRoot}, {seal.AIDagRoot, manifest.AIDagRoot},
		{seal.TensorRoot, manifest.TensorRoot}, {seal.PoWRoot, manifest.PoWCommitRoot}, {seal.ModelHash, manifest.ModelHash},
	}
	for _, check := range checks {
		if !equalHex(check[0], check[1]) {
			return errors.New("AISeal does not match manifest")
		}
	}
	return nil
}

func materialFromProof(pageProof PageProof, manifest Manifest) (pageMaterial, error) {
	page, err := decodeHexBytes(pageProof.Page)
	if err != nil {
		return pageMaterial{}, err
	}
	if uint64(len(page)) != manifest.PageSize {
		return pageMaterial{}, fmt.Errorf("bad page size for proof page %d", pageProof.PageIndex)
	}
	header, payloadHash, pageCommit, err := verifyPage(page, pageProof.PageIndex, manifest.PageSize)
	if err != nil {
		return pageMaterial{}, err
	}
	payload := append([]byte(nil), page[manifest.HeaderSize:]...)
	material := pageMaterial{
		PageIndex: pageProof.PageIndex, Header: header, Page: page, Payload: payload,
		PayloadHash: payloadHash, PageCommit: pageCommit,
		PoWLeaf: merkleLeafHash(treePoW, header, payloadHash, pageCommit),
	}
	if header.PageType == PageTypeModel {
		material.TensorLeaf = merkleLeafHash(treeTensor, header, payloadHash, pageCommit)
	}
	return material, nil
}

func deriveRequiredPages(manifest Manifest, seal Seal) ([]uint64, []uint64, []uint64, error) {
	if uint64(seal.PoWSampleCount) > manifest.TotalPages || uint64(seal.TensorSampleCount) > manifest.ModelPageCount {
		return nil, nil, nil, errors.New("AISeal sample count exceeds manifest")
	}
	pow := sampleUnique(domainSamplePoW, manifest, seal, uint64(seal.PoWSampleCount), 0, manifest.TotalPages)
	tensor := sampleUnique(domainSampleTensor, manifest, seal, uint64(seal.TensorSampleCount), manifest.ModelStartPage, manifest.ModelPageCount)
	set := make(map[uint64]struct{}, len(pow)+len(tensor))
	for _, index := range pow {
		set[index] = struct{}{}
	}
	for _, index := range tensor {
		set[index] = struct{}{}
	}
	required := make([]uint64, 0, len(set))
	for index := range set {
		required = append(required, index)
	}
	sort.Slice(required, func(i, j int) bool { return required[i] < required[j] })
	return pow, tensor, required, nil
}

func sampleUnique(domain string, manifest Manifest, seal Seal, count, start, span uint64) []uint64 {
	if count == 0 || span == 0 {
		return nil
	}
	seed := taggedHash(domain, func(w io.Writer) {
		writeString(w, normalizeArbitrary(seal.BlockHash))
		writeString(w, strings.ToLower(strings.TrimSpace(seal.Miner)))
		writeUint64(w, seal.Epoch)
		writeUint64(w, seal.Nonce)
		writeString(w, normalizeHex(manifest.ManifestRoot))
		writeString(w, normalizeHex(manifest.PoWCommitRoot))
		writeString(w, normalizeHex(manifest.TensorRoot))
		writeUint64(w, manifest.TotalPages)
		writeUint64(w, manifest.ModelStartPage)
		writeUint64(w, manifest.ModelPageCount)
		writeUint64(w, count)
		writeUint64(w, start)
		writeUint64(w, span)
	})
	out := make([]uint64, 0, count)
	seen := make(map[uint64]struct{}, count)
	for counter := uint64(0); uint64(len(out)) < count; counter++ {
		h := taggedHash(domain, func(w io.Writer) {
			writeFixed32(w, seed)
			writeUint64(w, counter)
		})
		index := start + binary.LittleEndian.Uint64(h[:8])%span
		if _, exists := seen[index]; exists {
			continue
		}
		seen[index] = struct{}{}
		out = append(out, index)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func containsSorted(values []uint64, target uint64) bool {
	i := sort.Search(len(values), func(i int) bool { return values[i] >= target })
	return i < len(values) && values[i] == target
}

func computeMixDigest(seal Seal, manifest Manifest, powSamples, tensorSamples []uint64, materials []pageMaterial) [32]byte {
	byIndex := make(map[uint64]pageMaterial, len(materials))
	for _, material := range materials {
		byIndex[material.PageIndex] = material
	}
	return taggedHash(domainMixDigest, func(w io.Writer) {
		writeSealFields(w, seal)
		writeString(w, normalizeHex(manifest.ManifestRoot))
		writeUint64(w, manifest.TotalPages)
		writeUint64(w, manifest.ModelStartPage)
		writeUint64(w, manifest.ModelPageCount)
		writeUint64(w, uint64(len(powSamples)))
		for _, index := range powSamples {
			material := byIndex[index]
			writeUint64(w, index)
			writeUint16(w, material.Header.PageType)
			writeFixed32(w, material.PayloadHash)
			writeFixed32(w, material.PageCommit)
			writeFixed32(w, material.PoWLeaf)
		}
		writeUint64(w, uint64(len(tensorSamples)))
		for _, index := range tensorSamples {
			material := byIndex[index]
			writeUint64(w, index)
			writeUint64(w, material.Header.ShardID)
			writeUint64(w, material.Header.ModelOffset)
			writeUint64(w, material.Header.ModelSize)
			writeFixed32(w, material.PayloadHash)
			writeFixed32(w, material.PageCommit)
			writeFixed32(w, material.TensorLeaf)
		}
	})
}

func computeAIDigest(seal Seal, manifest Manifest, tensorSamples []uint64, materials []pageMaterial) [32]byte {
	byIndex := make(map[uint64]pageMaterial, len(materials))
	for _, material := range materials {
		byIndex[material.PageIndex] = material
	}
	return taggedHash(domainAIDigest, func(w io.Writer) {
		writeSealFields(w, seal)
		writeString(w, normalizeHex(manifest.ModelHash))
		writeUint64(w, manifest.ModelSize)
		writeUint64(w, manifest.ModelPageCount)
		writeUint64(w, uint64(len(tensorSamples)))
		for _, index := range tensorSamples {
			material := byIndex[index]
			writeUint64(w, index)
			writeUint64(w, material.Header.ShardID)
			writeUint64(w, material.Header.ModelOffset)
			writeUint64(w, material.Header.ModelSize)
			writeFixed32(w, material.PayloadHash)
			writeFixed32(w, material.PageCommit)
			writeFixed32(w, tensorChallenge(seal, index, material.Payload))
		}
	})
}

func tensorChallenge(seal Seal, index uint64, payload []byte) [32]byte {
	return taggedHash(domainAIDigest, func(w io.Writer) {
		writeSealFields(w, seal)
		writeUint64(w, index)
		writeUint64(w, uint64(len(payload)))
		if len(payload) == 0 {
			return
		}
		seed := taggedHash(domainAIDigest, func(sw io.Writer) {
			writeSealFields(sw, seal)
			writeUint64(sw, index)
		})
		for i := uint64(0); i < 32; i++ {
			h := taggedHash(domainAIDigest, func(hw io.Writer) {
				writeFixed32(hw, seed)
				writeUint64(hw, i)
			})
			offset := int(binary.LittleEndian.Uint64(h[:8]) % uint64(len(payload)))
			end := offset + 32
			if end <= len(payload) {
				writeBytes(w, payload[offset:end])
			} else {
				wrapped := append(append([]byte(nil), payload[offset:]...), payload[:end-len(payload)]...)
				writeBytes(w, wrapped)
			}
		}
	})
}

func writeSealFields(w io.Writer, seal Seal) {
	writeUint16(w, seal.Version)
	writeUint64(w, seal.Epoch)
	writeString(w, normalizeArbitrary(seal.BlockHash))
	writeString(w, strings.ToLower(strings.TrimSpace(seal.Miner)))
	writeUint64(w, seal.Nonce)
	writeUint64(w, uint64(seal.PoWSampleCount))
	writeUint64(w, uint64(seal.TensorSampleCount))
	writeString(w, normalizeHex(seal.ManifestRoot))
	writeString(w, normalizeHex(seal.AIDagRoot))
	writeString(w, normalizeHex(seal.TensorRoot))
	writeString(w, normalizeHex(seal.PoWRoot))
	writeString(w, normalizeHex(seal.ModelHash))
}

func computeWorkHash(seal Seal) [32]byte {
	return taggedHash(domainSealWork, func(w io.Writer) {
		writeSealFields(w, seal)
		writeString(w, normalizeHex(seal.MixDigest))
		writeString(w, normalizeHex(seal.AIDigest))
	})
}

func computeProofHash(proof Proof) [32]byte {
	proof.Seal.ProofHash = ""
	proof.Pages = append([]PageProof(nil), proof.Pages...)
	sort.Slice(proof.Pages, func(i, j int) bool { return proof.Pages[i].PageIndex < proof.Pages[j].PageIndex })
	return taggedHash(domainProofHash, func(w io.Writer) {
		writeUint64(w, uint64(proof.Version))
		writeSealFields(w, proof.Seal)
		writeString(w, normalizeHex(proof.Seal.MixDigest))
		writeString(w, normalizeHex(proof.Seal.AIDigest))
		writeString(w, normalizeHex(proof.Seal.WorkHash))
		writeUint64(w, uint64(len(proof.Pages)))
		for _, page := range proof.Pages {
			writeUint64(w, page.PageIndex)
			writeString(w, normalizeHex(page.Page))
			writeProofPath(w, page.PoWPath)
			writeProofPath(w, page.TensorPath)
		}
	})
}

func writeProofPath(w io.Writer, path []MerkleProofStep) {
	writeUint64(w, uint64(len(path)))
	for _, step := range path {
		writeString(w, step.Side)
		writeString(w, normalizeHex(step.Hash))
	}
}
