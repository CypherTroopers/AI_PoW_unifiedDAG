package aiseal

import (
	"encoding/binary"
	"encoding/hex"
	"io"
	"testing"
)

func TestLightVerifierAcceptsRealPageAndMerkleProofs(t *testing.T) {
	manifest, proof := syntheticArtifact(t)
	result, err := Verify(manifest, proof, "", DefaultLimits())
	if err != nil {
		t.Fatalf("valid proof rejected: %v", err)
	}
	if result.PageCount == 0 || !equalHex(result.ProofHash, proof.Seal.ProofHash) {
		t.Fatalf("unexpected verification result: %+v", result)
	}
	if err := VerifyExpected(proof, result, ExpectedChallenge{
		BlockHash: "0x010203", Miner: "node0", Epoch: 1, ManifestRoot: manifest.ManifestRoot,
		MinPoWSamples: 2, MinTensorSamples: 1,
	}); err != nil {
		t.Fatal(err)
	}
}

func TestLightVerifierRejectsTamperedPage(t *testing.T) {
	manifest, proof := syntheticArtifact(t)
	raw, err := decodeHexBytes(proof.Pages[0].Page)
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-1] ^= 0xff
	proof.Pages[0].Page = "0x" + hex.EncodeToString(raw)
	if _, err := Verify(manifest, proof, "", DefaultLimits()); err == nil {
		t.Fatal("tampered page was accepted")
	}
}

func syntheticArtifact(t *testing.T) (Manifest, Proof) {
	t.Helper()
	const pageSize = uint64(256)
	pages := make([][]byte, 4)
	materials := make([]pageMaterial, 4)
	powLeaves := make([][32]byte, 4)
	var tensorLeaves [][32]byte
	for index := range pages {
		pageType := PageTypeFiller
		if index == 0 {
			pageType = PageTypeIndex
		} else if index == 1 {
			pageType = PageTypeModel
		}
		pages[index] = makeSyntheticPage(uint64(index), pageType, pageSize)
		header, payloadHash, pageCommit, err := verifyPage(pages[index], uint64(index), pageSize)
		if err != nil {
			t.Fatal(err)
		}
		materials[index] = pageMaterial{
			PageIndex: uint64(index), Header: header, Page: pages[index], Payload: pages[index][UnifiedHeaderSize:],
			PayloadHash: payloadHash, PageCommit: pageCommit,
			PoWLeaf: merkleLeafHash(treePoW, header, payloadHash, pageCommit),
		}
		powLeaves[index] = materials[index].PoWLeaf
		if pageType == PageTypeModel {
			materials[index].TensorLeaf = merkleLeafHash(treeTensor, header, payloadHash, pageCommit)
			tensorLeaves = append(tensorLeaves, materials[index].TensorLeaf)
		}
	}
	powLevels := buildLevels(powLeaves)
	tensorLevels := buildLevels(tensorLeaves)
	manifest := Manifest{
		Version: ManifestVersion, Name: "synthetic", HashAlgorithm: HashAlgorithm, MerkleVersion: MerkleVersion,
		PageSize: pageSize, HeaderSize: UnifiedHeaderSize, PayloadSize: pageSize - UnifiedHeaderSize,
		TotalBytes: pageSize * 4, TotalPages: 4, AIDagRoot: formatHash(taggedHash("test-aidag", nil)),
		TensorRoot: formatHash(tensorLevels[len(tensorLevels)-1][0]), PoWCommitRoot: formatHash(powLevels[len(powLevels)-1][0]),
		AIDagLeafCount: 4, TensorLeafCount: 1, PoWCommitLeafCount: 4,
		ModelHash: formatHash(taggedHash("test-model", nil)), ModelSize: pageSize - UnifiedHeaderSize,
		ModelStartPage: 1, ModelPageCount: 1, ModelEndPage: 2, IndexPage: 0, FillerStart: 2, FillerEnd: 4,
	}
	manifest.ManifestRoot = formatHash(ManifestHash(manifest))
	seal := Seal{
		Version: SealVersion, Epoch: 1, BlockHash: "0x010203", Miner: "node0", Nonce: 7,
		PoWSampleCount: 2, TensorSampleCount: 1, ManifestRoot: manifest.ManifestRoot,
		AIDagRoot: manifest.AIDagRoot, TensorRoot: manifest.TensorRoot, PoWRoot: manifest.PoWCommitRoot, ModelHash: manifest.ModelHash,
	}
	powSamples, tensorSamples, required, err := deriveRequiredPages(manifest, seal)
	if err != nil {
		t.Fatal(err)
	}
	proofPages := make([]PageProof, 0, len(required))
	requiredMaterials := make([]pageMaterial, 0, len(required))
	for _, index := range required {
		page := PageProof{PageIndex: index, Page: "0x" + hex.EncodeToString(pages[index]), PoWPath: proofFor(powLevels, index)}
		if containsSorted(tensorSamples, index) {
			page.TensorPath = proofFor(tensorLevels, index-manifest.ModelStartPage)
		}
		proofPages = append(proofPages, page)
		requiredMaterials = append(requiredMaterials, materials[index])
	}
	seal.MixDigest = formatHash(computeMixDigest(seal, manifest, powSamples, tensorSamples, requiredMaterials))
	seal.AIDigest = formatHash(computeAIDigest(seal, manifest, tensorSamples, requiredMaterials))
	seal.WorkHash = formatHash(computeWorkHash(seal))
	proof := Proof{Version: ProofVersion, Seal: seal, Pages: proofPages}
	proof.Seal.ProofHash = formatHash(computeProofHash(proof))
	return manifest, proof
}

func makeSyntheticPage(index uint64, pageType uint16, pageSize uint64) []byte {
	payload := make([]byte, pageSize-UnifiedHeaderSize)
	for i := range payload {
		payload[i] = byte(index*17 + uint64(i))
	}
	header := PageHeader{
		Magic: [4]byte{'A', 'I', 'D', 'G'}, Version: FormatVersion, PageType: pageType,
		HeaderSize: uint32(UnifiedHeaderSize), PageSize: uint32(pageSize), PayloadSize: uint32(len(payload)), PageIndex: index,
	}
	if pageType == PageTypeModel {
		header.ModelSize = uint64(len(payload))
		header.ShardID = 0
		header.ShardCount = 1
	}
	header.PayloadHash = taggedHash(domainHashBytes, func(w io.Writer) { writeBytes(w, payload) })
	page := make([]byte, pageSize)
	writeSyntheticHeader(page, header)
	copy(page[UnifiedHeaderSize:], payload)
	header.PageHash = taggedHash(domainPageCommit, func(w io.Writer) {
		writeBytes(w, page[:pageHashOffset])
		writeFixed32(w, [32]byte{})
		writeBytes(w, page[pageHashEnd:])
	})
	writeSyntheticHeader(page, header)
	return page
}

func writeSyntheticHeader(dst []byte, h PageHeader) {
	copy(dst[:4], h.Magic[:])
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

func buildLevels(leaves [][32]byte) [][][32]byte {
	levels := [][][32]byte{append([][32]byte(nil), leaves...)}
	for len(levels[len(levels)-1]) > 1 {
		current := levels[len(levels)-1]
		next := make([][32]byte, (len(current)+1)/2)
		for i := 0; i < len(current); i += 2 {
			next[i/2] = current[i]
			if i+1 < len(current) {
				next[i/2] = merkleNodeHash(treeForTestLeaves(leaves), current[i], current[i+1])
			}
		}
		levels = append(levels, next)
	}
	return levels
}

func treeForTestLeaves(leaves [][32]byte) string {
	if len(leaves) == 1 {
		return treeTensor
	}
	return treePoW
}

func proofFor(levels [][][32]byte, index uint64) []MerkleProofStep {
	treeID := treeForTestLeaves(levels[0])
	proof := make([]MerkleProofStep, 0, len(levels)-1)
	idx := int(index)
	for level := 0; level < len(levels)-1; level++ {
		if idx%2 == 0 && idx+1 < len(levels[level]) {
			proof = append(proof, MerkleProofStep{Side: proofSideRight, Hash: formatHash(levels[level][idx+1])})
		} else if idx%2 == 1 {
			proof = append(proof, MerkleProofStep{Side: proofSideLeft, Hash: formatHash(levels[level][idx-1])})
		}
		idx /= 2
	}
	_ = treeID
	return proof
}
