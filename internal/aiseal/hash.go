package aiseal

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/zeebo/blake3"
)

const (
	pageHashOffset = 112
	pageHashEnd    = 144
	magicAIDG      = "AIDG"
)

func canonicalManifestForRoot(m Manifest) Manifest {
	m.ManifestRoot = ""
	m.GenerationTime = ""
	m.ModelName = ""
	m.ModelFormat = ""
	m.UnifiedLayout = ""
	return m
}

func ManifestHash(m Manifest) [32]byte {
	m = canonicalManifestForRoot(m)
	b, _ := json.Marshal(m)
	return taggedHash(domainManifest, func(w io.Writer) { writeBytes(w, b) })
}

func readHeader(src []byte) (PageHeader, error) {
	var h PageHeader
	if len(src) < int(UnifiedHeaderSize) {
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

func validateHeader(h PageHeader, pageIndex, pageSize uint64) error {
	if h.Version != FormatVersion {
		return fmt.Errorf("unsupported page version: got=%d want=%d", h.Version, FormatVersion)
	}
	if uint64(h.HeaderSize) != UnifiedHeaderSize || uint64(h.PageSize) != pageSize {
		return errors.New("page/header size mismatch")
	}
	if uint64(h.PayloadSize) != pageSize-UnifiedHeaderSize || h.PageIndex != pageIndex {
		return errors.New("payload size or page index mismatch")
	}
	switch h.PageType {
	case PageTypeIndex, PageTypeModel, PageTypeFiller:
		return nil
	default:
		return fmt.Errorf("unknown page type: %d", h.PageType)
	}
}

func verifyPage(page []byte, pageIndex, pageSize uint64) (PageHeader, [32]byte, [32]byte, error) {
	var zero [32]byte
	if uint64(len(page)) != pageSize || pageSize < UnifiedHeaderSize {
		return PageHeader{}, zero, zero, fmt.Errorf("bad page size at %d", pageIndex)
	}
	h, err := readHeader(page[:UnifiedHeaderSize])
	if err != nil {
		return PageHeader{}, zero, zero, err
	}
	if err := validateHeader(h, pageIndex, pageSize); err != nil {
		return PageHeader{}, zero, zero, err
	}
	payloadHash := taggedHash(domainHashBytes, func(w io.Writer) { writeBytes(w, page[UnifiedHeaderSize:]) })
	if payloadHash != h.PayloadHash {
		return PageHeader{}, zero, zero, fmt.Errorf("payload hash mismatch at page %d", pageIndex)
	}
	pageCommit := taggedHash(domainPageCommit, func(w io.Writer) {
		writeBytes(w, page[:pageHashOffset])
		writeFixed32(w, zero)
		writeBytes(w, page[pageHashEnd:])
	})
	if pageCommit != h.PageHash {
		return PageHeader{}, zero, zero, fmt.Errorf("page commit mismatch at page %d", pageIndex)
	}
	return h, payloadHash, pageCommit, nil
}

func merkleLeafHash(treeID string, h PageHeader, payloadHash, pageCommit [32]byte) [32]byte {
	return taggedHash(domainMerkleLeaf, func(w io.Writer) {
		writeString(w, treeID)
		writeUint16(w, h.Version)
		writeUint16(w, h.PageType)
		writeUint32(w, h.HeaderSize)
		writeUint32(w, h.PageSize)
		writeUint32(w, h.PayloadSize)
		writeUint32(w, h.Flags)
		writeUint64(w, h.PageIndex)
		writeUint64(w, h.ModelOffset)
		writeUint64(w, h.ModelSize)
		writeUint64(w, h.ShardID)
		writeUint64(w, h.ShardCount)
		writeUint32(w, h.TensorID)
		writeUint32(w, h.LayerID)
		writeUint32(w, h.ShardID2)
		writeUint32(w, h.QuantID)
		writeFixed32(w, payloadHash)
		writeFixed32(w, pageCommit)
	})
}

func merkleNodeHash(treeID string, left, right [32]byte) [32]byte {
	return taggedHash(domainMerkleNode, func(w io.Writer) {
		writeString(w, treeID)
		writeFixed32(w, left)
		writeFixed32(w, right)
	})
}

func verifyMerkleProof(treeID string, leaf [32]byte, leafIndex, leafCount uint64, proof []MerkleProofStep, root [32]byte) error {
	if leafCount == 0 || leafIndex >= leafCount {
		return errors.New("invalid merkle leaf index/count")
	}
	node, idx, count, pos := leaf, leafIndex, leafCount, 0
	for count > 1 {
		if idx%2 == 0 {
			if idx+1 < count {
				if pos >= len(proof) || proof[pos].Side != proofSideRight {
					return errors.New("missing or invalid right merkle sibling")
				}
				sibling, err := decodeHex32(proof[pos].Hash)
				if err != nil {
					return err
				}
				pos++
				node = merkleNodeHash(treeID, node, sibling)
			}
		} else {
			if pos >= len(proof) || proof[pos].Side != proofSideLeft {
				return errors.New("missing or invalid left merkle sibling")
			}
			sibling, err := decodeHex32(proof[pos].Hash)
			if err != nil {
				return err
			}
			pos++
			node = merkleNodeHash(treeID, sibling, node)
		}
		idx /= 2
		count = (count + 1) / 2
	}
	if pos != len(proof) || node != root {
		return errors.New("merkle proof root mismatch or unused steps")
	}
	return nil
}

func taggedHash(domain string, fn func(io.Writer)) [32]byte {
	h := blake3.New()
	writeString(h, domain)
	if fn != nil {
		fn(h)
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func decodeHexBytes(value string) ([]byte, error) {
	value = strings.TrimSpace(value)
	value = strings.TrimPrefix(strings.TrimPrefix(value, "0x"), "0X")
	if value == "" || len(value)%2 != 0 {
		return nil, errors.New("invalid empty or odd-length hex")
	}
	return hex.DecodeString(value)
}

func decodeHex32(value string) ([32]byte, error) {
	var out [32]byte
	b, err := decodeHexBytes(value)
	if err != nil {
		return out, err
	}
	if len(b) != 32 {
		return out, fmt.Errorf("expected 32 bytes, got %d", len(b))
	}
	copy(out[:], b)
	return out, nil
}

func formatHash(h [32]byte) string { return "0x" + hex.EncodeToString(h[:]) }

func normalizeHex(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" || strings.HasPrefix(value, "0x") {
		return value
	}
	return "0x" + value
}

func normalizeArbitrary(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return ""
	}
	raw := strings.TrimPrefix(value, "0x")
	if _, err := hex.DecodeString(raw); err == nil && len(raw)%2 == 0 {
		return "0x" + raw
	}
	return value
}

func equalHex(a, b string) bool { return normalizeHex(a) == normalizeHex(b) }

func hashMeetsTarget(hash, target [32]byte) bool { return bytes.Compare(hash[:], target[:]) <= 0 }

func WorkMeetsTarget(workHash, target string) (bool, error) {
	work, err := decodeHex32(workHash)
	if err != nil {
		return false, err
	}
	targetHash, err := decodeHex32(target)
	if err != nil {
		return false, err
	}
	return hashMeetsTarget(work, targetHash), nil
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
