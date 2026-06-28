package aiseal

const (
	UnifiedHeaderSize = uint64(160)
	FormatVersion     = uint16(2)
	ManifestVersion   = 2
	MerkleVersion     = 1
	SealVersion       = uint16(1)
	ProofVersion      = 1

	HashAlgorithm = "BLAKE3-256"

	PageTypeIndex  = uint16(0x0001)
	PageTypeModel  = uint16(0x0002)
	PageTypeFiller = uint16(0x0003)
)

const (
	treeTensor = "AIDG_MERKLE_TREE_TENSOR_V1"
	treePoW    = "AIDG_MERKLE_TREE_POW_COMMIT_V1"

	domainHashBytes    = "AIDG_HASH_BYTES_V1"
	domainPageCommit   = "AIDG_PAGE_COMMIT_V1"
	domainMerkleLeaf   = "AIDG_MERKLE_LEAF_V1"
	domainMerkleNode   = "AIDG_MERKLE_NODE_V1"
	domainManifest     = "AIDG_MANIFEST_ROOT_V1"
	domainSamplePoW    = "AIDG_SAMPLE_POW_V1"
	domainSampleTensor = "AIDG_SAMPLE_TENSOR_V1"
	domainMixDigest    = "AIDG_MIX_DIGEST_V1"
	domainAIDigest     = "AIDG_AI_DIGEST_V1"
	domainProofHash    = "AIDG_PROOF_HASH_V1"
	domainSealWork     = "AIDG_SEAL_WORK_HASH_V1"

	proofSideLeft  = "left"
	proofSideRight = "right"
)

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
	ModelEndPage   uint64 `json:"modelEndPage,omitempty"`

	IndexPage     uint64 `json:"indexPage"`
	FillerStart   uint64 `json:"fillerStart,omitempty"`
	FillerEnd     uint64 `json:"fillerEnd,omitempty"`
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
	PageIndex   uint64
	ModelOffset uint64
	ModelSize   uint64
	ShardID     uint64
	ShardCount  uint64
	TensorID    uint32
	LayerID     uint32
	ShardID2    uint32
	QuantID     uint32
	PayloadHash [32]byte
	PageHash    [32]byte
	Reserved    [16]byte
}

type MerkleProofStep struct {
	Side string `json:"side"`
	Hash string `json:"hash"`
}

type PageProof struct {
	PageIndex  uint64            `json:"pageIndex"`
	Page       string            `json:"page"`
	PoWPath    []MerkleProofStep `json:"powPath"`
	TensorPath []MerkleProofStep `json:"tensorPath,omitempty"`
}

type Seal struct {
	Version uint16 `json:"version"`
	Epoch   uint64 `json:"epoch"`

	BlockHash string `json:"blockHash"`
	Miner     string `json:"miner"`
	Nonce     uint64 `json:"nonce"`

	PoWSampleCount    int `json:"powSampleCount"`
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

type Proof struct {
	Version int         `json:"version"`
	Seal    Seal        `json:"seal"`
	Pages   []PageProof `json:"pages"`
}

type VerificationResult struct {
	WorkHash  string `json:"workHash"`
	ProofHash string `json:"proofHash"`
	AIDigest  string `json:"aiDigest"`
	MixDigest string `json:"mixDigest"`
	PageCount int    `json:"pageCount"`
}

type Limits struct {
	MaxProofBytes int
	MaxPages      int
	MaxPageSize   uint64
}

func DefaultLimits() Limits {
	return Limits{MaxProofBytes: 8 << 20, MaxPages: 256, MaxPageSize: 1 << 20}
}

type pageMaterial struct {
	PageIndex   uint64
	Header      PageHeader
	Page        []byte
	Payload     []byte
	PayloadHash [32]byte
	PageCommit  [32]byte
	PoWLeaf     [32]byte
	TensorLeaf  [32]byte
}

type SidecarLevel struct {
	Level  int    `json:"level"`
	Count  uint64 `json:"count"`
	Offset uint64 `json:"offset"`
	Bytes  uint64 `json:"bytes"`
}

type SidecarManifest struct {
	Version        int            `json:"version"`
	HashAlgorithm  string         `json:"hashAlgorithm"`
	MerkleVersion  int            `json:"merkleVersion"`
	CreatedAt      string         `json:"createdAt"`
	PageSize       uint64         `json:"pageSize"`
	HeaderSize     uint64         `json:"headerSize"`
	PayloadSize    uint64         `json:"payloadSize"`
	TotalPages     uint64         `json:"totalPages"`
	ModelStartPage uint64         `json:"modelStartPage"`
	ModelPageCount uint64         `json:"modelPageCount"`
	ManifestRoot   string         `json:"manifestRoot"`
	AIDagRoot      string         `json:"aidagRoot"`
	TensorRoot     string         `json:"tensorRoot"`
	PoWCommitRoot  string         `json:"powCommitRoot"`
	PoWLevels      []SidecarLevel `json:"powLevels"`
	TensorLevels   []SidecarLevel `json:"tensorLevels"`
}
