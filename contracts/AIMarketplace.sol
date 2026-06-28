// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title AI Marketplace for the AI-PoW HotStuff network
/// @notice Development contract. The consensusOracle should be a threshold-controlled account.
contract AIMarketplace {
    enum JobStatus { Open, Finalized, Cancelled }

    struct Model {
        address owner;
        bytes32 manifestRoot;
        string metadataURI;
        uint256 minimumReward;
        bool active;
    }

    struct Job {
        address requester;
        uint256 modelId;
        bytes32 inputHash;
        string inputURI;
        uint256 reward;
        uint64 deadline;
        JobStatus status;
        address worker;
        bytes32 outputHash;
        bytes32 aiSealProofHash;
        bytes32 executionBlockHash;
    }

    address public owner;
    address public consensusOracle;
    uint256 public modelCount;
    uint256 public jobCount;

    mapping(uint256 => Model) public models;
    mapping(uint256 => Job) public jobs;
    mapping(address => uint256) public credits;

    uint256 private unlocked = 1;

    event ModelRegistered(uint256 indexed modelId, address indexed modelOwner, bytes32 manifestRoot, string metadataURI, uint256 minimumReward);
    event ModelStatusChanged(uint256 indexed modelId, bool active);
    event JobSubmitted(uint256 indexed jobId, uint256 indexed modelId, address indexed requester, bytes32 inputHash, string inputURI, uint256 reward, uint64 deadline);
    event JobFinalized(uint256 indexed jobId, address indexed worker, bytes32 outputHash, bytes32 aiSealProofHash, bytes32 executionBlockHash);
    event JobCancelled(uint256 indexed jobId);
    event ConsensusOracleChanged(address indexed previousOracle, address indexed newOracle);

    error Unauthorized();
    error InvalidModel();
    error InvalidJob();
    error InsufficientReward();
    error TransferFailed();

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier onlyOracle() {
        if (msg.sender != consensusOracle) revert Unauthorized();
        _;
    }

    modifier nonReentrant() {
        require(unlocked == 1, "reentrant call");
        unlocked = 2;
        _;
        unlocked = 1;
    }

    constructor(address initialOracle) {
        require(initialOracle != address(0), "zero oracle");
        owner = msg.sender;
        consensusOracle = initialOracle;
    }

    function setConsensusOracle(address newOracle) external onlyOwner {
        require(newOracle != address(0), "zero oracle");
        emit ConsensusOracleChanged(consensusOracle, newOracle);
        consensusOracle = newOracle;
    }

    function registerModel(bytes32 manifestRoot, string calldata metadataURI, uint256 minimumReward) external returns (uint256 modelId) {
        require(manifestRoot != bytes32(0), "zero manifest root");
        modelId = ++modelCount;
        models[modelId] = Model(msg.sender, manifestRoot, metadataURI, minimumReward, true);
        emit ModelRegistered(modelId, msg.sender, manifestRoot, metadataURI, minimumReward);
    }

    function setModelActive(uint256 modelId, bool active) external {
        Model storage model = models[modelId];
        if (model.owner == address(0)) revert InvalidModel();
        if (msg.sender != model.owner && msg.sender != owner) revert Unauthorized();
        model.active = active;
        emit ModelStatusChanged(modelId, active);
    }

    function submitJob(uint256 modelId, bytes32 inputHash, string calldata inputURI, uint64 deadline) external payable returns (uint256 jobId) {
        Model storage model = models[modelId];
        if (!model.active) revert InvalidModel();
        if (msg.value < model.minimumReward) revert InsufficientReward();
        require(inputHash != bytes32(0) && deadline > block.timestamp, "invalid job data");

        jobId = ++jobCount;
        jobs[jobId] = Job({
            requester: msg.sender,
            modelId: modelId,
            inputHash: inputHash,
            inputURI: inputURI,
            reward: msg.value,
            deadline: deadline,
            status: JobStatus.Open,
            worker: address(0),
            outputHash: bytes32(0),
            aiSealProofHash: bytes32(0),
            executionBlockHash: bytes32(0)
        });
        emit JobSubmitted(jobId, modelId, msg.sender, inputHash, inputURI, msg.value, deadline);
    }

    /// @notice Called after HotStuff finalizes the block containing the AI receipt.
    function finalizeJob(
        uint256 jobId,
        address worker,
        bytes32 outputHash,
        bytes32 aiSealProofHash,
        bytes32 executionBlockHash
    ) external onlyOracle {
        Job storage job = jobs[jobId];
        if (job.status != JobStatus.Open || worker == address(0) || outputHash == bytes32(0) || aiSealProofHash == bytes32(0)) {
            revert InvalidJob();
        }
        job.status = JobStatus.Finalized;
        job.worker = worker;
        job.outputHash = outputHash;
        job.aiSealProofHash = aiSealProofHash;
        job.executionBlockHash = executionBlockHash;
        credits[worker] += job.reward;
        emit JobFinalized(jobId, worker, outputHash, aiSealProofHash, executionBlockHash);
    }

    function cancelExpiredJob(uint256 jobId) external {
        Job storage job = jobs[jobId];
        if (job.status != JobStatus.Open || block.timestamp <= job.deadline || msg.sender != job.requester) revert InvalidJob();
        job.status = JobStatus.Cancelled;
        credits[job.requester] += job.reward;
        emit JobCancelled(jobId);
    }

    function withdraw() external nonReentrant {
        uint256 amount = credits[msg.sender];
        credits[msg.sender] = 0;
        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        if (!ok) revert TransferFailed();
    }
}
