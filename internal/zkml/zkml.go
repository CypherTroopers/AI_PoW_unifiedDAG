package zkml

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"sync"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	nativeMiMC "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	circuitMiMC "github.com/consensys/gnark/std/hash/mimc"
)

const (
	Dimensions    = 8
	Scheme        = "groth16-bn254-quantized-linear-v1"
	inputDomain   = uint64(0x4149494e50555431)
	modelDomain   = uint64(0x41494d4f44454c31)
	maxProofBytes = 4 << 10
)

type Input [Dimensions]uint64

type Model struct {
	Weights [Dimensions]uint64 `json:"weights"`
	Bias    uint64             `json:"bias"`
}

type Proof struct {
	Scheme          string `json:"scheme"`
	CircuitID       string `json:"circuitId"`
	ModelCommitment string `json:"modelCommitment"`
	InputCommitment string `json:"inputCommitment"`
	Output          string `json:"output"`
	Proof           []byte `json:"proof"`
}

type circuit struct {
	Inputs  [Dimensions]frontend.Variable
	Weights [Dimensions]frontend.Variable
	Bias    frontend.Variable

	ModelCommitment frontend.Variable `gnark:",public"`
	InputCommitment frontend.Variable `gnark:",public"`
	Output          frontend.Variable `gnark:",public"`
}

func (c *circuit) Define(api frontend.API) error {
	inputHash, err := circuitMiMC.NewMiMC(api)
	if err != nil {
		return err
	}
	inputHash.Write(inputDomain, Dimensions)
	for i := 0; i < Dimensions; i++ {
		api.ToBinary(c.Inputs[i], 16)
		inputHash.Write(c.Inputs[i])
	}
	api.AssertIsEqual(c.InputCommitment, inputHash.Sum())

	modelHash, err := circuitMiMC.NewMiMC(api)
	if err != nil {
		return err
	}
	modelHash.Write(modelDomain, Dimensions+1)
	output := frontend.Variable(c.Bias)
	api.ToBinary(c.Bias, 32)
	for i := 0; i < Dimensions; i++ {
		api.ToBinary(c.Weights[i], 16)
		modelHash.Write(c.Weights[i])
		output = api.Add(output, api.Mul(c.Inputs[i], c.Weights[i]))
	}
	modelHash.Write(c.Bias)
	api.AssertIsEqual(c.ModelCommitment, modelHash.Sum())
	api.ToBinary(c.Output, 64)
	api.AssertIsEqual(c.Output, output)
	return nil
}

type Prover struct {
	mu        sync.Mutex
	ccs       constraint.ConstraintSystem
	pk        groth16.ProvingKey
	vk        groth16.VerifyingKey
	circuitID string
}

type Verifier struct {
	vk        groth16.VerifyingKey
	circuitID string
}

func Setup() (*Prover, *Verifier, error) {
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit{})
	if err != nil {
		return nil, nil, err
	}
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		return nil, nil, err
	}
	id, err := verifyingKeyID(vk)
	if err != nil {
		return nil, nil, err
	}
	return &Prover{ccs: ccs, pk: pk, vk: vk, circuitID: id}, &Verifier{vk: vk, circuitID: id}, nil
}

func (p *Prover) CircuitID() string   { return p.circuitID }
func (v *Verifier) CircuitID() string { return v.circuitID }

func (p *Prover) Prove(input Input, model Model) (Proof, error) {
	if err := validateValues(input, model); err != nil {
		return Proof{}, err
	}
	modelCommitment := commitment(modelDomain, append(model.Weights[:], model.Bias))
	inputCommitment := commitment(inputDomain, input[:])
	output := Evaluate(input, model)
	assignment := assignmentFor(input, model, modelCommitment, inputCommitment, output)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		return Proof{}, err
	}
	p.mu.Lock()
	proof, err := groth16.Prove(p.ccs, p.pk, witness)
	p.mu.Unlock()
	if err != nil {
		return Proof{}, err
	}
	var encoded bytes.Buffer
	if _, err := proof.WriteTo(&encoded); err != nil {
		return Proof{}, err
	}
	return Proof{
		Scheme: Scheme, CircuitID: p.circuitID, ModelCommitment: modelCommitment.String(),
		InputCommitment: inputCommitment.String(), Output: new(big.Int).SetUint64(output).String(), Proof: encoded.Bytes(),
	}, nil
}

func (v *Verifier) Verify(proofData Proof) error {
	if proofData.Scheme != Scheme || proofData.CircuitID != v.circuitID {
		return errors.New("unsupported zkML scheme or circuit id")
	}
	if len(proofData.Proof) == 0 || len(proofData.Proof) > maxProofBytes {
		return errors.New("invalid zkML proof size")
	}
	modelCommitment, ok := new(big.Int).SetString(proofData.ModelCommitment, 10)
	if !ok {
		return errors.New("invalid model commitment")
	}
	inputCommitment, ok := new(big.Int).SetString(proofData.InputCommitment, 10)
	if !ok {
		return errors.New("invalid input commitment")
	}
	output, ok := new(big.Int).SetString(proofData.Output, 10)
	if !ok {
		return errors.New("invalid zkML output")
	}
	assignment := circuit{ModelCommitment: modelCommitment, InputCommitment: inputCommitment, Output: output}
	publicWitness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		return err
	}
	proof := groth16.NewProof(ecc.BN254)
	read, err := proof.ReadFrom(bytes.NewReader(proofData.Proof))
	if err != nil {
		return err
	}
	if read != int64(len(proofData.Proof)) {
		return errors.New("trailing data in zkML proof")
	}
	return groth16.Verify(proof, v.vk, publicWitness)
}

func Evaluate(input Input, model Model) uint64 {
	output := model.Bias
	for i := 0; i < Dimensions; i++ {
		output += input[i] * model.Weights[i]
	}
	return output
}

func Derive(modelName, prompt string) (Input, Model) {
	inputSeed := sha256.Sum256([]byte("AIDCHAIN_ZKML_INPUT_V1|" + prompt))
	modelSeed := sha256.Sum256([]byte("AIDCHAIN_ZKML_MODEL_V1|" + modelName))
	var input Input
	var model Model
	for i := 0; i < Dimensions; i++ {
		input[i] = uint64(binary.LittleEndian.Uint16(inputSeed[i*2 : i*2+2]))
		model.Weights[i] = uint64(binary.LittleEndian.Uint16(modelSeed[i*2 : i*2+2]))
	}
	model.Bias = uint64(binary.LittleEndian.Uint32(modelSeed[16:20]))
	return input, model
}

func StatementFor(modelName, prompt string) (modelCommitment, inputCommitment, output string) {
	input, model := Derive(modelName, prompt)
	return commitment(modelDomain, append(model.Weights[:], model.Bias)).String(),
		commitment(inputDomain, input[:]).String(), new(big.Int).SetUint64(Evaluate(input, model)).String()
}

func ProofHash(proof Proof) string {
	encoded, _ := json.Marshal(proof)
	hash := sha256.Sum256(encoded)
	return "0x" + hex.EncodeToString(hash[:])
}

func commitment(domain uint64, values []uint64) *big.Int {
	h := nativeMiMC.NewFieldHasher()
	writeElement := func(value uint64) {
		var element fr.Element
		element.SetUint64(value)
		h.WriteElement(element)
	}
	writeElement(domain)
	writeElement(uint64(len(values)))
	for _, value := range values {
		writeElement(value)
	}
	result := h.SumElement()
	return result.BigInt(new(big.Int))
}

func assignmentFor(input Input, model Model, modelCommitment, inputCommitment *big.Int, output uint64) circuit {
	assignment := circuit{
		Bias: model.Bias, ModelCommitment: modelCommitment, InputCommitment: inputCommitment,
		Output: output,
	}
	for i := 0; i < Dimensions; i++ {
		assignment.Inputs[i] = input[i]
		assignment.Weights[i] = model.Weights[i]
	}
	return assignment
}

func validateValues(input Input, model Model) error {
	if model.Bias >= 1<<32 {
		return errors.New("zkML bias exceeds 32 bits")
	}
	for i := 0; i < Dimensions; i++ {
		if input[i] >= 1<<16 || model.Weights[i] >= 1<<16 {
			return errors.New("zkML input or weight exceeds 16 bits")
		}
	}
	return nil
}

func verifyingKeyID(vk groth16.VerifyingKey) (string, error) {
	var encoded bytes.Buffer
	if _, err := vk.WriteTo(&encoded); err != nil {
		return "", err
	}
	hash := sha256.Sum256(append([]byte(Scheme+"|"), encoded.Bytes()...))
	return "0x" + hex.EncodeToString(hash[:]), nil
}

type artifactManifest struct {
	Scheme    string `json:"scheme"`
	CircuitID string `json:"circuitId"`
	Curve     string `json:"curve"`
}

func (p *Prover) Save(directory string) error {
	if err := os.MkdirAll(directory, 0o750); err != nil {
		return err
	}
	if err := writeObject(filepath.Join(directory, "circuit.r1cs"), p.ccs); err != nil {
		return err
	}
	if err := writeObject(filepath.Join(directory, "proving.key"), p.pk); err != nil {
		return err
	}
	if err := writeObject(filepath.Join(directory, "verifying.key"), p.vk); err != nil {
		return err
	}
	manifest, _ := json.MarshalIndent(artifactManifest{Scheme: Scheme, CircuitID: p.circuitID, Curve: "BN254"}, "", "  ")
	return os.WriteFile(filepath.Join(directory, "manifest.json"), manifest, 0o640)
}

func LoadProver(directory string) (*Prover, *Verifier, error) {
	ccs := groth16.NewCS(ecc.BN254)
	if err := readObject(filepath.Join(directory, "circuit.r1cs"), ccs); err != nil {
		return nil, nil, err
	}
	pk := groth16.NewProvingKey(ecc.BN254)
	if err := readObject(filepath.Join(directory, "proving.key"), pk); err != nil {
		return nil, nil, err
	}
	vk, id, err := loadVerifyingKey(directory)
	if err != nil {
		return nil, nil, err
	}
	return &Prover{ccs: ccs, pk: pk, vk: vk, circuitID: id}, &Verifier{vk: vk, circuitID: id}, nil
}

func LoadVerifier(directory string) (*Verifier, error) {
	vk, id, err := loadVerifyingKey(directory)
	if err != nil {
		return nil, err
	}
	return &Verifier{vk: vk, circuitID: id}, nil
}

func loadVerifyingKey(directory string) (groth16.VerifyingKey, string, error) {
	vk := groth16.NewVerifyingKey(ecc.BN254)
	if err := readObject(filepath.Join(directory, "verifying.key"), vk); err != nil {
		return nil, "", err
	}
	id, err := verifyingKeyID(vk)
	return vk, id, err
}

func writeObject(path string, object interface {
	WriteTo(io.Writer) (int64, error)
}) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o640)
	if err != nil {
		return err
	}
	if _, err := object.WriteTo(file); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func readObject(path string, object interface {
	ReadFrom(io.Reader) (int64, error)
}) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = object.ReadFrom(file)
	return err
}
