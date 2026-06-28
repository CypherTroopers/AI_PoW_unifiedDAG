package zkml

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	nativeMiMC "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	circuitMiMC "github.com/consensys/gnark/std/hash/mimc"
)

type oneHashCircuit struct {
	Preimage frontend.Variable
	Hash     frontend.Variable `gnark:",public"`
}

func (c *oneHashCircuit) Define(api frontend.API) error {
	h, _ := circuitMiMC.NewMiMC(api)
	h.Write(c.Preimage)
	api.AssertIsEqual(c.Hash, h.Sum())
	return nil
}

func TestMiMCCompatibility(t *testing.T) {
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &oneHashCircuit{})
	if err != nil {
		t.Fatal(err)
	}
	var element fr.Element
	element.SetUint64(42)
	h := nativeMiMC.NewFieldHasher()
	h.WriteElement(element)
	hash := h.SumElement()
	witness, err := frontend.NewWitness(&oneHashCircuit{Preimage: element, Hash: hash}, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	if err := ccs.IsSolved(witness); err != nil {
		t.Fatal(err)
	}
}

type sequenceHashCircuit struct {
	Values [9]frontend.Variable
	Hash   frontend.Variable `gnark:",public"`
}

func (c *sequenceHashCircuit) Define(api frontend.API) error {
	h, _ := circuitMiMC.NewMiMC(api)
	h.Write(modelDomain, 9)
	for i := range c.Values {
		h.Write(c.Values[i])
	}
	api.AssertIsEqual(c.Hash, h.Sum())
	return nil
}

func TestMiMCSequenceCompatibility(t *testing.T) {
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &sequenceHashCircuit{})
	if err != nil {
		t.Fatal(err)
	}
	_, model := Derive("model.gguf", "prove this inference")
	values := append(model.Weights[:], model.Bias)
	assignment := sequenceHashCircuit{Hash: commitment(modelDomain, values)}
	for i, value := range values {
		assignment.Values[i] = value
	}
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	if err := ccs.IsSolved(witness); err != nil {
		t.Fatal(err)
	}
}

func TestGroth16QuantizedInference(t *testing.T) {
	prover, verifier, err := Setup()
	if err != nil {
		t.Fatal(err)
	}
	input, model := Derive("model.gguf", "prove this inference")
	proof, err := prover.Prove(input, model)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifier.Verify(proof); err != nil {
		t.Fatalf("valid zkML proof rejected: %v", err)
	}

	tampered := proof
	output, _ := new(big.Int).SetString(tampered.Output, 10)
	tampered.Output = output.Add(output, big.NewInt(1)).String()
	if err := verifier.Verify(tampered); err == nil {
		t.Fatal("zkML proof accepted a tampered output")
	}

	trailing := proof
	trailing.Proof = append(append([]byte(nil), proof.Proof...), 0x01)
	if err := verifier.Verify(trailing); err == nil {
		t.Fatal("zkML proof accepted trailing bytes")
	}

	directory := t.TempDir()
	if err := prover.Save(directory); err != nil {
		t.Fatal(err)
	}
	loadedProver, loadedVerifier, err := LoadProver(directory)
	if err != nil {
		t.Fatal(err)
	}
	loadedProof, err := loadedProver.Prove(input, model)
	if err != nil {
		t.Fatal(err)
	}
	if err := loadedVerifier.Verify(loadedProof); err != nil {
		t.Fatalf("loaded zkML artifacts failed: %v", err)
	}
}
