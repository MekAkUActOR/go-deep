package deep

import (
	"encoding/json"
)

// Dump is a neural network dump
type Dump struct {
	Config  *Config
	Weights [][][]float64
}

// ApplyWeights sets the weights from a three-dimensional slice
func (n *Neural) ApplyWeights(weights [][][]float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				n.Layers[i].Neurons[j].In[k].Weight = weights[i][j][k]
			}
		}
	}
}

// Weights returns all weights in sequence
func (n Neural) Weights() [][][]float64 {
	weights := make([][][]float64, len(n.Layers))
	for i, l := range n.Layers {
		weights[i] = make([][]float64, len(l.Neurons))
		for j, n := range l.Neurons {
			weights[i][j] = make([]float64, len(n.In))
			for k, in := range n.In {
				weights[i][j][k] = in.Weight
			}
		}
	}
	return weights
}

// Dump generates a network dump
func (n Neural) Dump() *Dump {
	return &Dump{
		Config:  n.Config,
		Weights: n.Weights(),
	}
}

// FromDump restores a Neural from a dump
func FromDump(dump *Dump) *Neural {
	n := NewNeural(dump.Config)
	n.ApplyWeights(dump.Weights)

	return n
}

// Marshal marshals to JSON from network
func (n Neural) Marshal() ([]byte, error) {
	return json.Marshal(n.Dump())
}

// Unmarshal restores network from a JSON blob
func Unmarshal(bytes []byte) (*Neural, error) {
	var dump Dump
	if err := json.Unmarshal(bytes, &dump); err != nil {
		return nil, err
	}
	return FromDump(&dump), nil
}

// ZeroWeights sets all weights in the Dump struct to zero
func ZeroWeights(dump *Dump) {
	for l, layer := range dump.Weights {
		for n, neuron := range layer {
			for w := range neuron {
				dump.Weights[l][n][w] = 0.0
			}
		}
	}
}

// FractionWeights sets all weights in the Dump struct to zero
func FractionWeights(dump *Dump, fraction float64) {
	for l, layer := range dump.Weights {
		for n, neuron := range layer {
			for w := range neuron {
				dump.Weights[l][n][w] /= fraction
			}
		}
	}
}

// AddWeights adds the weights of two Dump structs and stores the result in a third Dump struct
func AddWeights(dump1, dump2 *Dump) *Dump {
	if len(dump1.Weights) != len(dump2.Weights) {
		panic("The structures of the two Dump objects do not match")
	}

	result := Dump{
		Config:  dump1.Config, // Assuming same configuration for both
		Weights: make([][][]float64, len(dump1.Weights)),
	}

	for l := range dump1.Weights {
		if len(dump1.Weights[l]) != len(dump2.Weights[l]) {
			panic("The structures of the two Dump objects do not match")
		}
		result.Weights[l] = make([][]float64, len(dump1.Weights[l]))
		for n := range dump1.Weights[l] {
			if len(dump1.Weights[l][n]) != len(dump2.Weights[l][n]) {
				panic("The structures of the two Dump objects do not match")
			}
			result.Weights[l][n] = make([]float64, len(dump1.Weights[l][n]))
			for w := range dump1.Weights[l][n] {
				result.Weights[l][n][w] = dump1.Weights[l][n][w] + dump2.Weights[l][n][w]
			}
		}
	}

	return &result
}
