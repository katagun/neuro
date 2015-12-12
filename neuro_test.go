package neuro

import (
	"log"
	"testing"
)

func ImportExportTest(t *testing.T) {
	var (
		firstOutput, secondOutput [][]float64
	)

	n, err := New(NetData{
		Nodes:       []int{3, 10, 5, 2},
		Activations: []string{"tanh", "sigmoid", "softmax"},
		BatchSize:   3,
		Train:       true,
	})
	if err != nil {
		t.Error(err)
	}
	in := [][]float64{[]float64{1, 1, 0}, []float64{0, 1, 1}, []float64{1, 0, 1}}

	// Setting up the network's learn rate
	n.LearnRate = 0.1
	n.Momentum = 0.5

	if err := n.Forward(in); err != nil {
		t.Error(err)
	}
	firstOutput = n.GetOutput()
	netData, err := n.Export("")
	if err != nil {
		t.Error(err)
	}
	netData.Train = false
	netData.BatchSize = 3
	y, err := New(netData)
	if err != nil {
		t.Error(err)
	}
	if err := y.Forward(in); err != nil {
		t.Error(err)
	}
	secondOutput = y.GetOutput()
	if len(firstOutput) != len(secondOutput) {
		t.Errorf("First and Second output don't match \n First: \n %v \n Second:\n %v", firstOutput, secondOutput)
	}
	for k := range firstOutput {
		if len(firstOutput[k]) != len(secondOutput[k]) {
			t.Errorf("First and Second output don't match \n First: \n %v \n Second:\n %v", firstOutput, secondOutput)
		}
		for k2 := range firstOutput[k] {
			if firstOutput[k][k2] != secondOutput[k][k2] {
				t.Errorf("First and Second output don't match \n First: \n %v \n Second:\n %v", firstOutput, secondOutput)
			}

		}
	}
}

func AccuracyTest(t *testing.T) {
	n, err := New(NetData{
		Nodes:       []int{3, 10, 5, 2},
		Activations: []string{"tanh", "sigmoid", "softmax"},
		BatchSize:   3,
		Train:       true,
	})
	if err != nil {
		log.Fatal(err)
	}
	in := [][]float64{[]float64{1, 1, 0}, []float64{0, 1, 1}, []float64{1, 0, 1}}
	target := [][]float64{[]float64{1, 0}, []float64{0, 1}, []float64{0, 1}}

	// Setting up the network's learn rate
	n.LearnRate = 0.1
	n.Momentum = 0.5

	for i := 0; i < 10000; i++ {
		if err := n.Forward(in); err != nil {
			panic(err)
		}

		if err := n.Backward(target); err != nil {
			panic(err)
		}

	}
	if err := n.Forward(in); err != nil {
		panic(err)
	}
	output := n.GetOutput()
	for k := range output {
		for k2 := range output[k] {
			diff := target[k][k2] - output[k][k2]
			if diff < 0 {
				diff = diff * -1
			}
			if diff > 0.1 {
				t.Error("Network is not accurate")
			}
		}
	}
}
