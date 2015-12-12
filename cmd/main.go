package main

import (
	"fmt"
	"log"
	"time"

	"github.com/unigraph/neuro"
)

func main() {
	n, err := neuro.New(neuro.NetData{
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

	/////////////////////////////////////
	start := time.Now()
	/////////////////////////////////////

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
	fmt.Println(n.GetOutput())
	/////////////////////////////////////
	elapsed := time.Since(start)
	log.Printf("Neural took %s", elapsed)
	/////////////////////////////////////
	err = n.Export("/data/hdd/languageData/en/naskoTEST.json")
	if err != nil {
		log.Fatal(err)
	}
	y, err := neuro.ImportNetwork("/data/hdd/languageData/en/naskoTEST.json", 3, false)
	if err != nil {
		log.Fatal(err)
	}

	if err := y.Forward(in); err != nil {
		panic(err)
	}

	fmt.Println("\n\n", y.GetOutput())

}
