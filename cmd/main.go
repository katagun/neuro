package main

import (
	"fmt"
	"log"
	"time"

	"github.com/unigraph/neuro"
)

func main() {
	n, err := neuro.New([]int{3, 10, 5, 2}, []string{"tanh", "sigmoid", "softmax"}, 3, true)
	if err != nil {
		log.Fatal(err)
	}
	in := [][]float64{[]float64{1, 1, 0}, []float64{0, 1, 1}, []float64{1, 0, 1}}
	target := [][]float64{[]float64{1, 0}, []float64{0, 1}, []float64{0, 1}}

	// Setting up the network's learn rate
	n.LearnRate = 0.01
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

}
