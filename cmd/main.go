package main

import (
	"fmt"
	"log"
	"time"

	"github.com/gaspiman/ingenio/neuro"
)

func main() {
	n, err := neuro.New([]int{3, 3, 2}, []string{"tanh", "softmax"}, 3, true)
	if err != nil {
		log.Fatal(err)
	}
	in := [][]float64{[]float64{1, 1, 0}, []float64{0, 1, 1}, []float64{0, 0, 0}}
	target := [][]float64{[]float64{1, 0}, []float64{0, 1}, []float64{1, 0}}

	// Setting up the network's learn rate
	n.LearnRate = 0.005
	n.Momentum = 0.5

	/////////////////////////////////////
	start := time.Now()
	/////////////////////////////////////

	for i := 0; i < 20000; i++ {
		if err := n.Forward(in); err != nil {
			panic(err)
		}
		fmt.Println(n.GetOutput())

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
