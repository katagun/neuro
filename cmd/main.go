package main

import (
	"fmt"
	"log"
	"time"

	"github.com/ingeniosity/neuro"
)

const (
	exportPath = "/data/hdd/languageData/en/naskoTEST.json"
)

func main() {
	n, err := neuro.New(neuro.NetData{
		Nodes:       []int{3, 10, 2},
		Activations: []string{"tanh", "softmax"},
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

	for i := 0; i < 1000; i++ {
		if err := n.Forward(in); err != nil {
			panic(err)
		}

		if err := n.Backward(target); err != nil {
			panic(err)
		}

		netError, err := n.NetError(target)
		if err != nil {
			panic(err)
		}
		fmt.Println("NET ERROR:", netError)

	}
	if err := n.Forward(in); err != nil {
		panic(err)
	}
	fmt.Println(n.GetOutput())
	/////////////////////////////////////
	elapsed := time.Since(start)
	log.Printf("Neural took %s", elapsed)
	/////////////////////////////////////
	_, err = n.Export(exportPath)
	if err != nil {
		log.Fatal(err)
	}
	y, err := neuro.Import(exportPath, 3, false)
	if err != nil {
		log.Fatal(err)
	}

	if err := y.Forward(in); err != nil {
		panic(err)
	}

	fmt.Println("\n\n", y.GetOutput())

}
