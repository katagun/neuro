package neuro

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type sigmoidFunc struct{}

func init() {
	activationMap["sigmoid"] = &sigmoidFunc{}
}

func (sigmoidFunc) activate(in, out *mat64.Dense, deriv bool, transpose bool) error {
	return calcActivate(in, out, sigmoidActivate, sigmoidDerivative, deriv, transpose)
}

func sigmoidActivate(v float64) float64   { return 1.0 / (1.0 + math.Exp(-v)) }
func sigmoidDerivative(v float64) float64 { return v * (1 - v) }

func (f sigmoidFunc) backpropError(n *Network, layer int) error {
	return n.logisticBackprop(f.activate, layer)
}

// Returns the average error on the output layer
func (f sigmoidFunc) outputError(output *mat64.Dense, target [][]float64) error {
	return nil
}
