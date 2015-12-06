package neuro

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type tanhFunc struct{}

func init() {
	activationMap["tanh"] = &tanhFunc{}
}

func (tanhFunc) activate(in, out *mat64.Dense, deriv bool, transpose bool) error {
	return calcActivate(in, out, tanhActivate, tanhDerivative, deriv, transpose)
}

func tanhActivate(v float64) float64 {
	//return 1.7159 * math.Tanh(2.0/3.0*v)
	// Tanh approximation for performace taken from here: http://stackoverflow.com/a/6118100/1809456
	if v < -3 {
		return -1
	}
	if v > 3 {
		return 1
	}
	sq := math.Pow(v, 2)
	return v * (27 + sq) / (27 + 9*sq)
}
func tanhDerivative(v float64) float64 {
	return 1 - math.Pow(v, 2)
}

func (f tanhFunc) backpropError(n *Network, layer int) error {
	return n.logisticBackprop(f.activate, layer)
}

// Returns the average error on the output layer
func (f tanhFunc) layerError(output *mat64.Dense, target [][]float64) error {
	return nil
}
