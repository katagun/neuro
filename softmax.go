package neuro

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type softmaxFunc struct{}

func init() {
	activationMap["softmax"] = &softmaxFunc{}
}

func (softmaxFunc) activate(m, in *mat64.Dense, deriv bool, transpose bool) error {
	var g float64
	r, c := m.Dims()
	rowSum := make([]float64, r)
	m.Apply(
		func(r, c int, v float64) float64 {
			v = preventOverflow(v)
			g = math.Exp(v)
			rowSum[r] += g
			return g
		}, in)
	for k := 0; k < c; k++ {
		col := mat64.Col(nil, k, m)
		floatsDivision(col, rowSum)
		m.SetCol(k, col)
	}
	return nil
}

func (f softmaxFunc) backpropError(n *Network, layer int) error {
	return nil
}

// Returns the average error on the output layer
func (f softmaxFunc) outputError(output *mat64.Dense, target [][]float64) error {
	return nil
}
