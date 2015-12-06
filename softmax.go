package neuro

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

type softmaxFunc struct{}

func init() {
	activationMap["softmax"] = &softmaxFunc{}
}

/*
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
*/

func (softmaxFunc) activate(in, out *mat64.Dense, deriv bool, transpose bool) error {
	rowsIn, colsIn := in.Dims()
	rowsOut, colsOut := out.Dims()
	if transpose {
		if rowsIn != colsOut || colsIn != rowsOut {
			return errors.New(fmt.Sprint(ERROR_DIMENSIONS_MISMATCH, trace()))
		}
		if deriv {
			for i := 0; i < rowsIn; i++ {
				out.SetCol(i, activateSoftmaxFloat(mat64.Row(nil, i, in), softmaxDerivative))
			}
			return nil
		}
		for i := 0; i < rowsIn; i++ {
			out.SetCol(i, activateSoftmaxFloat(mat64.Row(nil, i, in), softmaxActivate))
		}
		return nil
	}
	if rowsIn != rowsOut || colsIn != colsOut {
		return errors.New(fmt.Sprint(ERROR_DIMENSIONS_MISMATCH, trace()))
	}
	if deriv {
		for i := 0; i < rowsIn; i++ {
			out.SetRow(i, activateSoftmaxFloat(mat64.Row(nil, i, in), softmaxDerivative))
		}
		return nil
	}
	for i := 0; i < rowsIn; i++ {
		out.SetRow(i, activateSoftmaxFloat(mat64.Row(nil, i, in), softmaxActivate))
	}
	return nil
}

func activateSoftmaxFloat(a []float64, f func(float64) float64) []float64 {
	var sum float64
	for k := range a {
		a[k] = f(a[k])
		sum += a[k]
	}
	for k := range a {
		a[k] = a[k] / sum
	}
	return a
}

func softmaxActivate(v float64) float64 {
	v = preventOverflow(v)
	return math.Exp(v)
}

func softmaxDerivative(v float64) float64 { return v * (1 - v) }

func (f softmaxFunc) backpropError(n *Network, layer int) error {
	return n.logisticBackprop(f.activate, layer)
}

// Returns the average error on the output layer
func (f softmaxFunc) layerError(output *mat64.Dense, target [][]float64) error {
	return nil
}
