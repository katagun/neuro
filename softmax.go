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
	var s, e int
	step := len(a) / splitSoftmax
	for i := 0; i < step; i++ {
		s = i * splitSoftmax
		e = s + splitSoftmax
		sum = 0
		for k := range a[s:e] {
			a[s+k] = f(a[s+k])
			sum += a[s+k]
		}
		for k := range a[s:e] {
			a[s+k] = a[s+k] / sum
		}
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
func (f softmaxFunc) layerError(output *mat64.Dense, target [][]float64) (float64, error) {
	r, _ := output.Dims()
	if len(target) != r {
		return 0, errors.New(ERROR_DIMENSIONS_MISMATCH)
	}
	netError := 0.0
	for i := 0; i < r; i++ {
		for k, v := range output.RawRowView(i) {
			if target[i][k] == 1.0 {
				netError -= math.Log(v)
			}
		}
	}
	netError = netError / float64(r)
	return netError, nil
}
