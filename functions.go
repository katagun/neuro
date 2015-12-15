package neuro

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"

	"github.com/gonum/matrix/mat64"
)

func randomFuncMat(m *mat64.Dense, min, max float64) {
	m.Apply(
		func(r, c int, v float64) float64 {
			return rand.Float64()*(max-min) + min
		}, m)
}

func randomFuncVec(v *mat64.Vector, min, max float64) {
	//rows, _ := v.Dims()
	fmt.Println(v)
}

func randomFunc(rows, cols int, min, max float64) []float64 {
	output := make([]float64, rows*cols)
	for k := range output {
		output[k] = rand.Float64()*(max-min) + min
	}
	return output
}

// Fucntion used for the activation function
func activateFloat(a []float64, f func(float64) float64) []float64 {
	for k := range a {
		a[k] = f(a[k])
	}
	return a
}

// Error tracing function for nested functions
func trace() string {
	pc := make([]uintptr, 10) // at least 1 entry needed
	runtime.Callers(2, pc)
	f := runtime.FuncForPC(pc[0])
	file, line := f.FileLine(pc[0])
	return fmt.Sprintf(" on line %s:%d", file, line)
}

func calcActivate(in, out *mat64.Dense, af, df func(float64) float64, deriv, transpose bool) error {
	rowsIn, colsIn := in.Dims()
	rowsOut, colsOut := out.Dims()
	if transpose {
		if rowsIn != colsOut || colsIn != rowsOut {
			return errors.New(fmt.Sprint(ERROR_DIMENSIONS_MISMATCH, trace()))
		}
		if deriv {
			for i := 0; i < rowsIn; i++ {
				out.SetCol(i, activateFloat(mat64.Row(nil, i, in), df))
			}
			return nil
		}
		for i := 0; i < rowsIn; i++ {
			out.SetCol(i, activateFloat(mat64.Row(nil, i, in), af))
		}
		return nil
	}
	if rowsIn != rowsOut || colsIn != colsOut {
		return errors.New(fmt.Sprint(ERROR_DIMENSIONS_MISMATCH, trace()))
	}
	if deriv {
		for i := 0; i < rowsIn; i++ {
			out.SetRow(i, activateFloat(mat64.Row(nil, i, in), df))
		}
		return nil
	}
	for i := 0; i < rowsIn; i++ {
		out.SetRow(i, activateFloat(mat64.Row(nil, i, in), af))
	}
	return nil
}

// Backpropagation for logistic activation functions like sigmoid or tanh
func (n *Network) logisticBackprop(f func(*mat64.Dense, *mat64.Dense, bool, bool) error, layer int) error {
	if layer != n.OutputLayer {
		n.Layers[layer].Errors.Mul(n.Layers[layer+1].Weights, n.Layers[layer+1].Errors)
	}
	err := f(n.Layers[layer].Nodes, n.Layers[layer].Derivative, true, true)
	if err != nil {
		return err
	}
	n.Layers[layer].Errors.MulElem(n.Layers[layer].Errors, n.Layers[layer].Derivative)
	return nil
}

// Custom Divider function with check for underflow NAN
func floatsDivision(a, b []float64) {
	for k := range a {
		a[k] = a[k] / b[k]
		if math.IsNaN(a[k]) {
			a[k] = 4.940656458412465441765687928682213723651e-324
		}
	}
}

// Function to check for overflow and underflow when math.Exp
func preventOverflow(v float64) float64 {
	if v > 7.09782712893383973096e+02 {
		return 7.09782712893383973096e+02
	}
	if v < -7.45133219101941108420e+02 {
		return -7.45133219101941108420e+02
	}
	return v
}

// Mean squared error
func meanSquaredError(output *mat64.Dense, target [][]float64) (float64, error) {
	r, _ := output.Dims()
	if len(target) != r {
		return 0, errors.New(ERROR_DIMENSIONS_MISMATCH)
	}
	netError := 0.0
	for i := 0; i < r; i++ {
		for k, v := range output.RawRowView(i) {
			netError += math.Pow(v-target[i][k], 2)
		}
	}
	netError = netError / float64(r)
	return netError, nil
}
