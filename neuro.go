package neuro

import (
	"encoding/gob"
	"errors"
	"math/rand"
	"os"
	"time"

	"github.com/gonum/matrix/mat64"
)

type (
	Network struct {
		Layers      []Layer
		Input       *mat64.Dense
		InputCount  int
		OutputLayer int
		BatchSize   int
		isTrain     bool
		Target      [][]float64
		LearnRate   float64
		Momentum    float64
	}
	Layer struct {
		Nodes            *mat64.Dense
		Weights          *mat64.Dense
		DeltaWeights     *mat64.Dense
		DeltaWeightsPrev *mat64.Dense
		PrevDeltaWeights *mat64.Dense
		Errors           *mat64.Dense
		Derivative       *mat64.Dense
		Activation       activationFunction
		NodesCount       int
		BiasWeights      *mat64.Vector
		BiasWeightsPrev  *mat64.Vector
	}
	activationFunction interface {
		activate(*mat64.Dense, *mat64.Dense, bool, bool) error
		backpropError(*Network, int) error
		outputError(*mat64.Dense, [][]float64) error
	}
)

var activationMap = map[string]activationFunction{}

const (
	ERROR_ACTIVATION_COUNT    = "[ERROR] The number of layers do not match the activation functions"
	ERROR_LAYERS_COUNT        = "[ERROR] The number of layers should be greater than 1"
	ERROR_UNKNOWN_ACTIVATION  = "[ERROR] Unknown activation function"
	ERROR_MOMENTUM            = "[ERROR] If learnRate is 0.0 then momentum has to be 0.0"
	ERROR_INT_POSITIVE        = "[ERROR] All values have to be positive values"
	ERROR_WRONG_INPUTS_COUNT  = "[ERROR] The input values don't match the network's input count"
	ERROR_WRONG_BATCH_COUNT   = "[ERROR] The input values don't match the batchSize"
	ERROR_BATCHSIZE           = "[ERROR] BatchSize should be bigger than 0"
	ERROR_DIMENSIONS_MISMATCH = "[ERROR] Dimensions mismatch"
	ERROR_LEARN_RATE          = "[ERROR] Set the network's learn rate. n.LearnRate > 0"
	ERROR_NOT_TRAINABLE       = "[ERROR] Network does not support training"
	ERROR_LAYERS_IMPORT       = "[ERROR] Network Layers mismatch"
)

func init() {
	//runtime.GOMAXPROCS(1)
}

// New returns an initialized the Neural Network
func New(nodes []int, activation []string, batchSize int, train bool) (*Network, error) {
	n := new(Network)
	if len(nodes) < 2 {
		return nil, errors.New(ERROR_LAYERS_COUNT)
	}
	if len(nodes)-1 != len(activation) {
		return nil, errors.New(ERROR_ACTIVATION_COUNT)
	}
	if batchSize <= 0 {
		return nil, errors.New(ERROR_BATCHSIZE)
	}
	for _, node := range nodes {
		if node < 1 {
			return nil, errors.New(ERROR_INT_POSITIVE)
		}
	}
	// The other layers nodes go here
	layerNodes := nodes[1:len(nodes)]
	// Batchsize of the network
	n.BatchSize = batchSize
	n.Layers = make([]Layer, len(nodes)-1)
	n.OutputLayer = len(nodes) - 2

	// Create the input matrice
	n.Input = mat64.NewDense(n.BatchSize, nodes[0], nil)
	n.InputCount = nodes[0]

	// Initialize the Seed for Rand
	rand.Seed(time.Now().UTC().UnixNano())
	// Go through all the nodes and layers
	for k := range layerNodes {
		n.Layers[k].NodesCount = layerNodes[k]
	}
	for k := range n.Layers {
		n.Layers[k].Nodes = mat64.NewDense(batchSize, layerNodes[k], nil)
		// Check if the activation function exists
		act, ok := activationMap[activation[k]]
		if !ok {
			return nil, errors.New(ERROR_UNKNOWN_ACTIVATION)
		}
		// Attach the activation function to the layer
		n.Layers[k].Activation = act
		// Create the BiasWeights vector and seed it with random values
		n.Layers[k].BiasWeights = mat64.NewVector(layerNodes[k], randomFunc(1, layerNodes[k], -1, 1))
		switch k {
		// Check if the input matrice is the previous layer
		case 0:
			// Create the Weights matrices and seed them with random values
			n.Layers[k].Weights = mat64.NewDense(n.InputCount, n.Layers[k].NodesCount, randomFunc(n.InputCount, n.Layers[k].NodesCount, -1, 1))
		default:
			// Create the Weights matrices and seed them with random values
			n.Layers[k].Weights = mat64.NewDense(n.Layers[k-1].NodesCount, n.Layers[k].NodesCount, randomFunc(n.Layers[k-1].NodesCount, n.Layers[k].NodesCount, -1, 1))
		}
	}
	if !train {
		return n, nil
	}
	// If we will use the network for training
	n.isTrain = true
	for k := range n.Layers {
		switch k {
		case 0:
			n.Layers[k].DeltaWeights = mat64.NewDense(n.Layers[k].NodesCount, n.InputCount, nil)
			n.Layers[k].DeltaWeightsPrev = mat64.NewDense(n.Layers[k].NodesCount, n.InputCount, nil)
		default:
			n.Layers[k].DeltaWeights = mat64.NewDense(n.Layers[k].NodesCount, n.Layers[k-1].NodesCount, nil)
			n.Layers[k].DeltaWeightsPrev = mat64.NewDense(n.Layers[k].NodesCount, n.Layers[k-1].NodesCount, nil)
		}
		n.Layers[k].Errors = mat64.NewDense(n.Layers[k].NodesCount, n.BatchSize, nil)
		n.Layers[k].Derivative = mat64.NewDense(n.Layers[k].NodesCount, n.BatchSize, nil)
	}
	return n, nil
}

// Forward takes inputs and passes through the network
func (n *Network) Forward(in [][]float64) error {
	if len(in) > n.BatchSize {
		return errors.New(ERROR_WRONG_BATCH_COUNT)
	}
	var prev int
	for k := range in {
		n.Input.SetRow(k, in[k])
	}
	for i := 0; i <= n.OutputLayer; i++ {
		prev = i - 1
		switch i {
		case 0:
			n.Layers[i].Nodes.Mul(n.Input, n.Layers[i].Weights)
		default:
			n.Layers[i].Nodes.Mul(n.Layers[prev].Nodes, n.Layers[i].Weights)
		}
		for a := 0; a < n.BatchSize; a++ {
			n.Layers[i].Nodes.RowView(a).AddVec(n.Layers[i].Nodes.RowView(a), n.Layers[i].BiasWeights)
		}
		n.Layers[i].Activation.activate(n.Layers[i].Nodes, n.Layers[i].Nodes, false, false)
	}
	return nil
}

// Backward takes target values and back propagates through the network
func (n *Network) Backward(target [][]float64) error {
	if n.isTrain == false {
		return errors.New(ERROR_NOT_TRAINABLE)
	}
	if len(target) > n.BatchSize {
		return errors.New(ERROR_WRONG_BATCH_COUNT)
	}
	if n.LearnRate <= 0.0 {
		return errors.New(ERROR_LEARN_RATE)
	}
	for k := range target {
		n.Layers[n.OutputLayer].Errors.SetCol(k, target[k])
	}
	n.Layers[n.OutputLayer].Errors.Sub(n.Layers[n.OutputLayer].Errors, n.Layers[n.OutputLayer].Nodes.T())
	for i := n.OutputLayer; i >= 0; i-- {
		err := n.Layers[i].Activation.backpropError(n, i)
		if err != nil {
			return err
		}
		switch i {
		case 0:
			n.Layers[i].DeltaWeights.Mul(n.Layers[i].Errors, n.Input)
		default:
			n.Layers[i].DeltaWeights.Mul(n.Layers[i].Errors, n.Layers[i-1].Nodes)
		}

		// Scale the weigths update by the learning rate
		n.Layers[i].DeltaWeights.Scale(n.LearnRate, n.Layers[i].DeltaWeights)
		// Update the bias weights
		for ib := 0; ib < n.BatchSize; ib++ {
			n.Layers[i].BiasWeights.AddVec(n.Layers[i].Errors.ColView(ib), n.Layers[i].BiasWeights)
		}
		if n.Momentum > 0 {
			n.Layers[i].DeltaWeightsPrev.Clone(n.Layers[i].DeltaWeights)
			n.Layers[i].DeltaWeightsPrev.Scale(n.Momentum, n.Layers[i].DeltaWeightsPrev)
		}
		n.Layers[i].DeltaWeights.Add(n.Layers[i].DeltaWeights, n.Layers[i].DeltaWeightsPrev)
	}
	// Update all the weights
	for i := n.OutputLayer; i >= 0; i-- {
		n.Layers[i].Weights.Add(n.Layers[i].Weights, n.Layers[i].DeltaWeights.T())
	}
	return nil
}

// GetError return the error in the network in relation to the Cost function
func (n *Network) GetError(target [][]float64) (float64, error) {
	err := n.Layers[n.OutputLayer].Activation.outputError(n.Layers[n.OutputLayer].Nodes, target)
	if err != nil {
		return 0, err
	}
	return 0, nil
}

// GetOutput returns the values from the last layer of the network
func (n *Network) GetOutput() [][]float64 {
	var output [][]float64
	output = make([][]float64, n.BatchSize)
	for i := 0; i < n.BatchSize; i++ {
		output[i] = mat64.Row(nil, i, n.Layers[n.OutputLayer].Nodes)
	}
	return output
}

// Export saves the layers weights in a specified file location
func (n *Network) Export(path string) error {
	// create a file
	dataFile, err := os.Create(path)
	if err != nil {
		return err
	}
	defer dataFile.Close()
	//weights := make([][][]float64, len(n.Layers))
	weights := map[string][][][]float64{}
	for k, layer := range n.Layers {
		r, _ := layer.Weights.Dims()
		weights["weights"][k] = make([][]float64, r)
		for i := 0; i < r; i++ {
			weights["weights"][k][i] = layer.Weights.RawRowView(i)
		}
	}

	//////////////
	/////////////////////
	/////////////////////////
	///////////////////////////////
	//FINISH EXPORT IMPORT CUSTOM VECS AND MATRICES
	///////////////////////////////
	/////////////////////////
	////////////////////
	//////////////

	dataEncoder := gob.NewEncoder(dataFile)
	err = dataEncoder.Encode(weights)
	if err != nil {
		return err
	}
	return nil
}

// Import takes loads layer weights in to the network
func (n *Network) Import(path string) error {
	weights := [][][]float64{}
	// open data file
	dataFile, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dataFile.Close()
	dataDecoder := gob.NewDecoder(dataFile)
	err = dataDecoder.Decode(&weights)
	if err != nil {
		return err
	}
	if len(weights) != len(n.Layers) {
		return errors.New(ERROR_LAYERS_IMPORT)
	}
	for k, layer := range n.Layers {
		r, c := layer.Weights.Dims()
		if len(weights[k]) != r {
			return errors.New(ERROR_DIMENSIONS_MISMATCH)
		}
		for k2, row := range weights[k] {
			if len(row) != c {
				return errors.New(ERROR_DIMENSIONS_MISMATCH)
			}
			layer.Weights.SetRow(k2, row)
		}
	}
	return nil
}