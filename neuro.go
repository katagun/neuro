package neuro

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	"github.com/gonum/matrix/mat64"
)

type (
	Network struct {
		Layers      []Layer
		Input       *mat64.Dense
		Activations []string
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
		layerError(*mat64.Dense, [][]float64) (float64, error)
	}
	NetData struct {
		Nodes        []int
		Activations  []string
		WeightsData  []DataWeights
		BatchSize    int
		Train        bool
		SplitSoftmax int
	}
	DataWeights struct {
		Weights     []float64
		BiasWeights []float64
	}
)

var activationMap = map[string]activationFunction{}
var splitSoftmax int

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
	ERROR_WEIGHT_MISMATCH     = "[ERROR] Provided weights and bias values do not match the nework structure"
)

func init() {
	//runtime.GOMAXPROCS(1)
}

// New returns an initialized the Neural Network
func New(data NetData) (*Network, error) {
	n := new(Network)
	if len(data.Nodes) < 2 {
		return nil, errors.New(ERROR_LAYERS_COUNT)
	}
	if len(data.Nodes)-1 != len(data.Activations) {
		return nil, errors.New(ERROR_ACTIVATION_COUNT)
	}
	if data.BatchSize <= 0 {
		return nil, errors.New(ERROR_BATCHSIZE)
	}
	for _, node := range data.Nodes {
		if node < 1 {
			return nil, errors.New(ERROR_INT_POSITIVE)
		}
	}
	// The other layers nodes go here
	layerNodes := data.Nodes[1:len(data.Nodes)]
	if data.WeightsData != nil {
		if len(data.WeightsData) != len(layerNodes) {
			return nil, errors.New(ERROR_WEIGHT_MISMATCH)
		}
	}
	// Batchsize of the network
	n.BatchSize = data.BatchSize
	n.Layers = make([]Layer, len(layerNodes))
	n.OutputLayer = len(data.Nodes) - 2
	splitSoftmax = data.SplitSoftmax
	n.Activations = data.Activations
	// Create the input matrice
	n.Input = mat64.NewDense(n.BatchSize, data.Nodes[0], nil)
	n.InputCount = data.Nodes[0]

	// Initialize the Seed for Rand
	rand.Seed(time.Now().UTC().UnixNano())
	//rand.Seed(0)
	// Go through all the nodes and layers
	for k := range layerNodes {
		n.Layers[k].NodesCount = layerNodes[k]
	}
	for k := range n.Layers {
		n.Layers[k].Nodes = mat64.NewDense(n.BatchSize, layerNodes[k], nil)
		// Check if the activation function exists
		act, ok := activationMap[data.Activations[k]]
		if !ok {
			return nil, errors.New(ERROR_UNKNOWN_ACTIVATION)
		}
		// Attach the activation function to the layer
		n.Layers[k].Activation = act
		// Create the BiasWeights vector and seed it with random values
		if data.WeightsData == nil || data.WeightsData[k].BiasWeights == nil {
			n.Layers[k].BiasWeights = mat64.NewVector(layerNodes[k], randomFunc(1, layerNodes[k], -1, 1))
		} else {
			if len(data.WeightsData[k].BiasWeights) != layerNodes[k] {
				return nil, errors.New(ERROR_WEIGHT_MISMATCH)
			}
			n.Layers[k].BiasWeights = mat64.NewVector(layerNodes[k], data.WeightsData[k].BiasWeights)
		}
		switch k {
		// Check if the input matrice is the previous layer
		case 0:
			// Create the Weights matrices and seed them with random values
			if data.WeightsData == nil || data.WeightsData[k].Weights == nil {
				n.Layers[k].Weights = mat64.NewDense(n.InputCount, n.Layers[k].NodesCount, randomFunc(n.InputCount, n.Layers[k].NodesCount, -1, 1))
			} else {
				if len(data.WeightsData[k].Weights) != n.InputCount*n.Layers[k].NodesCount {
					return nil, errors.New(ERROR_WEIGHT_MISMATCH)
				}
				n.Layers[k].Weights = mat64.NewDense(n.InputCount, n.Layers[k].NodesCount, data.WeightsData[k].Weights)
			}
		default:
			// Create the Weights matrices and seed them with random values
			if data.WeightsData == nil || data.WeightsData[k].Weights == nil {
				n.Layers[k].Weights = mat64.NewDense(n.Layers[k-1].NodesCount, n.Layers[k].NodesCount, randomFunc(n.Layers[k-1].NodesCount, n.Layers[k].NodesCount, -1, 1))
			} else {
				if len(data.WeightsData[k].Weights) != n.Layers[k-1].NodesCount*n.Layers[k].NodesCount {
					return nil, errors.New(ERROR_WEIGHT_MISMATCH)
				}
				n.Layers[k].Weights = mat64.NewDense(n.Layers[k-1].NodesCount, n.Layers[k].NodesCount, data.WeightsData[k].Weights)
			}
		}
	}
	if !data.Train {
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
func (n *Network) NetError(target [][]float64) (float64, error) {
	return n.Layers[n.OutputLayer].Activation.layerError(n.Layers[n.OutputLayer].Nodes, target)
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
func (n *Network) Export(path string) (NetData, error) {
	// Number of layers in the network
	layersCount := len(n.Layers)
	export := NetData{
		Nodes:       make([]int, layersCount+1),
		WeightsData: make([]DataWeights, layersCount),
		Activations: make([]string, layersCount),
	}
	export.Nodes[0] = n.InputCount
	for k := range n.Layers {
		// Set the layer node counts
		export.Nodes[k+1] = n.Layers[k].NodesCount
		// Retrieve the weights
		r, c := n.Layers[k].Weights.Dims()
		export.WeightsData[k].Weights = make([]float64, r*c)
		for i := 0; i < r; i++ {
			for i2, v := range n.Layers[k].Weights.RawRowView(i) {
				export.WeightsData[k].Weights[i*c+i2] = v
			}
		}
		// Retrieve the bias weights
		r, _ = n.Layers[k].BiasWeights.Dims()
		export.WeightsData[k].BiasWeights = make([]float64, r)
		for i := 0; i < r; i++ {
			export.WeightsData[k].BiasWeights[i] = n.Layers[k].BiasWeights.At(i, 0)
		}
		// Retrieve the activation functions
		export.Activations = n.Activations
	}
	if path == "" {
		return export, nil
	}
	// create the export file
	dataFile, err := os.Create(path)
	if err != nil {
		return NetData{}, err
	}
	defer dataFile.Close()

	js, err := json.Marshal(export)
	if err != nil {
		return NetData{}, err
	}
	dataFile.Write(js)
	return export, nil
}

// Import takes loads layer weights in to the network
func Import(path string, batchSize int, train bool) (*Network, error) {
	file, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	data := NetData{}
	err = json.Unmarshal(file, &data)
	if err != nil {
		return nil, err
	}
	data.BatchSize = batchSize
	data.Train = train
	fmt.Println("\n", data, "\n")
	return New(data)
}

// Custom override the network weights
func (n *Network) ImportWeights(customWeights []DataWeights) error {
	if len(customWeights) != len(n.Layers) {
		return errors.New(ERROR_WEIGHT_MISMATCH)
	}
	for k := range n.Layers {
		rw, cw := n.Layers[k].Weights.Dims()
		if len(customWeights[k].Weights) != rw*cw {
			return errors.New(ERROR_WEIGHT_MISMATCH)
		}
		_, cb := n.Layers[k].BiasWeights.Dims()
		if len(customWeights[k].Weights) != cb {
			return errors.New(ERROR_WEIGHT_MISMATCH)
		}
		n.Layers[k].Weights = mat64.NewDense(rw, cw, customWeights[k].Weights)
		n.Layers[k].BiasWeights = mat64.NewVector(cb, customWeights[k].BiasWeights)
	}
	return nil
}
