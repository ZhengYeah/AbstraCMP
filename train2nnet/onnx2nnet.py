import numpy as np
import onnx
from onnx import numpy_helper


def onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    '''
    Write a .nnet file from an onnx file

    Args:
        onnxFile: (string) Path to onnx file
        inputMins: (list) optional, Minimum values for each neural network input
        inputMaxes: (list) optional, Maximum values for each neural network output
        means: (list) optional, Mean value for each input and value for mean of all outputs, used for normalization
        ranges: (list) optional, Range value for each input and value for range of all outputs, used for normalization
        inputName: (string) optional, Name of operation corresponding to input
        outputName: (string) optional, Name of operation corresponding to output
    '''

    if nnetFile == "":
        nnetFile = onnxFile[:-4] + 'nnet'

    model = onnx.load(onnxFile)
    graph = model.graph

    if not inputName:
        assert len(graph.input) == 1
        inputName = graph.input[0].name
    if not outputName:
        assert len(graph.output) == 1
        outputName = graph.output[0].name

    # Search through nodes until we find the inputName
    # Accumulate the weight matrices and bias vectors into lists
    # Continue through the network until we reach outputName
    # This assumes that the network is "frozen", and the model uses initializers to set weight and bias array values
    weights = []
    biases = []

    # Loop through nodes in graph
    for node in graph.node:
        # Ignore nodes that do not use inputName as an input to the node
        if inputName in node.input:
            if node.op_type == "Gemm":
                assert len(node.input) == 3
                weightIndex = 0
                if node.input[0] == inputName:
                    weightIndex = 1
                    biasIndex = 2
                weightName = node.input[weightIndex]
                weights += [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name == weightName]
                biasName = node.input[biasIndex]
                biases += [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name == biasName]
                inputName = node.output[0]
            elif node.op_type == "Relu":
                inputName = node.output[0]
            else:
                print("Node operation type %s not supported!" % node.op_type)
                weights = []
                biases = []
                break
            # Terminate once we find the outputName in the graph
            if outputName == inputName:
                break

    # Check if the weights and biases were extracted correctly from the graph
    if outputName == inputName and len(weights) > 0 and len(weights) == len(biases):
        inputSize = weights[0].shape[1]

        # Default values for input bounds and normalization constants
        if inputMins is None: inputMins = inputSize * [np.finfo(np.float32).min]
        if inputMaxes is None: inputMaxes = inputSize * [np.finfo(np.float32).max]
        if means is None: means = (inputSize + 1) * [0.0]
        if ranges is None: ranges = (inputSize + 1) * [1.0]

        # Print statements
        print("Converted ONNX model at %s" % onnxFile)
        print("    to an NNet model at %s" % nnetFile)
        # Write NNet file
        writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
    # Something went wrong, so don't write the NNet file
    else:
        print("Could not write NNet file!")


def writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, fileName):
    '''
    Write network data to the .nnet file format

    Args:
        weights (list): Weight matrices in the network order
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''

    with open(fileName, 'w') as f2:
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        # Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[1]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize

        # Find maximum size of any hidden layer
        for b in biases:
            if len(b) > maxLayerSize:
                maxLayerSize = len(b)

        # Write data to header
        f2.write("%d,%d,%d,%d,\n" % (numLayers, inputSize, outputSize, maxLayerSize))
        f2.write("%d," % inputSize)
        for b in biases:
            f2.write("%d," % len(b))
        f2.write("\n")
        f2.write("0,\n")  # Unused Flag

        # Write Min, Max, Mean, and Range of each of the inputs and outputs for normalization
        f2.write(','.join(str(inputMins[i]) for i in range(inputSize)) + ',\n')  # Minimum Input Values
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + ',\n')  # Maximum Input Values
        f2.write(','.join(str(means[i]) for i in range(inputSize + 1)) + ',\n')  # Means for normalizations
        f2.write(','.join(str(ranges[i]) for i in range(inputSize + 1)) + ',\n')  # Ranges for noramlizations

        # Write weights and biases of neural network
        # First, the weights from the input layer to the first hidden layer are written
        # Then, the biases of the first hidden layer are written
        # The pattern is repeated by next writing the weights from the first hidden layer to the second hidden layer,
        # followed by the biases of the second hidden layer
        for w, b in zip(weights, biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f2.write(
                        "%.5e," % w[i][j])  # Five digits written. More can be used, but that requires more more space
                f2.write("\n")

            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i])  # Five digits written. More can be used, but that requires more more space


if __name__ == '__main__':
    onnx2nnet("checkpoint/mnist_net_gradient.onnx")
