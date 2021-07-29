from copy import deepcopy
import numpy as np

class neuron(object):
    """
    Attributes:
        algebra_lower_right (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper_right (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower_right (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper_right (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)

        concrete_lower_right (float): neuron's concrete lower bound with right triangle abstraction
        concrete_upper_right (float): neuron's concrete upper bound with right triangle abstraction
        concrete_lower_obtuse (float): neuron's concrete lower bound with obtuse abstraction
        concrete_upper_obtuse (float): neuron's concrete upper bound with obtuse abstraction

        concrete_highest_lower (float): neuron's highest concrete lower bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight        
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
    """

    def __init__(self):
        # algebra bounds represented by the previous layer
        self.algebra_lower_right = None
        self.algebra_upper_right = None
        self.algebra_lower_obtuse = None
        self.algebra_upper_obtuse = None
        self.algebra_lower_deepz = None
        self.algebra_upper_deepz = None
        # algebra bounds represented by the input layer
        self.concrete_algebra_lower_obtuse = None
        self.concrete_algebra_upper_obtuse = None
        self.concrete_algebra_lower_right = None
        self.concrete_algebra_upper_right = None
        self.concrete_algebra_lower_deepz = None
        self.concrete_algebra_upper_deepz = None
        # concrete bounds
        self.concrete_lower_obtuse = None
        self.concrete_upper_obtuse = None
        self.concrete_lower_right = None
        self.concrete_upper_right = None
        self.concrete_lower_deepz = None
        self.concrete_upper_deepz = None
        # chosen concrete bound
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None

        self.weight = None
        self.bias = None
        self.certain_flag = 0

    def print(self):
        print('algebra_lower_obtuse:', self.algebra_lower_obtuse)
        print('algebra_upper_obtuse:', self.algebra_upper_obtuse)
        print('algebra_lower_right:', self.algebra_lower_right)
        print('algebra_upper_right:', self.algebra_upper_right)
        print('algebra_lower_deepz:', self.algebra_lower_deepz)
        print('algebra_upper_deepz:', self.algebra_upper_deepz)
        print('concrete_algebra_lower_obtuse:', self.concrete_algebra_lower_obtuse)
        print('concrete_algebra_upper_obtuse:', self.concrete_algebra_upper_obtuse)
        print('concrete_algebra_lower_right:', self.concrete_algebra_lower_right)
        print('concrete_algebra_upper_right:', self.concrete_algebra_upper_right)
        print('concrete_algebra_lower_deepz:', self.concrete_algebra_lower_deepz)
        print('concrete_algebra_upper_deepz:', self.concrete_algebra_upper_deepz)
        print('concrete_lower_obtuse:', self.concrete_lower_obtuse)
        print('concrete_upper_obtuse:', self.concrete_upper_obtuse)
        print('concrete_lower_right:', self.concrete_lower_right)
        print('concrete_upper_right:', self.concrete_upper_right)
        print("concrete_lower_deepz:", self.concrete_lower_deepz)
        print("concrete_upper_deepz:", self.concrete_upper_deepz)
        print("concrete_highest_lower:", self.concrete_highest_lower)
        print("concrete_lowest_upper:", self.concrete_lowest_upper)
        print('weight:', self.weight)
        print('bias:', self.bias)
        print('certain_flag:', self.certain_flag)


class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER = 0
    AFFINE_LAYER = 1
    RELU_LAYER = 2

    def __init__(self):
        self.size = None
        self.neurons = None
        self.layer_type = None

    def print(self):
        print('Layer size:', self.size)
        print('Layer type:', self.layer_type)
        print('Neurons:')
        for neu in self.neurons:
            neu.print()
            print('\n')


class network(object):
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
    """

    def __init__(self):
        self.numlayers = None
        self.layerSizes = None
        self.inputSize = None
        self.outputSize = None
        self.mins = None
        self.maxes = None
        self.ranges = None
        self.layers = None
        self.cnt_deepz_tighter = 0
        self.property_flag = None

    def abstracmp(self):

        def back_propagation(cur_neuron, i):
            if i == 0:
                cur_neuron.concrete_algebra_lower_obtuse = deepcopy(cur_neuron.algebra_lower_obtuse)
                cur_neuron.concrete_algebra_upper_obtuse = deepcopy(cur_neuron.algebra_upper_obtuse)
                cur_neuron.concrete_algebra_lower_right = deepcopy(cur_neuron.algebra_lower_right)
                cur_neuron.concrete_algebra_upper_right = deepcopy(cur_neuron.algebra_upper_right)
                cur_neuron.concrete_algebra_lower_deepz = deepcopy(cur_neuron.algebra_lower_deepz)
                cur_neuron.concrete_algebra_upper_deepz = deepcopy(cur_neuron.algebra_upper_deepz)
            lower_bound_obtuse = deepcopy(cur_neuron.algebra_lower_obtuse)
            upper_bound_obtuse = deepcopy(cur_neuron.algebra_upper_obtuse)
            lower_bound_right = deepcopy(cur_neuron.algebra_lower_right)
            upper_bound_right = deepcopy(cur_neuron.algebra_upper_right)
            lower_bound_deepz = deepcopy(cur_neuron.algebra_lower_deepz)
            upper_bound_deepz = deepcopy(cur_neuron.algebra_upper_deepz)

            for k in range(i + 1)[::-1]:
                tmp_lower_obtuse = np.zeros(len(self.layers[k].neurons[0].algebra_lower_right))
                tmp_upper_obtuse = np.zeros(len(self.layers[k].neurons[0].algebra_lower_right))
                tmp_lower_right = np.zeros(len(self.layers[k].neurons[0].algebra_lower_right))
                tmp_upper_right = np.zeros(len(self.layers[k].neurons[0].algebra_lower_right))
                tmp_lower_deepz = np.zeros(len(self.layers[k].neurons[0].algebra_lower_deepz))
                tmp_upper_deepz = np.zeros(len(self.layers[k].neurons[0].algebra_upper_deepz))
                assert (self.layers[k].size + 1 == len(lower_bound_right))
                assert (self.layers[k].size + 1 == len(upper_bound_right))

                for p in range(self.layers[k].size):
                    if lower_bound_right[p] >= 0:
                        tmp_lower_right += lower_bound_right[p] * self.layers[k].neurons[p].algebra_lower_right
                    else:
                        tmp_lower_right += lower_bound_right[p] * self.layers[k].neurons[p].algebra_upper_right
                    if lower_bound_obtuse[p] >= 0:
                        tmp_lower_obtuse += lower_bound_obtuse[p] * self.layers[k].neurons[p].algebra_lower_obtuse
                    else:
                        tmp_lower_obtuse += lower_bound_obtuse[p] * self.layers[k].neurons[p].algebra_upper_obtuse
                    if lower_bound_deepz[p] >= 0:
                        tmp_lower_deepz += lower_bound_deepz[p] * self.layers[k].neurons[p].algebra_lower_deepz
                    else:
                        tmp_lower_deepz += lower_bound_deepz[p] * self.layers[k].neurons[p].algebra_upper_deepz

                    if upper_bound_right[p] >= 0:
                        tmp_upper_right += upper_bound_right[p] * self.layers[k].neurons[p].algebra_upper_right
                    else:
                        tmp_upper_right += upper_bound_right[p] * self.layers[k].neurons[p].algebra_lower_right
                    if upper_bound_obtuse[p] >= 0:
                        tmp_upper_obtuse += upper_bound_obtuse[p] * self.layers[k].neurons[p].algebra_upper_obtuse
                    else:
                        tmp_upper_obtuse += upper_bound_obtuse[p] * self.layers[k].neurons[p].algebra_lower_obtuse
                    if upper_bound_deepz[p] >= 0:
                        tmp_upper_deepz += upper_bound_deepz[p] * self.layers[k].neurons[p].algebra_upper_deepz
                    else:
                        tmp_upper_deepz += upper_bound_deepz[p] * self.layers[k].neurons[p].algebra_lower_deepz

                tmp_lower_obtuse[-1] += lower_bound_obtuse[-1]
                tmp_upper_obtuse[-1] += upper_bound_obtuse[-1]
                tmp_lower_right[-1] += lower_bound_right[-1]
                tmp_upper_right[-1] += upper_bound_right[-1]
                tmp_lower_deepz[-1] += lower_bound_deepz[-1]
                tmp_upper_deepz[-1] += upper_bound_deepz[-1]
                lower_bound_obtuse = deepcopy(tmp_lower_obtuse)
                upper_bound_obtuse = deepcopy(tmp_upper_obtuse)
                lower_bound_right = deepcopy(tmp_lower_right)
                upper_bound_right = deepcopy(tmp_upper_right)
                lower_bound_deepz = deepcopy(tmp_lower_deepz)
                upper_bound_deepz = deepcopy(tmp_upper_deepz)

                if k == 1:
                    cur_neuron.concrete_algebra_lower_obtuse = deepcopy(lower_bound_obtuse)
                    cur_neuron.concrete_algebra_upper_obtuse = deepcopy(upper_bound_obtuse)
                    cur_neuron.concrete_algebra_lower_right = deepcopy(lower_bound_right)
                    cur_neuron.concrete_algebra_upper_right = deepcopy(upper_bound_right)
                    cur_neuron.concrete_algebra_lower_deepz = deepcopy(lower_bound_deepz)
                    cur_neuron.concrete_algebra_upper_deepz = deepcopy(upper_bound_deepz)

            assert (len(lower_bound_right) == 1)
            assert (len(upper_bound_right) == 1)
            cur_neuron.concrete_lower_obtuse = lower_bound_obtuse[0]
            cur_neuron.concrete_upper_obtuse = upper_bound_obtuse[0]
            cur_neuron.concrete_lower_right = lower_bound_right[0]
            cur_neuron.concrete_upper_right = upper_bound_right[0]
            cur_neuron.concrete_lower_deepz = lower_bound_deepz[0]
            cur_neuron.concrete_upper_deepz = upper_bound_deepz[0]
            assert (cur_neuron.concrete_lower_right <= cur_neuron.concrete_upper_right)

            if cur_neuron.concrete_lower_deepz > cur_neuron.concrete_lower_right and cur_neuron.concrete_lower_deepz > cur_neuron.concrete_lower_obtuse:
                self.cnt_deepz_tighter += 1
            if cur_neuron.concrete_upper_deepz < cur_neuron.concrete_upper_right and cur_neuron.concrete_upper_deepz < cur_neuron.concrete_upper_obtuse:
                self.cnt_deepz_tighter += 1

            cur_neuron.concrete_highest_lower = max(cur_neuron.concrete_lower_right, cur_neuron.concrete_lower_obtuse, cur_neuron.concrete_lower_deepz)
            cur_neuron.concrete_lowest_upper = min(cur_neuron.concrete_upper_right, cur_neuron.concrete_upper_obtuse, cur_neuron.concrete_upper_deepz)

        for i in range(len(self.layers) - 1):
            # print('i=',i)
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]
            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons
            if cur_layer.layer_type == layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    cur_neuron.algebra_lower_obtuse = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper_obtuse = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_lower_right = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper_right = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_lower_deepz = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper_deepz = np.append(cur_neuron.weight, [cur_neuron.bias])
                    back_propagation(cur_neuron, i)
            elif cur_layer.layer_type == layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]
                    # concrete low bound >= 0, active surely, "or" un-needed?
                    if pre_neuron.concrete_highest_lower >= 0:
                        cur_neuron.algebra_lower_obtuse = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_obtuse = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_right = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_right = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_obtuse[j] = 1
                        cur_neuron.algebra_upper_obtuse[j] = 1
                        cur_neuron.algebra_lower_right[j] = 1
                        cur_neuron.algebra_upper_right[j] = 1
                        cur_neuron.algebra_lower_deepz[j] = 1
                        cur_neuron.algebra_upper_deepz[j] = 1
                        cur_neuron.concrete_algebra_lower_obtuse = deepcopy(pre_neuron.concrete_algebra_lower_obtuse)
                        cur_neuron.concrete_algebra_upper_obtuse = deepcopy(pre_neuron.concrete_algebra_upper_obtuse)
                        cur_neuron.concrete_algebra_lower_right = deepcopy(pre_neuron.concrete_algebra_lower_right)
                        cur_neuron.concrete_algebra_upper_right = deepcopy(pre_neuron.concrete_algebra_upper_right)
                        cur_neuron.concrete_algebra_lower_deepz = deepcopy(pre_neuron.concrete_algebra_lower_deepz)
                        cur_neuron.concrete_algebra_upper_deepz = deepcopy(pre_neuron.concrete_algebra_upper_deepz)
                        cur_neuron.concrete_lower_obtuse = pre_neuron.concrete_lower_obtuse
                        cur_neuron.concrete_upper_obtuse = pre_neuron.concrete_upper_obtuse
                        cur_neuron.concrete_lower_right = pre_neuron.concrete_lower_right
                        cur_neuron.concrete_upper_right = pre_neuron.concrete_upper_right
                        cur_neuron.concrete_lower_deepz = pre_neuron.concrete_lower_deepz
                        cur_neuron.concrete_upper_deepz = pre_neuron.concrete_upper_deepz

                        cur_neuron.concrete_highest_lower = pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper = pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag = 1

                    elif pre_neuron.concrete_lowest_upper <= 0:
                        cur_neuron.algebra_lower_obtuse = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_obtuse = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_right = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_right = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.concrete_algebra_lower_obtuse = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper_obtuse = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_lower_right = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper_right = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_lower_deepz = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper_deepz = np.zeros(self.inputSize)
                        cur_neuron.concrete_lower_obtuse = 0
                        cur_neuron.concrete_upper_obtuse = 0
                        cur_neuron.concrete_lower_right = 0
                        cur_neuron.concrete_upper_right = 0
                        cur_neuron.concrete_lower_deepz = 0
                        cur_neuron.concrete_upper_deepz = 0

                        cur_neuron.concrete_highest_lower = 0
                        cur_neuron.concrete_lowest_upper = 0
                        cur_neuron.certain_flag = 2

                    else:
                        cur_neuron.algebra_lower_obtuse = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_right = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        alpha = pre_neuron.concrete_lowest_upper / (pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_lower_obtuse[j] = 1
                        cur_neuron.algebra_lower_deepz[j] = alpha

                        cur_neuron.algebra_upper_obtuse = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_right = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_obtuse[j] = alpha
                        cur_neuron.algebra_upper_right[j] = alpha
                        cur_neuron.algebra_upper_deepz[j] = alpha
                        cur_neuron.algebra_upper_right[-1] = -alpha * pre_neuron.concrete_highest_lower
                        cur_neuron.algebra_upper_obtuse[-1] = -alpha * pre_neuron.concrete_highest_lower
                        cur_neuron.algebra_upper_deepz[-1] = -alpha * pre_neuron.concrete_highest_lower
                        back_propagation(cur_neuron, i)

    def print(self):
        print('numlayers:', self.numLayers)
        print('layerSizes:', self.layerSizes)
        print("inputSize:", self.inputSize)
        print('outputSize:', self.outputSize)
        print('mins:', self.mins)
        print('maxes:', self.maxes)
        print('ranges:', self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')
        print("Deepz_tighter:", self.cnt_deepz_tighter)

    def load_robustness(self, filename, delta, TRIM=False):
        if self.property_flag == True:
            self.layers.pop()
        self.property_flag = True
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata = [float(line.strip()) - delta, float(line.strip()) + delta]
                if TRIM:
                    if linedata[0] < 0: linedata[0] = 0
                    if linedata[1] > 1: linedata[1] = 1

                self.layers[0].neurons[i].concrete_lower_obtuse = linedata[0]
                self.layers[0].neurons[i].concrete_upper_obtuse = linedata[1]
                self.layers[0].neurons[i].concrete_lower_right = linedata[0]
                self.layers[0].neurons[i].concrete_upper_right = linedata[1]
                self.layers[0].neurons[i].concrete_lower_deepz = linedata[0]
                self.layers[0].neurons[i].concrete_upper_deepz = linedata[1]
                self.layers[0].neurons[i].concrete_algebra_lower_obtuse = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper_obtuse = np.array([linedata[1]])
                self.layers[0].neurons[i].concrete_algebra_lower_right = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper_right = np.array([linedata[1]])
                self.layers[0].neurons[i].concrete_algebra_lower_deepz = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper_deepz = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower_obtuse = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper_obtuse = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower_right = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper_right = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower_deepz = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper_deepz = np.array([linedata[1]])

            line = f.readline()
            verify_layer = layer()
            verify_layer.neurons = []
            while line:
                linedata = [float(x) for x in line.strip().split(' ')]
                assert (len(linedata) == self.outputSize + 1)
                verify_neuron = neuron()
                verify_neuron.weight = np.array(linedata[:-1])
                verify_neuron.bias = linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata = np.array(linedata)
                # print(linedata)
                assert (len(linedata) == self.outputSize + 1)
                line = f.readline()
            verify_layer.size = len(verify_layer.neurons)
            verify_layer.layer_type = layer.AFFINE_LAYER
            if len(verify_layer.neurons) > 0:
                self.layers.append(verify_layer)

    def load_nnet(self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line = f.readline()
                cnt += 1
            # numLayers doesn't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            # input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            # process the input layer
            self.layers = []
            new_layer = layer()
            new_layer.layer_type = layer.INPUT_LAYER
            new_layer.size = layerSizes[0]
            new_layer.neurons = []
            for i in range(layerSizes[0]):
                new_neuron = neuron()
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)

            for layernum in range(numLayers):
                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum + 1]
                new_layer = layer()
                new_layer.size = currentLayerSize
                new_layer.layer_type = layer.AFFINE_LAYER
                new_layer.neurons = []

                # weights
                for i in range(currentLayerSize):
                    line = f.readline()
                    new_neuron = neuron()
                    weight = [float(x) for x in line.strip().split(",")[:-1]]
                    assert (len(weight) == previousLayerSize)
                    new_neuron.weight = np.array(weight)
                    new_layer.neurons.append(new_neuron)
                # biases
                for i in range(currentLayerSize):
                    line = f.readline()
                    x = float(line.strip().split(",")[0])
                    new_layer.neurons[i].bias = x

                self.layers.append(new_layer)

                # add relu layer
                if layernum + 1 == numLayers:
                    break
                new_layer = layer()
                new_layer.size = currentLayerSize
                new_layer.layer_type = layer.RELU_LAYER
                new_layer.neurons = []
                for i in range(currentLayerSize):
                    new_neuron = neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges

    def load_rlv(self, filename):
        layersize = []
        dicts = []
        self.layers = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('#'):
                    linedata = line.replace('\n', '').split(' ')
                    layersize.append(int(linedata[3]))
                    layerdict = {}
                    if linedata[4] == 'Input':
                        new_layer = layer()
                        new_layer.layer_type = layer.INPUT_LAYER
                        new_layer.size = layersize[-1]
                        new_layer.neurons = []
                        for i in range(layersize[-1]):
                            new_neuron = neuron()
                            new_layer.neurons.append(new_neuron)
                            line = f.readline()
                            linedata = line.split(' ')
                            layerdict[linedata[1].replace('\n', '')] = i
                        dicts.append(layerdict)
                        self.layers.append(new_layer)
                    elif linedata[4] == 'ReLU':
                        new_layer = layer()
                        new_layer.layer_type = layer.AFFINE_LAYER
                        new_layer.size = layersize[-1]
                        new_layer.neurons = []
                        for i in range(layersize[-1]):
                            new_neuron = neuron()
                            new_neuron.weight = np.zeros(layersize[-2])
                            line = f.readline()
                            linedata = line.replace('\n', '').split(' ')
                            layerdict[linedata[1]] = i
                            new_neuron.bias = float(linedata[2])
                            nodeweight = linedata[3::2]
                            nodename = linedata[4::2]
                            assert (len(nodeweight) == len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]] = float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)
                        dicts.append(layerdict)
                        # add relu layer
                        new_layer = layer()
                        new_layer.layer_type = layer.RELU_LAYER
                        new_layer.size = layersize[-1]
                        new_layer.neurons = []
                        for i in range(layersize[-1]):
                            new_neuron = neuron()
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)
                    elif (linedata[4] == 'Linear') and (linedata[5] != 'Accuracy'):
                        new_layer = layer()
                        new_layer.layer_type = layer.AFFINE_LAYER
                        new_layer.size = layersize[-1]
                        new_layer.neurons = []
                        for i in range(layersize[-1]):
                            new_neuron = neuron()
                            new_neuron.weight = np.zeros(layersize[-2])
                            line = f.readline()
                            linedata = line.replace('\n', '').split(' ')
                            layerdict[linedata[1]] = i
                            new_neuron.bias = float(linedata[2])
                            nodeweight = linedata[3::2]
                            nodename = linedata[4::2]
                            assert (len(nodeweight) == len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]] = float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)
                        dicts.append(layerdict)
                line = f.readline()
        self.layerSizes = layersize
        self.inputSize = layersize[0]
        self.outputSize = layersize[-1]
        self.numLayers = len(layersize) - 1

    def find_max_disturbance(self, PROPERTY, L=0, R=1000, TRIM=False):
        ans = 0
        while L <= R:
            # print(L,R)
            mid = int((L + R) / 2)
            self.load_robustness(PROPERTY, mid / 1000, TRIM=TRIM)
            self.abstracmp()
            flag = True
            for neuron_i in self.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_lowest_upper > 0:
                    flag = False
            if flag == True:
                ans = mid / 1000
                L = mid + 1
            else:
                R = mid - 1
        return ans

def main():
    net = network()
    net.load_rlv("rlv/caffeprototxt_AI2_MNIST_FNN_1_testNetworkB.rlv")
    property_list = ["properties/mnist_" + str(i) + "_local_property.in" for i in range(50)]
    delta = 0.06
    for property_i in property_list:
        net.load_robustness(property_i, delta, TRIM=True)
        net.abstracmp()
        flag = True
        for neuron_i in net.layers[-1].neurons:
            if neuron_i.concrete_lowest_upper > 0:
                flag = False
        if flag == True:
            print("Success!")
        else:
            print("Failed!")


def acas_robustness_radius():
    net = network()
    net.load_nnet("nnet/ACASXU_experimental_v2a_3_2.nnet")
    property_list = ["acas_properties/local_robustness_" + str(i) + ".txt" for i in range(2, 7)]
    for property_i in property_list:
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=False)
        print(delta_base)

def test_example():
    net = network()
    net.load_nnet("nnet/abstracmp_paper_illustration.nnet")
    net.load_robustness("properties/abstracmp_paper_illustration.txt", 1)
    net.abstracmp()
    net.print()

if __name__ == "__main__":
    acas_robustness_radius()
