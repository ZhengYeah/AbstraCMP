import time
from copy import deepcopy
import numpy as np

class neuron(object):
    """
    Attributes:
        algebra_lower_deeppoly (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper_deeppoly (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower_deeppoly (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper_deeppoly (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)

        concrete_lower_deeppoly (float): neuron's concrete lower bound with deeppoly triangle abstraction
        concrete_upper_deeppoly (float): neuron's concrete upper bound with deeppoly triangle abstraction

        concrete_highest_lower (float): neuron's highest concrete lower bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight        
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
    """

    def __init__(self):
        # algebra bounds represented by the previous layer
        self.algebra_lower_deeppoly = None
        self.algebra_upper_deeppoly = None
        self.algebra_lower_deepz = None
        self.algebra_upper_deepz = None
        # algebra bounds represented by the input layer
        self.concrete_algebra_lower_deeppoly = None
        self.concrete_algebra_upper_deeppoly = None
        self.concrete_algebra_lower_deepz = None
        self.concrete_algebra_upper_deepz = None
        # concrete bounds
        self.concrete_lower_deeppoly = None
        self.concrete_upper_deeppoly = None
        self.concrete_lower_deepz = None
        self.concrete_upper_deepz = None
        # chosen concrete bound
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None

        self.weight = None
        self.bias = None
        self.certain_flag = 0

    def print(self):
        print('algebra_lower_deeppoly:', self.algebra_lower_deeppoly)
        print('algebra_upper_deeppoly:', self.algebra_upper_deeppoly)
        print('algebra_lower_deepz:', self.algebra_lower_deepz)
        print('algebra_upper_deepz:', self.algebra_upper_deepz)
        print('concrete_algebra_lower_deeppoly:', self.concrete_algebra_lower_deeppoly)
        print('concrete_algebra_upper_deeppoly:', self.concrete_algebra_upper_deeppoly)
        print('concrete_algebra_lower_deepz:', self.concrete_algebra_lower_deepz)
        print('concrete_algebra_upper_deepz:', self.concrete_algebra_upper_deepz)
        print('concrete_lower_deeppoly:', self.concrete_lower_deeppoly)
        print('concrete_upper_deeppoly:', self.concrete_upper_deeppoly)
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
                cur_neuron.concrete_algebra_lower_deeppoly = deepcopy(cur_neuron.algebra_lower_deeppoly)
                cur_neuron.concrete_algebra_upper_deeppoly = deepcopy(cur_neuron.algebra_upper_deeppoly)
                cur_neuron.concrete_algebra_lower_deepz = deepcopy(cur_neuron.algebra_lower_deepz)
                cur_neuron.concrete_algebra_upper_deepz = deepcopy(cur_neuron.algebra_upper_deepz)
            lower_bound_deeppoly = deepcopy(cur_neuron.algebra_lower_deeppoly)
            upper_bound_deeppoly = deepcopy(cur_neuron.algebra_upper_deeppoly)
            lower_bound_deepz = deepcopy(cur_neuron.algebra_lower_deepz)
            upper_bound_deepz = deepcopy(cur_neuron.algebra_upper_deepz)

            for k in range(i + 1)[::-1]:
                tmp_lower_deeppoly = np.zeros(len(self.layers[k].neurons[0].algebra_lower_deeppoly))
                tmp_upper_deeppoly = np.zeros(len(self.layers[k].neurons[0].algebra_lower_deeppoly))
                tmp_lower_deepz = np.zeros(len(self.layers[k].neurons[0].algebra_lower_deepz))
                tmp_upper_deepz = np.zeros(len(self.layers[k].neurons[0].algebra_upper_deepz))
                assert (self.layers[k].size + 1 == len(lower_bound_deeppoly))
                assert (self.layers[k].size + 1 == len(upper_bound_deeppoly))

                for p in range(self.layers[k].size):
                    if lower_bound_deeppoly[p] >= 0:
                        tmp_lower_deeppoly += lower_bound_deeppoly[p] * self.layers[k].neurons[p].algebra_lower_deeppoly
                    else:
                        tmp_lower_deeppoly += lower_bound_deeppoly[p] * self.layers[k].neurons[p].algebra_upper_deeppoly
                    if lower_bound_deepz[p] >= 0:
                        tmp_lower_deepz += lower_bound_deepz[p] * self.layers[k].neurons[p].algebra_lower_deepz
                    else:
                        tmp_lower_deepz += lower_bound_deepz[p] * self.layers[k].neurons[p].algebra_upper_deepz

                    if upper_bound_deeppoly[p] >= 0:
                        tmp_upper_deeppoly += upper_bound_deeppoly[p] * self.layers[k].neurons[p].algebra_upper_deeppoly
                    else:
                        tmp_upper_deeppoly += upper_bound_deeppoly[p] * self.layers[k].neurons[p].algebra_lower_deeppoly
                    if upper_bound_deepz[p] >= 0:
                        tmp_upper_deepz += upper_bound_deepz[p] * self.layers[k].neurons[p].algebra_upper_deepz
                    else:
                        tmp_upper_deepz += upper_bound_deepz[p] * self.layers[k].neurons[p].algebra_lower_deepz

                tmp_lower_deeppoly[-1] += lower_bound_deeppoly[-1]
                tmp_upper_deeppoly[-1] += upper_bound_deeppoly[-1]
                tmp_lower_deepz[-1] += lower_bound_deepz[-1]
                tmp_upper_deepz[-1] += upper_bound_deepz[-1]
                lower_bound_deeppoly = deepcopy(tmp_lower_deeppoly)
                upper_bound_deeppoly = deepcopy(tmp_upper_deeppoly)
                lower_bound_deepz = deepcopy(tmp_lower_deepz)
                upper_bound_deepz = deepcopy(tmp_upper_deepz)

                if k == 1:
                    cur_neuron.concrete_algebra_lower_deeppoly = deepcopy(lower_bound_deeppoly)
                    cur_neuron.concrete_algebra_upper_deeppoly = deepcopy(upper_bound_deeppoly)
                    cur_neuron.concrete_algebra_lower_deepz = deepcopy(lower_bound_deepz)
                    cur_neuron.concrete_algebra_upper_deepz = deepcopy(upper_bound_deepz)

            assert (len(lower_bound_deeppoly) == 1)
            assert (len(upper_bound_deeppoly) == 1)
            cur_neuron.concrete_lower_deeppoly = lower_bound_deeppoly[0]
            cur_neuron.concrete_upper_deeppoly = upper_bound_deeppoly[0]
            cur_neuron.concrete_lower_deepz = lower_bound_deepz[0]
            cur_neuron.concrete_upper_deepz = upper_bound_deepz[0]
            assert (cur_neuron.concrete_lower_deeppoly <= cur_neuron.concrete_upper_deeppoly)

            if cur_neuron.concrete_lower_deepz > cur_neuron.concrete_lower_deeppoly:
                self.cnt_deepz_tighter += 1
            if cur_neuron.concrete_upper_deepz < cur_neuron.concrete_upper_deeppoly:
                self.cnt_deepz_tighter += 1

            cur_neuron.concrete_highest_lower = max(cur_neuron.concrete_lower_deeppoly, cur_neuron.concrete_lower_deepz)
            cur_neuron.concrete_lowest_upper = min(cur_neuron.concrete_upper_deeppoly, cur_neuron.concrete_upper_deepz)

        for i in range(len(self.layers) - 1):
            # print('i=',i)
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]
            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons
            if cur_layer.layer_type == layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    cur_neuron.algebra_lower_deeppoly = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper_deeppoly = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_lower_deepz = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper_deepz = np.append(cur_neuron.weight, [cur_neuron.bias])
                    # note: layer index of cur_neuron is i + 1, so pack propagate form i
                    back_propagation(cur_neuron, i)
            elif cur_layer.layer_type == layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]
                    # concrete low bound >= 0, active surely, "or" un-needed?
                    if pre_neuron.concrete_highest_lower >= 0:
                        cur_neuron.algebra_lower_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deeppoly[j] = 1
                        cur_neuron.algebra_upper_deeppoly[j] = 1
                        cur_neuron.algebra_lower_deepz[j] = 1
                        cur_neuron.algebra_upper_deepz[j] = 1
                        cur_neuron.concrete_algebra_lower_deeppoly = deepcopy(pre_neuron.concrete_algebra_lower_deeppoly)
                        cur_neuron.concrete_algebra_upper_deeppoly = deepcopy(pre_neuron.concrete_algebra_upper_deeppoly)
                        cur_neuron.concrete_algebra_lower_deepz = deepcopy(pre_neuron.concrete_algebra_lower_deepz)
                        cur_neuron.concrete_algebra_upper_deepz = deepcopy(pre_neuron.concrete_algebra_upper_deepz)
                        cur_neuron.concrete_lower_deeppoly = pre_neuron.concrete_lower_deeppoly
                        cur_neuron.concrete_upper_deeppoly = pre_neuron.concrete_upper_deeppoly
                        cur_neuron.concrete_lower_deepz = pre_neuron.concrete_lower_deepz
                        cur_neuron.concrete_upper_deepz = pre_neuron.concrete_upper_deepz

                        cur_neuron.concrete_highest_lower = pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper = pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag = 1

                    elif pre_neuron.concrete_lowest_upper <= 0:
                        cur_neuron.algebra_lower_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.concrete_algebra_lower_deeppoly = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper_deeppoly = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_lower_deepz = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper_deepz = np.zeros(self.inputSize)
                        cur_neuron.concrete_lower_deeppoly = 0
                        cur_neuron.concrete_upper_deeppoly = 0
                        cur_neuron.concrete_lower_deepz = 0
                        cur_neuron.concrete_upper_deepz = 0

                        cur_neuron.concrete_highest_lower = 0
                        cur_neuron.concrete_lowest_upper = 0
                        cur_neuron.certain_flag = 2

                    elif pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper <= 0:
                        cur_neuron.algebra_lower_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        alpha = pre_neuron.concrete_lowest_upper / (pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_lower_deepz[j] = alpha

                        cur_neuron.algebra_upper_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deeppoly[j] = alpha
                        cur_neuron.algebra_upper_deeppoly[-1] = -alpha * pre_neuron.concrete_highest_lower
                        cur_neuron.algebra_upper_deepz[j] = alpha
                        cur_neuron.algebra_upper_deepz[-1] = -alpha * pre_neuron.concrete_highest_lower
                        back_propagation(cur_neuron, i)

                    else:
                        cur_neuron.algebra_lower_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower_deepz = np.zeros(cur_layer.size + 1)
                        alpha = pre_neuron.concrete_lowest_upper / (pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_lower_deeppoly[j] = 1
                        cur_neuron.algebra_lower_deepz[j] = alpha

                        cur_neuron.algebra_upper_deeppoly = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deepz = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper_deeppoly[j] = alpha
                        cur_neuron.algebra_upper_deeppoly[-1] = -alpha * pre_neuron.concrete_highest_lower
                        cur_neuron.algebra_upper_deepz[j] = alpha
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

                self.layers[0].neurons[i].concrete_lower_deeppoly = linedata[0]
                self.layers[0].neurons[i].concrete_upper_deeppoly = linedata[1]
                self.layers[0].neurons[i].concrete_lower_deepz = linedata[0]
                self.layers[0].neurons[i].concrete_upper_deepz = linedata[1]
                self.layers[0].neurons[i].concrete_algebra_lower_deeppoly = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper_deeppoly = np.array([linedata[1]])
                self.layers[0].neurons[i].concrete_algebra_lower_deepz = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper_deepz = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower_deeppoly = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper_deeppoly = np.array([linedata[1]])
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


def abstracmp_paper_illustration():
    net = network()
    net.load_nnet('abstracmp_paper_illustration.nnet')
    net.load_robustness('abstracmp_paper_illustration.txt', 1)
    net.abstracmp()
    net.print()

if __name__ == "__main__":
    abstracmp_paper_illustration()
