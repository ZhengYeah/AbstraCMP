from copy import deepcopy
import numpy as np
import time

class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        weight (numpy ndarray of float): neuron's weight        
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
    """

    def __init__(self):
        self.algebra_lower = None
        self.algebra_upper = None
        self.concrete_algebra_lower = None
        self.concrete_algebra_upper = None
        self.concrete_lower = None
        self.concrete_upper = None

        self.weight = None
        self.bias = None
        self.certain_flag = 0

    def print(self):
        print('algebra_lower:', self.algebra_lower)
        print('algebra_upper:', self.algebra_upper)
        print('concrete_algebra_lower:', self.concrete_algebra_lower)
        print('concrete_algebra_upper:', self.concrete_algebra_upper)
        print('concrete_lower:', self.concrete_lower)
        print('concrete_upper:', self.concrete_upper)
        # print('weight:', self.weight)
        # print('bias:', self.bias)
        # print('certain_flag:', self.certain_flag)


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
        self.property_flag = None

    def deeppoly(self):

        def back_propagation(cur_neuron, i):
            if i == 0:
                cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)
            lower_bound = deepcopy(cur_neuron.algebra_lower)
            upper_bound = deepcopy(cur_neuron.algebra_upper)
            for k in range(i + 1)[::-1]:
                tmp_lower = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                for p in range(self.layers[k].size):
                    if lower_bound[p] >= 0:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_lower
                    else:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p] >= 0:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_lower
                tmp_lower[-1] += lower_bound[-1]
                tmp_upper[-1] += upper_bound[-1]
                lower_bound = deepcopy(tmp_lower)
                upper_bound = deepcopy(tmp_upper)

                if k == 1:
                    cur_neuron.concrete_algebra_lower = deepcopy(lower_bound)
                    cur_neuron.concrete_algebra_upper = deepcopy(upper_bound)

            assert (len(lower_bound) == 1)
            assert (len(upper_bound) == 1)
            cur_neuron.concrete_lower = lower_bound[0]
            cur_neuron.concrete_upper = upper_bound[0]

        for i in range(len(self.layers) - 1):
            # print('i=',i)
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]
            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons
            
            if cur_layer.layer_type == layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    cur_neuron.algebra_lower = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper = np.append(cur_neuron.weight, [cur_neuron.bias])
                    # note: layer index of cur_neuron is i + 1, so pack propagate form i
                    back_propagation(cur_neuron, i)

            elif cur_layer.layer_type == layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]

                    if pre_neuron.concrete_lower >= 0:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        cur_neuron.algebra_upper[j] = 1
                        cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower = pre_neuron.concrete_lower
                        cur_neuron.concrete_upper = pre_neuron.concrete_upper
                        cur_neuron.certain_flag = 1

                    elif pre_neuron.concrete_upper <= 0:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize)
                        cur_neuron.concrete_lower = 0
                        cur_neuron.concrete_upper = 0
                        cur_neuron.certain_flag = 2

                    elif pre_neuron.concrete_lower + pre_neuron.concrete_upper <= 0:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = alpha
                        cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
                        back_propagation(cur_neuron, i)

                    else:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = alpha
                        cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
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

                self.layers[0].neurons[i].concrete_lower = linedata[0]
                self.layers[0].neurons[i].concrete_upper = linedata[1]
                self.layers[0].neurons[i].concrete_algebra_lower = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper = np.array([linedata[1]])

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
            self.deeppoly()
            flag = True
            for neuron_i in self.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper > 0:
                    flag = False
            if flag == True:
                ans = mid / 1000
                L = mid + 1
            else:
                R = mid - 1
        return ans


def main():
    net = network()
    net.load_nnet("nnet/cifar_net.nnet")
    property_list = ["mnist_properties/cifar_input.txt"]
    delta = 0.0
    for property_i in property_list:
        net.load_robustness(property_i, delta, TRIM=True)
        net.deeppoly()
        flag = True
        for neuron_i in net.layers[-1].neurons:
            if neuron_i.concrete_upper > 0:
                flag = False
        if flag == True:
            print("Success!")
        else:
            print("Failed!")


def mnist_robustness_radius():
    net = network()
    net.load_nnet("nnet/mnist_net_20x50.nnet")
    property_list = ["mnist_properties/mnist_property_" + str(i) + ".txt" for i in range(100)]
    for property_i in property_list:
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        print(delta_base)


def acas_robustness_radius():
    for recur in range(1, 11):
        print("Recur:", recur)
        net = network()
        net.load_nnet("nnet/ACASXU_experimental_v2a_2_3.nnet")
        property_list = ["acas_properties/local_robustness_" + str(i) + ".txt" for i in range(2, 3)]
        for property_i in property_list:
            star_time = time.time()
            delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=False)
            end_time = time.time()
            print('Time:', end_time - star_time)
            print(delta_base)


def cifar_robustness_radius():
    net = network()
    net.load_nnet("nnet/cifar_net_10x200.nnet")
    property_list = ["cifar_properties_10x200/cifar_property_" + str(i) + ".txt" for i in range(50)]
    for property_i in property_list:
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        print(delta_base)


def test_example():
    net = network()
    net.load_nnet('others/abstracmp_paper_illustration.nnet')
    net.load_robustness('others/abstracmp_paper_illustration.txt', 1)
    net.deeppoly()
    net.print()


if __name__ == "__main__":
    test_example()