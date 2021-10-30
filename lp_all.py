import numpy as np
import cvxpy as cp


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

    def load_robustness(self, filename, delta, trim=False):
        if self.property_flag == True:
            self.layers.pop()
        self.property_flag = True
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata = [float(line.strip()) - delta, float(line.strip()) + delta]
                if trim:
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

    def find_max_disturbance(self, PROPERTY, L=0, R=100, trim=False):
        ans = 0
        while L <= R:
            # print(L,R)
            mid = int((L + R) / 2)
            self.load_robustness(PROPERTY, mid / 1000, trim=trim)
            self.lp_all()
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

    def lp_all(self):
        for dim_i in range(len(self.layers)):
            lp_vars = []
            lp_constraints = []
            # Initialize variables from the input layer to the dim_i layer.
            for i in range(dim_i + 1):
                lp_vars.append(cp.Variable(self.layers[i].size))

            # Construct LP constraints from the input layer to the dim_i layer.
            for i in range(dim_i + 1):
                if self.layers[i].layer_type == layer.INPUT_LAYER:
                    for j in range(self.layers[i].size):
                        # i = 0
                        lp_constraints.append(lp_vars[i][j] >= self.layers[i].neurons[j].concrete_lower)
                        lp_constraints.append(lp_vars[i][j] <= self.layers[i].neurons[j].concrete_upper)
                elif self.layers[i].layer_type == layer.AFFINE_LAYER:
                    assert i > 0
                    for j in range(self.layers[i].size):
                        lp_constraints.append(lp_vars[i][j] == self.layers[i].neurons[j].weight @ lp_vars[i - 1] + self.layers[i].neurons[j].bias)
                elif self.layers[i].layer_type == layer.RELU_LAYER:
                    assert self.layers[i].size == self.layers[i - 1].size
                    assert i > 0
                    for j in range(self.layers[i].size):
                        if self.layers[i - 1].neurons[j].concrete_lower >= 0:
                            lp_constraints.append(lp_vars[i][j] == lp_vars[i - 1][j])
                        elif self.layers[i - 1].neurons[j].concrete_upper <= 0:
                            lp_constraints.append(lp_vars[i][j] == 0)
                        elif self.layers[i - 1].neurons[j].concrete_lower < 0 and self.layers[i - 1].neurons[j].concrete_upper > 0:
                            lp_constraints.append(cp.multiply(lp_vars[i][j], self.layers[i - 1].neurons[j].concrete_upper - self.layers[i - 1].neurons[j].concrete_lower) <=
                                                  cp.multiply(self.layers[i - 1].neurons[j].concrete_upper, lp_vars[i - 1][j] - self.layers[i - 1].neurons[j].concrete_lower))
                            lp_constraints.append(lp_vars[i][j] >= 0)
                            lp_constraints.append(lp_vars[i][j] >= lp_vars[i - 1][j])

            # Solve LP for dim_i layer intermediate bounds.
            if dim_i == 0:
                continue
            for j in range(self.layers[dim_i].size):
                if dim_i == len(self.layers) - 1:
                    lp_upper_prob = cp.Problem(cp.Maximize(lp_vars[dim_i][j]), lp_constraints)
                    lp_upper_prob.solve(verbose=False, solver=cp.ECOS)
                    lp_upper_val = lp_vars[dim_i][j].value
                    self.layers[dim_i].neurons[j].concrete_upper = lp_upper_val
                    print('margins:', lp_upper_val)
                    continue

                lp_upper_prob = cp.Problem(cp.Maximize(lp_vars[dim_i][j]), lp_constraints)
                lp_upper_prob.solve(verbose=False, solver=cp.ECOS)
                lp_upper_val = lp_vars[dim_i][j].value
                self.layers[dim_i].neurons[j].concrete_upper = lp_upper_val

                lp_lower_prob = cp.Problem(cp.Minimize(lp_vars[dim_i][j]), lp_constraints)
                lp_lower_prob.solve(verbose=False, solver=cp.ECOS)
                lp_lower_val = lp_vars[dim_i][j].value
                self.layers[dim_i].neurons[j].concrete_lower = lp_lower_val

                # print('Layer: %d, Neuron: %d - LP lower: %.8f, LP upper: %.8f' % (dim_i, j, lp_lower_val, lp_upper_val))


def test_example():
    net = network()
    net.load_nnet('paper_example/abstracmp_paper_illustration.nnet')
    net.load_robustness('paper_example/abstracmp_paper_illustration.txt', 1)
    net.lp_all()


def mnist_robustness_radius():
    net = network()
    net.load_nnet('nnet/mnist_net_10x80.nnet')
    # net.load_robustness('mnist_properties/mnist_property_15.txt', 0.001, trim=True)
    delta_base = net.find_max_disturbance('mnist_properties/mnist_property_15.txt', trim=True)
    print(delta_base)

    # with open('tmp/lp_all', 'w') as f1:
    #     f1.write('%.5f\n' % delta_base)


if __name__ == '__main__':
    mnist_robustness_radius()

