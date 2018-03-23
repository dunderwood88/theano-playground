import numpy as np
import matplotlib.pyplot as plt
import theano
from theano import tensor as T


class Layer:

    def __init__(self, weights_init, bias_init, activation):
        '''
        Individual layer constructor
        :param weights_init: initialized weight matrix connecting nodes between layers
        :param bias_init: initialized bias vector for the layer
        :param activation: activation function for the layer's output
        '''

        dim_output, dim_input = weights_init.shape

        assert bias_init.shape == (dim_output,)

        self.weights = theano.shared(value=weights_init.astype(theano.config.floatX), name='weights', borrow=True)
        self.bias = theano.shared(value=bias_init.reshape(dim_output, 1).astype(theano.config.floatX), name='bias',
                                  borrow=True, broadcastable=(False, True))

        self.activation = activation

        self.params = [self.weights, self.bias]

    def output(self, x):
        '''
        Computes an output based on processing an input vector through a weight matrix,
        adding a bias, and then feeding through an activation function: a(Wx + b)
        Note: activation is element-wise for output vector
        :param x: input feature vector
        :return: the final computational output of the layer (a vector)
        '''

        lin_output = T.dot(self.weights, x) + self.bias
        return lin_output if self.activation is None else self.activation(lin_output)


class MLP:

    def __init__(self, topology):
        '''
        Multilayer perceptron constructor
        :param topology: description of layer sequence defining the MLP
        '''

        self.layers = []

        for n_input, n_output in zip(topology[:-1], topology[1:]):
            self.layers.append(Layer(np.random.randn(n_output, n_input),
                                     np.ones(n_output), T.nnet.sigmoid))

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        '''
        Computes an output based on processing an input vector through multiple layers of the MLP
        :param x: input feature vector
        :return: the final computational output of the MLP
        '''

        # recursively compute the output through each layer
        for layer in self.layers:
            x = layer.output(x)
        return x

    def error(self, x, y):
        '''
        Cost function to be minimized using gradient descent method
        :param x: input
        :param y: target output
        :return: error function
        '''
        return T.sum(-(y * T.log(self.output(x)) + (1 - y) * T.log(1 - self.output(x))))

    def gradient_updates(self, cost, learning_rate):
        '''
        Provides the updates to weight and bias parameters in the MLP
        :param cost: cost function for determining derivatives w.r.t. parameters
        :param learning_rate: rate of gradient descent
        :return: updated parameter list
        '''

        updates = []

        for param in self.params:
            updates.append((param, param - learning_rate * T.grad(cost, param)))

        return updates


# EXAMPLE: XOR Function
inputs = np.vstack(np.array([[0, 0, 1, 1], [0, 1, 0, 1]])).astype(theano.config.floatX)
targets = np.array([1, 0, 0, 1]).astype(theano.config.floatX)

# First, set the size of each layer (and the number of layers)
# Input layer size is training data dimensionality (2)
# Output size is just 1-d: 0 or 1
# Finally, let the hidden layers be twice the size of the input.
# If we wanted more layers, just add another layer size to this list.

# topology = [inputs.shape[0], inputs.shape[0]*2, 1]
topology = [2, 3, 3, 1]
mlp = MLP(topology)

# Create Theano variables for the MLP input and targets
mlp_input = T.matrix('mlp_input')
mlp_target = T.vector('mlp_target')

# Learning rate
learning_rate = 0.01

# Create definition for computing the cost of the network given an input
cost = mlp.error(mlp_input, mlp_target)

# Create a theano function for training the network - parameters are updated based on the cost definition
train = theano.function([mlp_input, mlp_target], cost, updates=mlp.gradient_updates(cost, learning_rate))

# Create a theano function for computing the MLP output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))


iteration = 0
cost = []
max_iteration = 30000
while iteration < max_iteration:

    current_cost = train(inputs, targets)
    cost.append(current_cost)
    iteration += 1

output = mlp_output(inputs)

for i in range(len(inputs[1])):

    print('The output for x1 = %d | x2 = %d is %.2f' % (inputs[0][i], inputs[1][i], output[0][i]))

# plot the cost minimization:
plt.plot(cost)
plt.show()
