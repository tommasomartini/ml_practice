import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import neural_networks.activations as activations

import common.point_drawer as drawer

sns.set()
np.random.seed(2)

_min_x = -5
_max_x = 5

_min_y = - 2
_max_y = 2


_learning_rate = 0.01
_num_epochs = 1000
_input_dims = 2     # input is 2 dimensional
_output_dims = 1

# The size of each hidden layer.
_layers = [3]


class NeuralNetwork:

    def __init__(self, hidden_layers, activation):
        # Add the output layer.
        self._hidden_layers = hidden_layers + [_output_dims]
        self._activation = activation

        self._weights = []
        self._biases = []

        # Store the gradients of the parameters.
        self._weights_grads = []
        self._biases_grads = []

        # Store the intermediate features.
        self._features_before_act = []
        self._features_after_act = []

        num_inputs = _input_dims
        for layer_idx, num_neurons in enumerate(self._hidden_layers):
            layer_weights = np.zeros((num_inputs, num_neurons))
            layer_biases = np.zeros((num_neurons,))

            self._weights.append(layer_weights)
            self._biases.append(layer_biases)

            num_inputs = num_neurons

    def _clear(self):
        # Store the gradients of the parameters.
        self._weights_grads = [None] * len(self._hidden_layers)
        self._biases_grads = [None] * len(self._hidden_layers)

        # Store the intermediate features.
        self._features_before_act = []
        self._features_after_act = []

    def fw(self, x):
        # Performs a forward pass.
        # x has shape (<batch size>, D)
        self._clear()
        self._features_after_act.append(x)
        for W, b in zip(self._weights, self._biases):
            x = x @ W + b
            self._features_before_act.append(x)

            x = self._activation.fw(x)
            self._features_after_act.append(x)

        return x

    def grad(self, loss_grad):
        """Performs a backpropagation pass.

        Given X in input matrix of shape (N, D), whose N rows are the N
        D-dimensional input vectors:
            Y = f(Z) = f(X @ W + b)

        W is (D{l-1}, D{l})
            with D{l-1} number of neurons in the previous layer and D{l} number
            of layers in the next layer.
        B is (D{l},) -> apply dimension propagation in numpy
        Z and Y are (N, D{l}).

        The derivative of the loss wrt the parameters is:
            dL/dW = dL/dY dY/dZ dZ/dW
            dL/dB = dL/dY dY/dZ dZ/dB

        and the derivative of the loss wrt the input to a neuron is:
            dL/dX = dL/dY dY/dZ dZ/dX

        Note that dL/dY is the derivative of the loss wrt the inputs of the
        following layer. For the last layer dL/dY is given as argument of
        this function.

        dY/dZ is the derivative of the activation function.

        dZ/dW = X
        dZ/dB = np.ones()
        dZ/dX = W

        Args:
            loss_grad: Array of shape (<batch size>, 1).
        """
        dL_dY = loss_grad
        for layer_idx in range(len(self._hidden_layers) - 1, -1, -1):
            # dL_dY has shape (batch, D{l}).

            input_to_activation = self._features_before_act[layer_idx]

            batch_size = input_to_activation.shape[0]
            D = input_to_activation.shape[1]

            # Shape (batch, D{l}).
            dY_dZ = self._activation.grad(input_to_activation)

            # Shape (batch, D{l-1}).
            dZ_dW = self._features_after_act[layer_idx]

            # Shape (batch, D{l}).
            dZ_dB = np.ones((batch_size, D))

            # Shape (D{l-1}, D{l}).
            dZ_dX = self._weights[layer_idx]

            # Shape (batch, D{l}).
            dL_dZ = dL_dY * dY_dZ

            # Shape (batch, D{l-1}, D{l}).
            #  (batch, 1, D{l}) x (batch, D{l-1}, 1) = (batch, D{l-1}, D{l})
            dL_dW = np.expand_dims(dL_dZ, axis=1) * np.expand_dims(dZ_dW, axis=2)
            gradW = np.mean(dL_dW, axis=0)
            self._weights_grads[layer_idx] = gradW

            # Shape (batch, D{l}).
            dL_dB = dL_dZ * dZ_dB
            gradB = np.mean(dL_dB, axis=0)
            self._biases_grads[layer_idx] = gradB

            # Shape (batch, D{l-1}).
            # The gradient of the input vector depends on the gradient coming
            # from each neuron of the layer. Therefore we sum across all the
            # gradients.
            #   (batch,    D{l}) x (   D{l-1}, D{l}) -> expand
            #   (batch, 1, D{l}) x (1, D{l-1}, D{l})
            # = (batch, D{l-1}, D{l}) -> sum along the last dimension
            # -> (batch, D{l-1})
            dL_dX = np.sum(
                np.expand_dims(dL_dZ, axis=1) * np.expand_dims(dZ_dX, axis=0),
                axis=2
            )

            # The input of this layer is the output of the previous one.
            dL_dY = dL_dX

    def _flatten(self, W, b):
        flattened = []
        for layer_idx in range(len(self._hidden_layers)):
            flatW = np.ravel(W[layer_idx])
            flatB = np.ravel(b[layer_idx])
            flattened += [flatW, flatB]

        flattened = np.hstack(flattened)
        return flattened

    def update(self, flatW):
        # Updates the parameters: receives a vector and unwraps it.
        start = 0
        for layer_idx in range(len(self._hidden_layers)):
            shape_W = self._weights[layer_idx].shape
            num_W = shape_W[0] * shape_W[1]
            params_W = flatW[start: start + num_W]
            params_W = params_W.reshape(shape_W)
            self._weights[layer_idx] = params_W

            start += num_W

            num_B = self._biases[layer_idx].shape[0]
            params_B = flatW[start: start + num_B]
            params_B = params_B.reshape((num_B,))
            self._biases[layer_idx] = params_B

            start += num_B

    @property
    def parameters(self):
        # Returns the vector of parameters.
        return self._flatten(self._weights, self._biases)

    @property
    def gradients(self):
        # Returns the vector of gradients.
        return self._flatten(self._weights_grads, self._biases_grads)


def optimize(nn):
    pass


def _draw_prediction(ax, xsA, xsB):
    zs = np.expand_dims(np.linspace(_min_x, _max_x, 1001), axis=1)

    nn = NeuralNetwork(hidden_layers=[3],
                       activation=activations.ReLU())

    batch_size = 10
    input = np.random.randn(batch_size, _input_dims)
    labels = np.random.randn(batch_size, _output_dims)
    output = nn.fw(input)

    residuals = labels - output
    loss = np.mean(residuals ** 2)

    dL_dResiduals = 2 * residuals
    dResiduals_dOutput = - np.ones((batch_size, 1))
    dL_dOuput = dL_dResiduals * dResiduals_dOutput

    nn.grad(dL_dOuput)

    gg = nn.gradients
    nn.update(gg)
    print(nn.gradients)

    # # Plot the mean.
    # ax.plot(zs, ys_hat, label='lambda={:.1f}'.format(lambda_reg))

    # plt.legend()


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Left-click for red dots,\n'
                 'Right-click for blue crosses\n'
                 'Enter to predict')

    ax.set_xlim([_min_x, _max_x])
    ax.set_ylim([_min_y, _max_y])

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')
    blue_crosses, = ax.plot([], [], linestyle='none', marker='x', color='b')

    drawer.PointDrawer(red_dots, blue_crosses, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
