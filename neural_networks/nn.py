import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import common.point_drawer as drawer
import neural_networks.activations as activations

sns.set()
np.random.seed(2)

_min_x = -5
_max_x = 5

_min_y = 0
_max_y = 2


_learning_rate = 0.001
_weight_decay = 0.1
_num_epochs = 1000
_batch_size = -1

_input_dims = 1     # input is 2 dimensional
_output_dims = 1

# The size of each hidden layer.
_hidden_layers = [10, 10, 10]


class NeuralNetwork:

    def __init__(self, hidden_layers, activation):
        # Add the output layer.
        self._hidden_layers = hidden_layers + [_output_dims]
        self._activation = activation

        self._reset_state()

        self._weights = []
        self._biases = []

        num_inputs = _input_dims
        for layer_idx, num_neurons in enumerate(self._hidden_layers):
            layer_weights = np.zeros((num_inputs, num_neurons))
            layer_biases = np.zeros((num_neurons,))

            self._weights.append(layer_weights)
            self._biases.append(layer_biases)

            num_inputs = num_neurons

    def initialize(self):
        # Initialize the network using Xavier initialization.
        for layer_idx in range(len(self._hidden_layers)):
            shape_W = self._weights[layer_idx].shape
            num_inputs, num_outputs = shape_W
            params_W = (1 / np.sqrt(num_inputs)) * np.random.randn(*shape_W)
            self._weights[layer_idx] = params_W

            params_B = (1 / np.sqrt(num_inputs)) * np.random.randn(num_outputs)
            self._biases[layer_idx] = params_B

    def _reset_state(self):
        # Store the gradients of the parameters.
        self._weights_grads = [None] * len(self._hidden_layers)
        self._biases_grads = [None] * len(self._hidden_layers)

        # Store the intermediate features.
        self._features_before_act = []
        self._features_after_act = []

    def fw(self, x):
        # Performs a forward pass.
        # x has shape (<batch size>, D)
        self._reset_state()
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


def _draw_prediction(ax, xs, ys):
    nn = NeuralNetwork(hidden_layers=_hidden_layers,
                       activation=activations.ReLU())
    nn.initialize()

    N = xs.shape[0]
    batch_size = _batch_size if _batch_size > 0 else len(xs)
    batches_per_epoch = int(np.floor(N / batch_size))
    for epoch_idx in range(_num_epochs):
        all_indices = list(range(N))
        np.random.shuffle(all_indices)
        for batch_idx in range(batches_per_epoch):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_indices = all_indices[batch_start: batch_end]
            batch_xs = xs[batch_indices]
            batch_ys = ys[batch_indices]

            prediction = nn.fw(batch_xs)
            residuals = batch_ys - prediction

            # Mean Square Error.
            training_loss = np.mean(residuals ** 2)
            print('Epoch {}, batch {}, loss: {:.3f}'.format(epoch_idx,
                                                            batch_idx,
                                                            training_loss))

            dL_dResiduals = 2 * residuals
            dResiduals_dOutput = - np.ones((batch_size, 1))
            dL_dOuput = dL_dResiduals * dResiduals_dOutput

            # Compute the gradients.
            nn.grad(dL_dOuput)
            grads = nn.gradients

            if np.linalg.norm(grads) < 1e-4:
                print(' Warning: vanishing gradient')

            params = nn.parameters

            # Apply weight decay.
            grads += _weight_decay * params

            new_params = params - _learning_rate * grads
            nn.update(new_params)

    zs = np.expand_dims(np.linspace(_min_x, _max_x, 1001), axis=1)
    ys_hat = nn.fw(zs)

    # Plot the mean.
    ax.plot(zs, ys_hat)

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
