import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import artificial_neural_networks.activations as activations
import artificial_neural_networks.neural_network as model
import common.point_drawer as drawer

sns.set()
np.random.seed(2)

_min_x = -5
_max_x = 5

_min_y = 0
_max_y = 2


_learning_rate = 0.01
_weight_decay = 0.1
_num_epochs = 1000
_batch_size = -1

_input_dims = 1     # input is 2 dimensional
_output_dims = 1

# The size of each hidden layer.
_hidden_layers = [10, 10, 10]


def _draw_prediction(ax, xs, ys):
    nn = model.NeuralNetwork(hidden_layers=_hidden_layers,
                             _input_dims=_input_dims,
                             _output_dims=_output_dims,
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
