import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import artificial_neural_networks.activations as activations
import artificial_neural_networks.neural_network as model
import artificial_neural_networks.optimizers as optimizers
import common.point_drawer as drawer

sns.set()
np.random.seed(2)

_min_x = 0.5
_max_x = 1

_min_y = - 2
_max_y = 2


_learning_rate = 0.1
_weight_decay = 0.0
_num_epochs = 1000
_batch_size = -1

_input_dims = 1
_output_dims = 1

# The size of each hidden layer.
_hidden_layers = [10, 10, 10, 10, 10, 10]


def _draw_prediction(ax, xs, ys):
    nn = model.NeuralNetwork(hidden_layers=_hidden_layers,
                             input_dims=_input_dims,
                             output_dims=_output_dims,
                             activation=activations.LeakyReLU())
    nn.initialize()

    sgd = optimizers.SGD()

    # Constants to normalize the data.
    center_x = (_min_x + _max_x) / 2
    spread_x = (_max_x - _min_y) / 2

    N = xs.shape[0]
    batch_size = _batch_size if _batch_size > 0 else len(xs)
    batches_per_epoch = int(np.floor(N / batch_size))
    training_losses = []
    parameters_mean_norms = []
    parameters_std_norms = []
    gradients_mean_norms = []
    gradients_std_norms = []
    for epoch_idx in range(_num_epochs):
        all_indices = list(range(N))
        np.random.shuffle(all_indices)
        for batch_idx in range(batches_per_epoch):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_indices = all_indices[batch_start: batch_end]
            batch_xs = xs[batch_indices]
            batch_ys = ys[batch_indices]

            # Normalize the data to be between -1 and 1.
            batch_xs = (batch_xs - center_x) / spread_x

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
            nn.backprop(dL_dOuput)
            grads = nn.gradients

            if np.linalg.norm(grads) < 1e-4:
                print(' Warning: vanishing gradient: {}'.format(np.linalg.norm(grads)))

            params = nn.parameters

            # Apply weight decay.
            grads += _weight_decay * params

            new_params = sgd.update(parameters=params,
                                    gradients=grads,
                                    learning_rate=_learning_rate)
            nn.update(new_params)

            if batch_idx == 0:
                training_losses.append(training_loss)
                parameters_mean_norms.append(np.mean(params))
                parameters_std_norms.append(np.std(params))
                gradients_mean_norms.append(np.mean(grads))
                gradients_std_norms.append(np.std(grads))

    zs = np.expand_dims(np.linspace(_min_x, _max_x, 1001), axis=1)
    zs_normalized = (zs - center_x) / spread_x
    ys_hat = nn.fw(zs_normalized)

    print('{} parameters'.format(nn.size))

    # Plot the mean.
    ax.plot(zs, ys_hat)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training loss')
    ax1.tick_params(axis='y')

    # Instantiate a second axes that shares the same x-axis.
    ax2 = ax1.twinx()
    ax2.set_ylabel('Norms')
    ax2.tick_params(axis='y')

    # Plot the training loss.
    ax1.plot(range(len(training_losses)), training_losses,
             label='Training loss', color='k')

    # Plot the parameters spread.
    ax2.plot(range(len(parameters_mean_norms)), parameters_mean_norms,
             label='Parameters',  linestyle=':', color='g')
    ax2.fill_between(range(len(parameters_mean_norms)),
                     np.array(parameters_mean_norms) - np.array(parameters_std_norms),
                     np.array(parameters_mean_norms) + np.array(parameters_std_norms),
                     color='g', alpha=0.4)

    # Plot the gradients spread.
    ax2.plot(range(len(gradients_mean_norms)), gradients_mean_norms,
             label='Gradients', linestyle='--', color='r')
    ax2.fill_between(range(len(gradients_mean_norms)),
                     np.array(gradients_mean_norms) - np.array(
                         gradients_std_norms),
                     np.array(gradients_mean_norms) + np.array(
                         gradients_std_norms),
                     color='r', alpha=0.4)

    fig.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()


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
