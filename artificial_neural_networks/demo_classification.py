import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

import artificial_neural_networks.activations as activations
import artificial_neural_networks.loss_functions as losses
import artificial_neural_networks.neural_network as model
import artificial_neural_networks.optimizers as optimizers
import common.metrics as metrics
import common.point_drawer as drawer

sns.set()
np.random.seed(2)

_min_x = - 5
_max_x = 5

_learning_rate = 0.001
_weight_decay = 0.0001
_num_epochs = 1000
_batch_size = -1

_input_dims = 2
_output_dims = 1

# The size of each hidden layer.
_hidden_layers = [100]

_binary_threshold = 0.5


def _color(ax, x, ys_hat):
    # Draw the contour regions.
    contourf_res = ax.contourf(x, x, ys_hat,
                               levels=10,
                               cmap=cm.get_cmap('RdBu_r'),
                               alpha=0.2)

    # Draw the boundary.
    contour_res = ax.contour(x, x, ys_hat,
                             levels=[0.5],
                             colors=('k',),
                             linestyles=('solid',))

    return contourf_res, contour_res


def _draw_prediction(ax, canvas, xsA, xsB):
    ys = np.expand_dims(np.r_[[1] * len(xsA), [0] * len(xsB)], axis=1)
    xs = np.r_[xsA, xsB]

    # Constants to normalize the data.
    mean_xs = np.mean(xs, axis=0)
    std_xs = np.std(xs, axis=0)
    xs = (xs - mean_xs) / std_xs

    # Create a grid.
    x = np.linspace(_min_x, _max_x, 101)
    x1, x2 = np.meshgrid(x, x)

    # Flatten the coordinates.
    zs = np.c_[np.ravel(x1), np.ravel(x2)]
    zs_normalized = (zs - mean_xs) / std_xs

    contourf_res, contour_res = None, None

    nn = model.NeuralNetwork(hidden_layers=_hidden_layers,
                             input_dims=_input_dims,
                             output_dims=_output_dims,
                             activation=activations.ReLU())
    nn.initialize()

    opt = optimizers.Adam()
    loss = losses.BinaryCrossEntropy()

    N = xs.shape[0]
    batch_size = _batch_size if _batch_size > 0 else len(xs)
    batches_per_epoch = int(np.floor(N / batch_size))

    training_losses = []
    weights_mean_norms = []
    weights_std_norms = []
    biases_mean_norms = []
    biases_std_norms = []
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

            prediction = nn.fw(batch_xs)

            # Use the logistic function to fit the outputs in [0, 1].
            logistic = activations.Logistic()
            logistic_prediction = logistic.fw(prediction)

            # Compute the training loss.
            training_loss = loss.fw(label=batch_ys,
                                    prediction=logistic_prediction)

            # Compute the accuracy, precision and recall.
            binary_prediction = logistic_prediction > _binary_threshold
            accuracy = metrics.compute_accuracy(label=batch_ys,
                                                prediction=binary_prediction)
            precision = metrics.compute_precision(label=batch_ys,
                                                  prediction=binary_prediction)
            recall = metrics.compute_recall(label=batch_ys,
                                            prediction=binary_prediction)

            # Print the epoch information.
            print('Epoch {}, batch {}'.format(epoch_idx, batch_idx))
            print(' Loss:      {:6.3f}'.format(training_loss))
            print(' Accuracy:  {:6.3f}'.format(accuracy))
            print(' Precision: {:6.3f}'.format(precision))
            print(' Recall:    {:6.3f}'.format(recall))

            dL_dLogOutput = loss.grad(label=batch_ys,
                                      prediction=logistic_prediction)
            dLogOutput_dOutput = logistic.grad(prediction)
            dL_dOuput = dL_dLogOutput * dLogOutput_dOutput

            # Compute the gradients.
            nn.backprop(dL_dOuput)
            grads = nn.gradients

            if np.linalg.norm(grads) < 1e-4:
                print(' Warning: vanishing gradient: {}'.format(np.linalg.norm(grads)))

            params = nn.parameters

            # Apply weight decay.
            grads += _weight_decay * params

            new_params = opt.update(parameters=params,
                                    gradients=grads,
                                    learning_rate=_learning_rate)
            nn.update(new_params)

            if batch_idx == 0:
                if epoch_idx % 10 == 0:
                    if contourf_res is not None:
                        for coll in contourf_res.collections:
                            coll.remove()

                    if contour_res is not None:
                        for coll in contour_res.collections:
                            coll.remove()

                    ys_hat = nn.fw(zs_normalized)
                    ys_hat = activations.Logistic.fw(ys_hat)
                    ys_hat = ys_hat.reshape(x1.shape)
                    contourf_res, contour_res = _color(ax, x, ys_hat)
                    canvas.draw()

                training_losses.append(training_loss)
                weights_mean_norms.append(np.mean(nn.weights))
                weights_std_norms.append(np.std(nn.weights))
                biases_mean_norms.append(np.mean(nn.biases))
                biases_std_norms.append(np.std(nn.biases))
                gradients_mean_norms.append(np.mean(grads))
                gradients_std_norms.append(np.std(grads))

    print('{} parameters'.format(nn.size))

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

    # Plot the weights spread.
    ax2.plot(range(len(weights_mean_norms)), weights_mean_norms,
             label='Weights',  linestyle=':', color='g')
    ax2.fill_between(range(len(weights_mean_norms)),
                     np.array(weights_mean_norms) - np.array(weights_std_norms),
                     np.array(weights_mean_norms) + np.array(weights_std_norms),
                     color='g', alpha=0.2)

    # Plot the biases spread.
    ax2.plot(range(len(biases_mean_norms)), biases_mean_norms,
             label='Biases', linestyle=':', color='b')
    ax2.fill_between(range(len(biases_mean_norms)),
                     np.array(biases_mean_norms) - np.array(biases_std_norms),
                     np.array(biases_mean_norms) + np.array(biases_std_norms),
                     color='b', alpha=0.2)

    # Plot the gradients spread.
    ax2.plot(range(len(gradients_mean_norms)), gradients_mean_norms,
             label='Gradients', linestyle='--', color='r')
    ax2.fill_between(range(len(gradients_mean_norms)),
                     np.array(gradients_mean_norms) - np.array(
                         gradients_std_norms),
                     np.array(gradients_mean_norms) + np.array(
                         gradients_std_norms),
                     color='r', alpha=0.2)

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
    ax.set_ylim([_min_x, _max_x])

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')
    blue_crosses, = ax.plot([], [], linestyle='none', marker='x', color='b')

    drawer.PointDrawer2D(red_dots, blue_crosses, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
