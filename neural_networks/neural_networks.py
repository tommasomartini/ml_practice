import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

np.random.seed(2)


def get_underlying_model():
    noise_mean = 0.0
    noise_std = 0.03

    clean_f = lambda x: np.sqrt(x[:, 0]) - x[:, 1] ** 3 + 0.3

    def _f(x):
        noise = noise_std * np.random.randn(len(x),) + noise_mean
        return clean_f(x) + noise

    return _f


def main():

    # Generate the dataset.
    num_samples = 1000

    underlying_model = get_underlying_model()
    xs = np.random.rand(num_samples, 2)
    ys = underlying_model(xs)

    eval_xs = np.random.rand(num_samples, 2)
    eval_ys = underlying_model(eval_xs)

    # Train the model.
    learning_rate = 0.01
    num_epochs = 1000

    # Neural Network with 3 hidden units.
    num_hidden_units_layer0 = 3
    num_hidden_units_layer1 = 4
    num_input_features = 2
    num_outputs = 1

    def _wrap_layers(w0, b0, w1, b1, wOut, bOut):
        return np.r_[
            np.ravel(w0),
            b0,
            np.ravel(w1),
            b1,
            np.ravel(wOut),
            bOut
        ]

    def _unwrap_layers(params):
        k1 = num_input_features * num_hidden_units_layer0
        k2 = k1 + num_hidden_units_layer0

        k3 = k2 + num_hidden_units_layer0 * num_hidden_units_layer1
        k4 = k3 + num_hidden_units_layer1

        k5 = k4 + num_hidden_units_layer1 * num_outputs
        k6 = k5 + num_outputs

        assert len(params) == k6

        w0 = params[:k1].reshape(num_input_features, num_hidden_units_layer0)
        b0 = params[k1: k2]

        w1 = params[k2: k3].reshape(num_hidden_units_layer0, num_hidden_units_layer1)
        b1 = params[k3: k4]

        wOut = params[k4: k5]
        bOut = params[k5: k6]

        layers = (w0, b0, w1, b1, wOut, bOut)
        return layers

    def _create_model_from_params(params):
        w0, b0, w1, b1, wOut, bOut = _unwrap_layers(params)

        def _predict(x):
            f0a = x @ w0 + b0
            f0b = activation(f0a)

            f1a = f0b @ w1 + b1
            f1b = activation(f1a)

            outa = f1b @ wOut + bOut
            predicted = activation(outa)

            feature_maps = (f0a, f0b, f1a, f1b, outa)

            return predicted, feature_maps

        return _predict

    # Use Xavier initialization.

    # From input to inner layer.
    layer0_weights = (1 / np.sqrt(num_input_features)) * np.random.rand(num_input_features, num_hidden_units_layer0)
    layer0_biases = np.random.rand(num_hidden_units_layer0)

    layer1_weights = (1 / np.sqrt(num_hidden_units_layer0)) * np.random.rand(num_hidden_units_layer0,
                                                                             num_hidden_units_layer1)
    layer1_biases = np.random.rand(num_hidden_units_layer1)

    # From layer to output, which is a scalar.
    output_weights = (1 / np.sqrt(num_hidden_units_layer1)) * np.random.rand(num_hidden_units_layer1 * num_outputs)
    output_biases = np.random.rand(num_outputs)

    def _relu(x):
        return np.clip(x, 0, None)

    def _relu_grad(x):
        """Computes d(ReLU(x))/dx
        """
        return (x > 0).astype(float)

    def _sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    def _sigmoid_grad(x):
        """Computes d(sigmoid(x))/dx
        """
        return np.exp(x) / (1 + np.exp(x)) ** 2

    activation = _relu
    activation_grad = _relu_grad

    training_losses = []
    evaluation_losses = []
    params = [_wrap_layers(layer0_weights,
                           layer0_biases,
                           layer1_weights,
                           layer1_biases,
                           output_weights,
                           output_biases)]

    prev_grads = np.zeros(params[0].shape)

    lambd = 0.01    # for weight decay
    beta = 0.9      # for momentum
    batch_size = 200
    batches_per_epoch = int(np.floor(num_samples / batch_size))
    for epoch_idx in range(num_epochs):
        all_indices = list(range(num_samples))
        np.random.shuffle(all_indices)
        for batch_idx in range(batches_per_epoch):
            batch_indices = all_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_xs = xs[batch_indices]
            batch_ys = ys[batch_indices]

            # Predict the output.
            # fnx means: feature map of layer n. "a" is before the activation, "b" is after.

            predicted_ys, feature_maps = _create_model_from_params(params[-1])(batch_xs)
            f0a, f0b, f1a, f1b, outa = feature_maps

            # Compute the loss (just to print it).
            residuals = batch_ys - predicted_ys
            training_loss = np.sum(residuals ** 2) / batch_size

            if batch_idx + 1 == batches_per_epoch:
                # Test on the evaluation set.
                predicted_eval_ys, _ = _create_model_from_params(params[-1])(eval_xs)
                evaluation_loss = np.sum((eval_ys - predicted_eval_ys) ** 2) / len(eval_ys)
                evaluation_losses.append(evaluation_loss)
                training_losses.append(training_loss)
                print('[Epoch {:3d}] Training loss   {}'.format(epoch_idx, training_loss))
                print('            Evaluation loss {}'.format(evaluation_loss))

            # Backpropagation.
            # The gradient of the loss reduces to a summation over all the terms.
            # All the variables markes as "partial" are what is inside the summation.

            partial_dLoss_dResidual = 2 * residuals / batch_size

            ######################
            # Output layer
            partial_dLoss_dOutput = - partial_dLoss_dResidual
            partial_dLoss_dOuta = partial_dLoss_dOutput * activation_grad(outa)     # (1000, 1) x (1000, 1) = (1000, 1)
            partial_dLoss_df1b = np.expand_dims(partial_dLoss_dOuta, axis=1) * output_weights   # (1000, 1) x (L1,) = (1000, L1)
            partial_dLoss_dWeightsOutput = np.expand_dims(partial_dLoss_dOuta, axis=1) * f1b    # (1000, 1) x (1000, 3) = (1000, 3)
            partial_dLoss_dBiasesOutput = np.expand_dims(partial_dLoss_dOuta, axis=1)    # (1000, 1)

            dLoss_dWeightsOutput = np.sum(partial_dLoss_dWeightsOutput, axis=0)
            dLoss_dBiasesOutput = np.sum(partial_dLoss_dBiasesOutput, axis=0)

            ######################
            # Layer 1

            partial_dLoss_df1a = partial_dLoss_df1b * activation_grad(f1a)  # (1000, 3)
            partial_dLoss_df0b = np.sum(np.expand_dims(partial_dLoss_df1a, axis=1) * np.expand_dims(layer1_weights, axis=0), axis=2)   # (1000, L0, L1)
            partial_dLoss_dWeights1 = np.expand_dims(partial_dLoss_df1a, 1) * np.expand_dims(f0b, 2)
            partial_dLoss_dBiases1 = partial_dLoss_df1a

            dLoss_dWeights1 = np.sum(partial_dLoss_dWeights1, axis=0)
            dLoss_dBiases1 = np.sum(partial_dLoss_dBiases1, axis=0)

            ######################
            # Layer 0

            partial_dLoss_df0a = partial_dLoss_df0b * activation_grad(f0a)
            partial_dLoss_dWeights0 = np.expand_dims(partial_dLoss_df0a, 1) * np.expand_dims(batch_xs, 2)
            partial_dLoss_dBiases0 = partial_dLoss_df0a

            dLoss_dWeights0 = np.sum(partial_dLoss_dWeights0, axis=0)
            dLoss_dBiases0 = np.sum(partial_dLoss_dBiases0, axis=0)

            # GD + Momentum.
            grads = _wrap_layers(dLoss_dWeights0,
                                 dLoss_dBiases0,
                                 dLoss_dWeights1,
                                 dLoss_dBiases1,
                                 dLoss_dWeightsOutput,
                                 dLoss_dBiasesOutput)

            # Weight decay component.
            grads += lambd * params[-1]

            grads = beta * prev_grads + (1 - beta) * grads
            new_params = params[-1] - learning_rate * grads
            params.append(new_params)
            prev_grads = grads

    plt.figure()
    ax = plt.axes(projection=Axes3D.name)

    # Plot the training set.
    ax.scatter(xs[:, 0], xs[:, 1], ys, c=ys, cmap='viridis', linewidth=0.5)

    _model = _create_model_from_params(params[-1])

    linsp = np.linspace(0, 1, 10)
    grid_x, grid_y = np.meshgrid(linsp, linsp)
    zs, _ = _model(np.c_[np.ravel(grid_x), np.ravel(grid_y)])
    zs = zs.reshape((10, 10))
    ax.plot_surface(grid_x, grid_y, zs, linewidth=0, antialiased=False, alpha=0.5)

    # Plot the loss.
    plt.figure()
    plt.plot(range(len(training_losses)), training_losses, label='Train')
    plt.plot(range(len(evaluation_losses)), evaluation_losses, label='Eval')
    plt.legend()
    plt.title('Loss')

    # Plot the loss landscape.
    plt.figure()
    ax = plt.axes(projection=Axes3D.name)

    params = np.array(params)
    pca = PCA(n_components=2)
    reduced_params = pca.fit_transform(params)

    parameter_span = np.linspace(-2, 2, 21)
    grid_w1, grid_w2 = np.meshgrid(parameter_span, parameter_span)
    losses = []
    for p in np.c_[np.ravel(grid_w1), np.ravel(grid_w2)]:
        exp_p = pca.inverse_transform(p)

        if True:
            predicted, _ = _create_model_from_params(exp_p)(xs)
            l = np.sum((ys - predicted) ** 2) / len(xs)
        else:
            predicted, _ = _create_model_from_params(exp_p)(eval_xs)
            l = np.sum((eval_ys - predicted) ** 2) / len(eval_xs)

        losses.append(l)

    loss = np.array(losses).reshape(grid_w1.shape)
    ax.plot_surface(grid_w1, grid_w2, loss, cmap='viridis', linewidth=0, antialiased=False, alpha=0.5)

    loss_path = []
    for p in params:
        if True and False:
            # Convert parameters to 2-dims and convert back
            p2 = pca.transform([p])
            p = pca.inverse_transform(p2)[0]

        # Use parameters in original dimension.
        if True:
            predicted, _ = _create_model_from_params(p)(xs)
            l = np.sum((ys - predicted) ** 2) / len(xs)
        else:
            predicted, _ = _create_model_from_params(p)(eval_xs)
            l = np.sum((eval_ys - predicted) ** 2) / len(eval_xs)

        loss_path.append(l)

    ax.plot(reduced_params[:, 0], reduced_params[:, 1], loss_path)
    ax.scatter(reduced_params[:, 0], reduced_params[:, 1], loss_path, c=loss_path)
    ax.scatter(reduced_params[0, 0], reduced_params[0, 1], loss_path[0], marker='*')

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
