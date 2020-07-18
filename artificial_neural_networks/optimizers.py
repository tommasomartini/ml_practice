class SGD:

    def update(self, parameters, gradients, learning_rate):
        new_parameters = parameters - learning_rate * gradients
        return new_parameters
