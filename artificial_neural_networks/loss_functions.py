import numpy as np


class BinaryCrossEntropy:

    @staticmethod
    def fw(label, prediction):
        term1 = label * np.log(prediction)
        term2 = (1 - label) * np.log(1 - prediction)
        sample_loss = - (term1 + term2)
        loss = np.mean(sample_loss)
        return loss

    @staticmethod
    def grad(label, prediction):
        # Returns the gradients of the loss with respect to the predictions.
        return (prediction - label) / (prediction * (1 - prediction))
