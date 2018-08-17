"""
loss function measure how good our predictions are,
it can be used to adjust the parameters of our network
"""

import numpy as np

from kaynet.tensor import Tensor


class Loss:
    """
    Abstract base LOSS class
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE: mean square error
    but will implement total square error for now
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
