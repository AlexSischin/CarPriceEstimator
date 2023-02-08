from dataclasses import dataclass
from numbers import Number

import numpy as np


def isfinite(*args):
    for a in args:
        if not np.isfinite(a).all():
            return False
    return True


@dataclass(frozen=True)
class LearningHistPoint:
    cost: float
    dj_dw: np.ndarray
    dj_db: float
    w: np.ndarray
    b: float


class LinearRegressor:
    def __init__(
            self,
            w: np.ndarray | Number,
            b: float
    ):
        if not isfinite(w, b):
            raise ValueError('Infinite and NaN values are not allowed')
        self._w = w
        self._b = b

    # Parameters:
    #   x - 2D array with shape (m, n), where m - number of training examples, n - number of features
    #   y - 1D array of targets with shape of m
    #   a - learning rate
    #   iterations - number of iterations over the whole training set
    #   debug - if True, function returns learning history. If False - converging cost
    # Returns:
    #   float - convergence cost, if debug set to False.
    #   list[LearnHistPoint] - the history of learning, if debug set to True.
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            a: float,
            iterations: int,
            debug=False
            ) -> list[LearningHistPoint] | float:
        if not isfinite(x, y, a, iterations):
            raise ValueError('Infinite and NaN values are not allowed')
        if debug:
            return self._debugging_fit(x, y, a, iterations)
        else:
            return self._fit(x, y, a, iterations)

    # Parameters:
    #   x - 2D array with shape (m, n), where m - number of training examples, n - number of features
    # Returns:
    #   np.ndarray - 1D array of estimates
    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self._w + self._b

    # Parameters:
    #   x - 2D array with shape (m, n), where m - number of training examples, n - number of features
    #   y - 1D array of targets with shape of m
    # Returns:
    #   float - cost of model for x.
    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self._loss(x, y))

    def _loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (self.predict(x) - y) ** 2 / 2

    def _gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        error = self.predict(x) - y
        dj_dw = (error @ x) / x.shape[0]
        dj_db = np.mean(error)
        return dj_dw, dj_db

    def _fit(self, x, y, a, iterations) -> float:
        for i in range(iterations):
            dj_dw, dj_db = self._gradient(x, y)
            self._w -= a * dj_dw
            self._b -= a * dj_db
        return self.cost(x, y)

    def _debugging_fit(self, x, y, a, iterations) -> list[LearningHistPoint]:
        hist = []
        for i in range(iterations):
            dj_dw, dj_db = self._gradient(x, y)
            self._w -= a * dj_dw
            self._b -= a * dj_db
            hist.append(LearningHistPoint(cost=self.cost(x, y), dj_dw=dj_dw, dj_db=dj_db, w=self._w, b=self._b))
        return hist
