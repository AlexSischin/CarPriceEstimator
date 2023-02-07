import numpy as np


class ZScoreNormalizer:
    def __init__(self, feature: np.ndarray):
        self._m = np.mean(feature)
        self._sd = np.std(feature)

    def scale(self, feature):
        return (feature - self._m) / self._sd
