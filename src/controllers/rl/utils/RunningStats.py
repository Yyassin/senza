import numpy as np
from collections import deque

"""
Online statistics calculations (just mean right now).
"""


class RunningMeanStats:
    def __init__(self, n=10):
        """
        Creates a new running stats instance.

        :param n: The initial data buffer length.
        """
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        """
        Adds a new sample to the estimate.

        :param x: The sample to add.
        """
        self.stats.append(x)

    def get(self):
        """
        Returns the current mean estimate.
        """
        return np.mean(self.stats)
