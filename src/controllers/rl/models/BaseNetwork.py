import torch.nn as nn
import torch

"""
    Base network class
"""


class BaseNetwork(nn.Module):
    def save(self, path):
        """
        Saves the network weights to the specified path.

        :param path: Where the weights should be saved.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Loads the network weights from the specified path.

        :param path: Where the weights should be loaded from.
        """
        self.load_state_dict(torch.load(path))
