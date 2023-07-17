from controllers.rl.models.BaseNetwork import BaseNetwork
from controllers.rl.models.QNetwork import QNetwork

import torch

"""
Two Q networks for clipped double Q learning.
We learn two q functions and use the smaller of
the two Q values to reduce return overestimation.
"""


class TwinnedQNetwork(BaseNetwork):
    def __init__(
        self, num_inputs, num_actions, hidden_units=[256, 256], initializer="xavier"
    ):
        """
        Creates a new Twinned Q network

        :param num_inputs: The state dimension for each network.
        :param num_actions: The action dimension for each network.
        :param hidden_units: The dimension of each hidden layer for each network.
        :param initializer: Initializer to initialize network weights for each network.
        """
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(num_inputs, num_actions, hidden_units, initializer)
        self.Q2 = QNetwork(num_inputs, num_actions, hidden_units, initializer)

    def forward(self, states, actions):
        """
        Forwards and returns the Q estimates from
        both networks according to the supplied state-action batch.

        :param states: Batch of states.
        :param actions: Batch of actions.
        """
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2
