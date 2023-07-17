from controllers.rl.models.BaseNetwork import BaseNetwork
from controllers.rl.utils.nn import create_linear_network

from torch.distributions import Normal
import torch

"""
Fully connected FFNN to approximate
a normal policy, pi.
"""


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(
        self, num_inputs, num_actions, hidden_units=[256, 256], initializer="xavier"
    ):
        """
        Creates a new policy network.

        :param num_inputs: The state dimension.
        :param num_actions: The action dimension.
        :param hidden_units: The dimension of each hidden layer.
        :param initializer: Initializer to initialize network weights.
        """
        super(GaussianPolicy, self).__init__()

        self.policy = create_linear_network(
            num_inputs,
            num_actions * 2,
            hidden_units=hidden_units,
            initializer=initializer,
        )

    def forward(self, states):
        """
        Forwards the supplied state batch through the network and
        returns an action mean, mu, and associated log standard deviation
        for sampling an action.

        :param states: The batch of states to forward.
        """
        mean, log_std = torch.chunk(self.policy(states), 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        """
        Returns a batch of actions corresponding to the
        supplied batch of states, along with the
        associated entropies and squashed action means.

        a ~ pi(s)

        :param states: The batch of states to forward.
        """
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
