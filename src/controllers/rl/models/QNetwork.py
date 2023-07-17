from controllers.rl.models.BaseNetwork import BaseNetwork
from controllers.rl.utils.nn import create_linear_network

"""
Fully connected FFNN to approximate a Q function.
"""


class QNetwork(BaseNetwork):
    def __init__(
        self, num_inputs, num_actions, hidden_units=[256, 256], initializer="xavier"
    ):
        """
        Creates a new Q network - Q(s, a)

        :param num_inputs: The state dimension.
        :param num_actions: The action dimension.
        :param hidden_units: The dimension of each hidden layer.
        :param initializer: Initializer to initialize network weights.
        """
        super(QNetwork, self).__init__()

        self.Q = create_linear_network(
            num_inputs + num_actions,
            1,  # G output
            hidden_units=hidden_units,
            initializer=initializer,
        )

    def forward(self, x):
        """
        Forwards the specified vector through the network.

        :param x: The vector to forward. This should be the state
        concatenated with the action.
        """
        q = self.Q(x)
        return q
