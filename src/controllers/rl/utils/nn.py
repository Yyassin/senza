from torch.distributions import Normal
import torch.nn as nn
import torch

"""
Common network creation utils.
"""

"""
Weight initializer map
"""
str_to_initializer = {
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "eye": nn.init.eye_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "he": nn.init.kaiming_normal_,
    "orthogonal": nn.init.orthogonal_,
}

"""
Activations map
"""
str_to_activation = {
    "elu": nn.ELU(),
    "hardshrink": nn.Hardshrink(),
    "hardtanh": nn.Hardtanh(),
    "leakyrelu": nn.LeakyReLU(),
    "logsigmoid": nn.LogSigmoid(),
    "prelu": nn.PReLU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "rrelu": nn.RReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "logsoftmax": nn.LogSoftmax(),
    "softshrink": nn.Softshrink(),
    "softsign": nn.Softsign(),
    "tanh": nn.Tanh(),
    "tanhshrink": nn.Tanhshrink(),
    "softmin": nn.Softmin(),
    "softmax": nn.Softmax(dim=1),
    "none": None,
}


def initialize_weights(initializer):
    """
    Returns an function that will initialize a
    network's parameters with the supplied initializer.

    :param initializer: The initializer to use.
    """

    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    return initialize


def create_linear_network(
    input_dim,
    output_dim,
    hidden_units=[],
    hidden_activation="relu",
    output_activation=None,
    initializer="xavier_uniform",
):
    """
    Constructs a feed forward neural net with the
    specified input dimension, output dimension and
    hidden dimensions wth the supplied activation. If specified,
    an output activation is also applied. The network's
    weights are initialized with the supplied initializer.

    :param input_dim: The network's input dimension.
    :param output_dim: The network's output dimension.
    :param hidden_units: The hidden layer dimensions.
    :param hidden_activation: Activation function to
    use for the hidden layers.
    :param output_activation: Activation function to use
    for the output.
    :param initializer: Initializer to initialize network weights.
    """
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(str_to_activation[hidden_activation])
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(str_to_activation[output_activation])

    return nn.Sequential(*model).apply(
        initialize_weights(str_to_initializer[initializer])
    )
