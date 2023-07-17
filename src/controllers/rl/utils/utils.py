"""
General torch utility functions
"""


def to_batch(state, action, reward, next_state, done, device):
    """
    Casts the provided trajectory into a torch tensor batch.

    :param state: The start state.
    :param action: The action taken.
    :param reward: The reward received.
    :param next_state: The resulting state.
    :param done: 1 if terminated, 0 otherwise.
    :param device: The model device.
    """
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    # Cast to single row with k columns
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    """
    Backpropagates the loss across the supplied network.

    :param optim: The network optimizer.
    :param network: The model being trained.
    :param loss: The loss to backpropagate.
    :param grad_clip: True if gradient clipping, False otherwise.
    :param retain_graph: True if the backprop graph should be retained
    and false otherwise (this lets us call loss.backward multiple times).
    """
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    """
    Soft updates each parameter in the target network towards the
    source according to tau (memory factor).

    :param target: The network to update.
    :param source: The network to update towards.
    :param tau: The memory factor (how much of target is remembered).
    """
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    """
    Loads the source parameters into target.

    :param target: The network to load into.
    :param source: The network containing the parameters to copy.
    """
    target.load_state_dict(source.state_dict())


def grad_false(network):
    """
    Freezes the supplied network's parameters
    by disabling gradient calculations.
    """
    for param in network.parameters():
        param.requires_grad = False
