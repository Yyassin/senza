import os
import shutil
import torch

"""
General utilities to save network 
training checkpoints for resuming later.
"""


def save_ckp(state, is_best, checkpoint_dir, best_model_dir, name):
    """
    Saves a training checkpoint.

    :param state: The network state (weights, optimizer state and epoch as a dict).
    :param is_best: True if this is the best network
    so far (lowest loss usually), False otherwise.
    :param checkpoints_dir: The checkpoint directory.
    :param best_model_dir: The directory where the current
    best model should be saved.
    :param name: The filename to save.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    f_path = os.path.join(checkpoint_dir, name)
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, f"best_{name}")
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Loads a training checkpoint.

    :param checkpoint_fpath: Filepath of the checkpoint to load.
    :param model: The model to load the saved weights into.
    :param optimizer: The model's optimizer to load the saved state into.
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]
