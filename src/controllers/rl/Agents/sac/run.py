from controllers.rl.Agents.sac.SACAgent import SacAgent
from drone.DroneEnv import DroneEnv

from datetime import datetime
import torch
import os


def run(eval=False, render=False, load=False):
    """
    Runs the SAC agent.

    :param eval: True if evaluation, False otherwise.
    :param render: True if the environment should be rendered, False otherwise.
    :param load: True if models should be loaded, False otherwise.
    """

    torch.autograd.set_detect_anomaly(False)

    env_id = "DroneEnv"
    cuda = False  # This is a mistake
    seed = 0

    # You can define configs in the external json or yaml file.
    configs = {
        "num_steps": 3000000,
        "batch_size": 256,
        "lr": 0.0003,
        "hidden_units": [256, 256],
        "memory_size": 1e6,
        "gamma": 0.99,
        "tau": 0.005,
        "entropy_tuning": True,
        "ent_coef": 0.2,  # It's ignored when entropy_tuning=True.
        "multi_step": 1,
        "per": False,  # prioritized experience replay
        "alpha": 0.6,  # It's ignored when per=False.
        "beta": 0.4,  # It's ignored when per=False.
        "beta_annealing": 0.0001,  # It's ignored when per=False.
        "grad_clip": None,
        "updates_per_step": 1,
        "start_steps": 10000,
        "log_interval": 10,
        "target_update_interval": 1,
        "eval_interval": 10000,
        "cuda": cuda,
        "seed": seed,
    }

    env = DroneEnv(render=render)

    log_dir = os.path.join(
        "logs", env_id, f'sac-seed{seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    )

    agent = SacAgent(load=load, env=env, log_dir=log_dir, **configs)
    if not eval:
        agent.run()
    else:
        agent.evaluate()
    agent.save_models(True)
