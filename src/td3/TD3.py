import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
import math
import os
import shutil

# https://github.com/Ullar-Kask/TD3-PER/blob/master/Pytorch/src/td3_agent.py
# https://github.com/XinJingHao/TD3-Pytorch/blob/main/main.py

"""
checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}
save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
"""


def save_ckp(state, is_best, checkpoint_dir, best_model_dir, name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    f_path = os.path.join(checkpoint_dir, name)
    print(f_path, checkpoint_dir)
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, f"best_{name}")
        shutil.copyfile(f_path, best_fpath)


"""
model = MyModel(*args, **kwargs)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
ckp_path = "path/to/checkpoint/checkpoint.pt"
model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
"""


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # 没有先提取特征
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_Agent(object):
    def __init__(
        self,
        env_with_dw,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        net_width=128,
        a_lr=1e-4,
        c_lr=1e-4,
        batch_size=256,
        policy_delay_freq=1,
    ):
        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(state_dim, action_dim, net_width).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_dw = env_with_dw
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005
        self.batch_size = batch_size
        self.delay_counter = -1
        self.delay_freq = policy_delay_freq

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def mse(self, expected, targets, is_weights):
        """Custom loss function that takes into account the importance-sampling weights."""
        return torch.mean((expected - targets) ** 2 * is_weights)

    def train(self, replay_buffer):
        self.delay_counter += 1
        with torch.no_grad():
            batch, weights, tree_idxs = replay_buffer.sample(self.batch_size)
            s, a, r, s_prime, dw_mask = batch

            # s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            smoothed_target_a = (
                self.actor_target(s_prime) + noise  # Noisy on target action
            ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
        target_Q = torch.minimum(target_Q1, target_Q2)

        """Avoid impacts caused by reaching max episode steps"""
        if self.env_with_dw:
            target_Q = r + (1 - dw_mask) * self.gamma * target_Q.squeeze(
                dim=1
            )  # dw: die or win
        else:
            target_Q = r + self.gamma * target_Q.squeeze(dim=1)

        target_Q = target_Q.unsqueeze(dim=1)

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # Compute critic loss
        errors = (
            np.abs((current_Q1 - target_Q).detach().cpu().numpy())
            + np.abs((current_Q2 - target_Q).detach().cpu().numpy())
        ) / 2
        q_loss = self.mse(current_Q1, target_Q, weights) + self.mse(
            current_Q2, target_Q, weights
        )

        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # Update priorities in replay buffer
        replay_buffer.update_priorities(tree_idxs, errors)

        if self.delay_counter == self.delay_freq:
            # Update Actor
            a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.q_critic.parameters(), self.q_critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.delay_counter = -1

    def save(self, EnvName, episode, best=False):
        print("saving models")
        actor_checkpoint = {
            "epoch": 1,
            "state_dict": self.actor.state_dict(),
            "optimizer": self.actor_optimizer.state_dict(),
        }
        critic_checkpoint = {
            "epoch": 1,
            "state_dict": self.q_critic.state_dict(),
            "optimizer": self.q_critic_optimizer.state_dict(),
        }
        save_ckp(
            actor_checkpoint,
            best,
            os.path.join("model", EnvName),
            os.path.join("best", EnvName),
            "actor.pth",
        )
        save_ckp(
            critic_checkpoint,
            best,
            os.path.join("model", EnvName),
            os.path.join("best", EnvName),
            "critic.pth",
        )

    def load(self, EnvName):
        print("loading models from checkpoint")
        absolute_path = os.path.dirname(__file__)
        self.actor, self.actor_optimizer, _ = load_ckp(
            os.path.join(absolute_path, "model/{}_actor.pth".format(EnvName)),
            self.actor,
            self.actor_optimizer,
        )
        self.q_critic, self.q_critic_optimizer, _ = load_ckp(
            os.path.join(absolute_path, "model/{}_q_critic.pth".format(EnvName)),
            self.q_critic,
            self.q_critic_optimizer,
        )


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device),
        )
