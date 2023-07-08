from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
import gym
import random
from gym import spaces
from datetime import datetime
import shutil
from screen import Screen
from drone import Drone
import pygame

# https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
# https://github.com/toshikwa/soft-actor-critic.pytorch/issues?q=is%3Aissue+is%3Aclosed

torch.autograd.set_detect_anomaly(False)

# utils


def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:
    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


# Models

import torch
import torch.nn as nn
from torch.distributions import Normal

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


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):
    def __init__(
        self, num_inputs, num_actions, hidden_units=[256, 256], initializer="xavier"
    ):
        super(QNetwork, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.Q = create_linear_network(
            num_inputs + num_actions,
            1,
            hidden_units=hidden_units,
            initializer=initializer,
        )

    def forward(self, x):
        q = self.Q(x)
        return q


class TwinnedQNetwork(BaseNetwork):
    def __init__(
        self, num_inputs, num_actions, hidden_units=[256, 256], initializer="xavier"
    ):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(num_inputs, num_actions, hidden_units, initializer)
        self.Q2 = QNetwork(num_inputs, num_actions, hidden_units, initializer)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(
        self, num_inputs, num_actions, hidden_units=[256, 256], initializer="xavier"
    ):
        super(GaussianPolicy, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.policy = create_linear_network(
            num_inputs,
            num_actions * 2,
            hidden_units=hidden_units,
            initializer=initializer,
        )

    def forward(self, states):
        mean, log_std = torch.chunk(self.policy(states), 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
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


# Memory


class Memory:
    def __init__(self, capacity, state_shape, action_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(self, state, action, reward, next_state, done, episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=self.state_type)
        next_state = np.array(next_state, dtype=self.state_type)

        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.uint8)
            next_states = self.next_states[indices].astype(np.uint8)
            states = torch.ByteTensor(states).to(self.device).float() / 255.0
            next_states = torch.ByteTensor(next_states).to(self.device).float() / 255.0
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
        )

    def load(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(slice(self._p, self._p + num_data), batch, slice(0, num_data))
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            self._insert(slice(self._p, self.capacity), batch, slice(0, mid_index))
            self._insert(slice(0, end_index), batch, slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        states, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]


class MultiStepBuff:
    keys = ["state", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {key: deque(maxlen=self.maxlen) for key in self.keys}

    def append(self, state, action, reward):
        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([r * (gamma**i) for i, r in enumerate(self.memory["reward"])])

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f"There is no key {key} in MultiStepBuff.")
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory["state"])


class MultiStepMemory(Memory):
    def __init__(
        self, capacity, state_shape, action_shape, device, gamma=0.99, multi_step=3
    ):
        super(MultiStepMemory, self).__init__(
            capacity, state_shape, action_shape, device
        )

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done, episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(state, action, reward, next_state, done)


class PrioritizedMemory(MultiStepMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        action_shape,
        device,
        gamma=0.99,
        multi_step=3,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.001,
        epsilon=1e-4,
    ):
        super(PrioritizedMemory, self).__init__(
            capacity, state_shape, action_shape, device, gamma, multi_step
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(
        self, state, action, reward, next_state, done, error, episode_done=False
    ):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if len(self.buff) == self.multi_step:
                state, action, reward = self.buff.get(self.gamma)
                self.priorities[self._p] = self.calc_priority(error)
                self._append(state, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self.priorities[self._p] = self.calc_priority(error)
            self._append(state, action, reward, next_state, done)

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1.0 - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(self.priorities[: self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[: self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super(PrioritizedMemory, self).reset()
        self.priorities = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.priorities[valid],
        )

    def _insert(self, mem_indices, batch, batch_indices):
        states, actions, rewards, next_states, dones, priorities = batch
        self.states[mem_indices] = states[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.priorities[mem_indices] = priorities[batch_indices]


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


# SAC


class SacAgent:
    def __init__(
        self,
        load,
        env,
        log_dir,
        num_steps=3000000,
        batch_size=256,
        lr=0.0003,
        hidden_units=[256, 256],
        memory_size=1e6,
        gamma=0.99,
        tau=0.005,
        entropy_tuning=True,
        ent_coef=0.2,
        multi_step=1,
        per=False,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.0001,
        grad_clip=None,
        updates_per_step=1,
        start_steps=10000,
        log_interval=10,
        target_update_interval=1,
        eval_interval=1000,
        cuda=True,
        seed=0,
    ):
        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.backends.cudnn.deterministic = False  # It harms a performance.
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu"
        )

        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units,
        ).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units,
        ).to(self.device)
        self.critic_target = (
            TwinnedQNetwork(
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                hidden_units=hidden_units,
            )
            .to(self.device)
            .eval()
        )

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        self.steps = 0

        if load:
            absolute_path = os.path.dirname(__file__)
            print("loading models from checkpoint")
            self.policy, self.policy_optim, self.steps = load_ckp(
                os.path.join(absolute_path, "load/bester_123K/best_policy.pth"),
                self.policy,
                self.policy_optim,
            )
            self.critic, self.q1_optim, _ = load_ckp(
                os.path.join(absolute_path, "load/bester_123K/best_critic.pth"),
                self.critic,
                self.q1_optim,
            )
            self.critic_target, self.q2_optim, _ = load_ckp(
                os.path.join(absolute_path, "load/bester_123K/best_critic_target.pth"),
                self.critic_target,
                self.q2_optim,
            )

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(self.device)
            ).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size,
                self.env.observation_space.shape,
                self.env.action_space.shape,
                self.device,
                gamma,
                multi_step,
                alpha=alpha,
                beta=beta,
                beta_annealing=beta_annealing,
            )
        else:
            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MultiStepMemory(
                memory_size,
                self.env.observation_space.shape,
                self.env.action_space.shape,
                self.device,
                gamma,
                multi_step,
            )

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, "model")
        self.summary_dir = os.path.join(log_dir, "summary")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma**multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.best_return = float("-inf")

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.0
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, action, reward, next_state, masked_done, self.device
                )
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state,
                    action,
                    reward,
                    next_state,
                    masked_done,
                    error,
                    episode_done=done,
                )
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, action, reward, next_state, masked_done, episode_done=done
                )

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()

            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar("reward/train", self.train_rewards.get(), self.steps)

        print(
            f"episode: {self.episodes:<4}  "
            f"episode steps: {episode_steps:<4}  "
            f"reward: {episode_reward:<5.1f}"
        )

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.0

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(
            batch, weights
        )
        # policy_loss, entropies = self.calc_policy_loss(batch, weights)  #

        update_params(self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

        # calculate `policy_loss` after updating `Q1、Q2`
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            self.writer.add_scalar(
                "loss/alpha", entropy_loss.detach().item(), self.steps
            )

        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                "loss/Q1", q1_loss.detach().item(), self.learning_steps
            )
            self.writer.add_scalar(
                "loss/Q2", q2_loss.detach().item(), self.learning_steps
            )
            self.writer.add_scalar(
                "loss/policy", policy_loss.detach().item(), self.learning_steps
            )
            self.writer.add_scalar(
                "stats/alpha", self.alpha.detach().item(), self.learning_steps
            )
            self.writer.add_scalar("stats/mean_Q1", mean_q1, self.learning_steps)
            self.writer.add_scalar("stats/mean_Q2", mean_q2, self.learning_steps)
            self.writer.add_scalar(
                "stats/entropy", entropies.detach().mean().item(), self.learning_steps
            )

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = torch.mean((-q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def evaluate(self, render=False):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action, render)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = np.mean(returns)
        if mean_return > self.best_return:
            self.best_return = mean_return
            self.save_models(True)

        self.writer.add_scalar("reward/test", mean_return, self.steps)
        print("-" * 60)
        print(
            f"Num steps: {self.steps:<5}  "
            f"reward: {mean_return:<5.1f} (best {self.best_return:<5.1f})"
        )
        print("-" * 60)

    def save_models(self, best=False):
        print("Saving models", best)
        checkpoint_dir = self.model_dir
        best_model_dir = os.path.join(checkpoint_dir, "best_model")
        checkpoint_policy = {
            "epoch": self.episodes + 1,
            "state_dict": self.policy.state_dict(),
            "optimizer": self.policy_optim.state_dict(),
        }
        checkpoint_critic = {
            "epoch": self.episodes + 1,
            "state_dict": self.critic.state_dict(),
            "optimizer": self.q1_optim.state_dict(),
        }
        checkpoint_critic_target = {
            "epoch": self.episodes + 1,
            "state_dict": self.critic_target.state_dict(),
            "optimizer": self.q2_optim.state_dict(),
        }
        save_ckp(checkpoint_policy, best, checkpoint_dir, best_model_dir, "policy.pth")
        save_ckp(checkpoint_critic, best, checkpoint_dir, best_model_dir, "critic.pth")
        save_ckp(
            checkpoint_critic_target,
            best,
            checkpoint_dir,
            best_model_dir,
            "critic_target.pth",
        )

        # self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        # self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        # self.critic_target.save(
        #     os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()


class Constants:
    def __init__(self):
        self.PI_2 = np.pi / 2
        self.GRAVITY = np.array([0, -2500.0])
        self.DRONE_RADIUS = 20.0


constants = Constants()


class Thruster:
    def __init__(
        self,
        angle=0.0,
        target_angle=0.0,
        angle_var_speed=10.0,
        max_angle=constants.PI_2,
        max_power=10000.0,
    ):
        self.max_power = max_power
        self.target_angle = target_angle
        self.angle_var_speed = angle_var_speed
        self.max_angle = max_angle
        self.angle = angle
        self.power_ratio = 0.0
        self.angle_ratio = self.angle / self.max_angle

    def reset(self):
        self.angle = 0.0
        self.power_ratio = 0.0

    def set_angle(self, ratio):
        # Clamp between [-max, +max]
        self.target_angle = self.max_angle * max(-1.0, min(1.0, ratio))

    def get_angle(self):
        return self.angle_ratio * self.max_angle

    def set_power(self, ratio):
        self.power_ratio = max(-1.0, min(1.0, ratio))

    def get_power(self):
        return self.power_ratio * self.max_power

    def update(self, dt):
        self.angle += self.angle_var_speed * (self.target_angle - self.angle) * dt
        self.angle_ratio = self.angle / self.max_angle


class DroneDynamics:
    def __init__(self, radius=20.0, position=np.zeros((2,))):
        self.left = Thruster()
        self.right = Thruster()
        self.thruster_offset = 35.0
        self.radius = radius
        self.position = np.copy(position)
        self.acceleration = np.zeros((2,))
        self.velocity = np.zeros((2,))
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.inertia = 8e-4
        self.mass = 1
        self.air_resistance_coeff = 1e-2

    def reset(self, position=np.zeros((2,))):
        self.velocity = np.zeros((2,))
        self.angle = 0.0
        self.position = np.copy(position)
        self.angular_velocity = 0.0
        self.left.reset()
        self.right.reset()
        self.set([0, 0, 0, 0])

    def get_thrust(self):
        angle_left = self.angle + self.left.angle
        angle_right = self.angle + self.right.angle

        unit_left = np.array([-np.sin(angle_left), np.cos(angle_left)])
        unit_right = np.array([-np.sin(angle_right), np.cos(angle_right)])

        thrust_left = self.left.get_power() * unit_left
        thrust_right = self.right.get_power() * unit_right

        return thrust_left + thrust_right

    def get_torque(self):
        thrust_left = self.left.get_power() * np.cos(self.left.angle)
        thrust_right = self.right.get_power() * np.cos(self.right.angle)

        return self.inertia * (thrust_right - thrust_left) * self.thruster_offset

    def update(self, dt):
        self.left.update(dt)
        self.right.update(dt)
        # Integration
        # print("thrust", self.get_thrust())
        force_net = (self.get_thrust() + constants.GRAVITY) / self.mass
        self.velocity += force_net * dt - self.velocity * self.air_resistance_coeff
        self.position += self.velocity * dt
        self.angular_velocity += (
            self.get_torque() * dt - self.angular_velocity * self.air_resistance_coeff
        )
        self.angle += self.angular_velocity * dt

    def set(self, action=np.zeros((4,))):
        self.left.set_power(0.5 * (np.clip(action[0], -1, 1) + 1.0))
        self.left.set_angle(action[1])
        self.right.set_power(0.5 * (np.clip(action[2], -1, 1) + 1.0))
        self.right.set_angle(action[3])


def normalize_angle(angle):
    normalized_angle = ((angle + 180) % 360) - 180
    normalized_angle /= 180
    return normalized_angle


def generate_waypoint_position(width, height):
    return np.array(
        [
            width // 2 + (width // 2) * random.uniform(-0.9, 0.9),
            height // 2 + (height // 2) * random.uniform(-0.9, 0.9),
        ]
    )


def is_object_close_to_waypoint(object_pos, waypoint_pos, thresh):
    dx = object_pos[0] - waypoint_pos[0]
    dy = object_pos[1] - waypoint_pos[1]
    error = dx * dx + dy * dy
    return error < thresh * thresh, error


class QuadrotorEnv:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        # set 2 dimensional continuous action space as continuous
        # [-1,2] for first dimension and [-2,4] for second dimension
        self.action_space = spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        # Distance to waypoint (r), angle_thrust_l, angle_trust_r, angle_D, angle_w, velocity_x, velocity_y, angular velocity,
        self.observation_space = spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self._max_episode_steps = 60000
        self.seed_ = None
        self.width = 640
        self.height = 480
        self.screen = Screen(width=self.width, height=self.height)
        self.drone_r = Drone(
            screen=self.screen,
            radius=20,
            position=np.array([self.width / 2, self.height / 2]),
        )
        self.drone = self.drone_r.drone
        self.waypoint_ts = 0
        self.waypoint_ts_thresh = 30
        self.step_count = 0
        self.waypoint_pos = generate_waypoint_position(self.width, self.height)

    def seed(self, seed):
        self.seed_ = seed

    def close(self):
        return True

    def reset(self):
        self.drone.reset(position=np.array([self.width / 2, self.height / 2]))
        self.waypoint_ts = 0
        self.step_count = 0
        self.waypoint_pos = generate_waypoint_position(self.width, self.height)

        # Normalize to 0 and 1
        velocity_x = (self.drone.velocity[0] - 100) / 200
        velocity_y = (self.drone.velocity[1] - 100) / 200
        angular_velocity = (self.drone.angular_velocity - 50) / 100
        angle_drone = normalize_angle(self.drone.angle)
        angle_thrust_left = normalize_angle(self.drone.left.get_angle())
        angle_thrust_right = normalize_angle(self.drone.right.get_angle())

        dx = self.waypoint_pos[0] - self.drone.position[0]
        dy = self.waypoint_pos[1] - self.drone.position[1]
        angle_waypoint = normalize_angle(np.arctan2(-dx, dy))

        _, error_sq = is_object_close_to_waypoint(
            self.drone.position, self.waypoint_pos, 20
        )
        distance = error_sq / (800 * 800)
        # [ 7.37416815e-02  1.84636685e-03 -1.21171450e-03 -5.44505904e-03
        # 1.70394089e-02  2.37647638e+00  1.57414376e+01 -5.95863938e-01]
        #  waypoint (r), angle_thrust_l, angle_trust_r, angle_D, angle_w, velocity_x, velocity_y, angular velocity,
        state = np.array(
            [
                distance,
                angle_thrust_left,
                angle_thrust_right,
                angle_drone,
                angle_waypoint,
                velocity_x,
                velocity_y,
                angular_velocity,
            ]
        )
        return state

    def step(self, action, render=False):
        self.step_count += 1

        reward = 0
        self.drone.set(action)
        self.drone.update(0.01)

        if render:
            self.screen().fill((0, 0, 0))
            self.drone_r.render()

            # Draw the waypoint
            pygame.draw.circle(
                self.screen(),
                (255, 255, 0),
                self.screen.to_pygame(
                    (int(self.waypoint_pos[0]), int(self.waypoint_pos[1]))
                ),
                10,
            )

            pygame.display.flip()
        self.clock.tick(60)

        done = (
            self.drone.position[0] + constants.DRONE_RADIUS < 0
            or self.drone.position[0] - constants.DRONE_RADIUS > self.width
            or self.drone.position[1] + constants.DRONE_RADIUS < 0
            or self.drone.position[1] - constants.DRONE_RADIUS > self.height
        ) or self.step_count > self._max_episode_steps
        if done and not self.step_count > self._max_episode_steps:
            reward -= 500

        is_close, error_sq = is_object_close_to_waypoint(
            self.drone.position, self.waypoint_pos, 20
        )
        if is_close:
            self.waypoint_ts += 1
            if self.waypoint_ts >= self.waypoint_ts_thresh:
                self.waypoint_ts = 0
                self.waypoint_pos = generate_waypoint_position(self.width, self.height)
                reward += 100

        reward += 1 - 1e-4 * error_sq

        # Normalize to 0 and 1
        velocity_x = (self.drone.velocity[0] - 100) / 200
        velocity_y = (self.drone.velocity[1] - 100) / 200
        angular_velocity = (self.drone.angular_velocity - 50) / 100
        angle_drone = normalize_angle(self.drone.angle)
        angle_thrust_left = normalize_angle(self.drone.left.get_angle())
        angle_thrust_right = normalize_angle(self.drone.right.get_angle())

        dx = self.waypoint_pos[0] - self.drone.position[0]
        dy = self.waypoint_pos[1] - self.drone.position[1]
        angle_waypoint = normalize_angle(np.arctan2(-dx, dy))

        distance = error_sq / (800 * 800)
        # [ 7.37416815e-02  1.84636685e-03 -1.21171450e-03 -5.44505904e-03
        # 1.70394089e-02  2.37647638e+00  1.57414376e+01 -5.95863938e-01]
        #  waypoint (r), angle_thrust_l, angle_trust_r, angle_D, angle_w, velocity_x, velocity_y, angular velocity,
        state = np.array(
            [
                distance,
                angle_thrust_left,
                angle_thrust_right,
                angle_drone,
                angle_waypoint,
                velocity_x,
                velocity_y,
                angular_velocity,
            ]
        )
        return state, reward, done, None


def run(eval=False):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env_id', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--cuda', action='store_true')
    # parser.add_argument('--seed', type=int, default=0)
    # args = parser.parse_args()

    env_id = "Quad-v2"
    cuda = False
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

    # env = gym.make(env_id)
    env = QuadrotorEnv()

    log_dir = os.path.join(
        "logs", env_id, f'sac-seed{seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    )

    agent = SacAgent(load=True, env=env, log_dir=log_dir, **configs)
    if not eval:
        agent.run()
    else:
        agent.evaluate(render=True)
    agent.save_models(True)
    pygame.quit()


if __name__ == "__main__":
    run(eval=True)
