from controllers.rl.models.GaussianPolicy import GaussianPolicy
from controllers.rl.models.TwinnedQNetwork import TwinnedQNetwork
from controllers.rl.utils.utils import hard_update, grad_false
from controllers.rl.utils.checkpoint import save_ckp, load_ckp
from controllers.rl.memory.MultiStepMemory import MultiStepMemory
from controllers.rl.utils.RunningStats import RunningMeanStats

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch
import numpy as np
import os


"""
Soft Actor Critic Agent
"""


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
        """
        Creates a new SAC agent.
        """
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

        # Copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)

        # Disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        self.steps = 0

        if load:
            absolute_path = os.path.dirname(__file__)
            print("loading models from checkpoint")
            self.policy, self.policy_optim, self.steps = load_ckp(
                os.path.join(absolute_path, "load/best_policy.pth"),
                self.policy,
                self.policy_optim,
            )
            self.critic, self.q1_optim, _ = load_ckp(
                os.path.join(absolute_path, "load/best_critic.pth"),
                self.critic,
                self.q1_optim,
            )
            self.critic_target, self.q2_optim, _ = load_ckp(
                os.path.join(absolute_path, "load/best_critic_target.pth"),
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
        """
        Runs the agent.
        """
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        """
        Checks whether the agent can be updated.
        """
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        """
        Retrieves the agent's next action according
        to the current state.

        :param state: The current state.
        """
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        """
        Samples an action from the learned policy according
        to the specified state.

        :param state: The current state.
        """
        # Act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        """
        Uses the mean action from the learned policy
        according to the specified state.

        :param state: The current state.
        """
        # Act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        """
        Calculates a batch of Q estimates according to the supplied batch of
        transitions.

        :param states: Batch of initial states.
        :param actions: Batch of actions.
        :param rewards: Batch of rewards.
        :param next_states: Batch of final states.
        :param dones: Batch of termination flags.
        """
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        """
        Calculates the new q target based on the most recent Q networks.

        :param states: Batch of initial states.
        :param actions: Batch of actions.
        :param rewards: Batch of rewards.
        :param next_states: Batch of final states.
        :param dones: Batch of termination flags.
        """
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        """
        Performs one episode of training.
        """
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
        """
        Updates networks.
        """
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

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
        """
        Calculates loss for both critics according to
        the specified state-action batch, with weight
        (unused, they're IS weights for prioritized replay).

        :param batch: The batch of state-action pairs.
        :param weights: IS weights.
        """
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
        """
        Calculates loss for policy according to
        the specified state-action batch, with weight
        (unused, they're IS weights for prioritized replay).

        :param batch: The batch of state-action pairs.
        :param weights: IS weights.
        """
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
        """
        Calculates entropy loss.

        :param entropy: Current entropy.
        :param weights: IS weights.
        """
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def evaluate(self):
        """
        Evaluates the current policy over 10 episodes.
        """
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
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
        """
        Saves the critic networks and policy checkpoints.

        :param best: True if these are the best models so far, False otherwise.
        """
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

    def __del__(self):
        """
        Agent destructor
        """
        self.env.close()
