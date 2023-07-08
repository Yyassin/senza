import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os
from gym import spaces
import random


def plot_learning_curve(scores, x, figure_file):
    # Plots running average of last 100 scores
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Uniform dist to sample from range
        max_mem = min(self.mem_cntr, self.mem_size)
        # Sample batch_size from [0:max_mem]
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class TD3CriticNetwork(nn.Module):
    def __init__(
        self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/td3"
    ):
        super(TD3CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_td3")

        # I think this breaks if the env has a 2D state representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)
        make_dir(self.checkpoint_dir)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        # print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class TD3ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        fc1_dims,
        fc2_dims,
        n_actions,
        name,
        chkpt_dir="tmp/td3",
    ):
        super(TD3ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_td3")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)
        make_dir(self.checkpoint_dir)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        prob = T.tanh(self.mu(prob))  # if action is > +/- 1 then multiply by max action

        return prob

    def save_checkpoint(self):
        # print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class TD3Agent:
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        n_actions=2,
        max_size=100000,
        layer1_size=400,
        layer2_size=300,
        batch_size=100,
        noise=0.1,
    ):  # noise is for action exploration, not regularized update
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = TD3ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="actor",
        )
        self.critic_1 = TD3CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="critic_1",
        )
        self.critic_2 = TD3CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="critic_2",
        )

        self.target_actor = TD3ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="target_actor",
        )
        self.target_critic_1 = TD3CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="target_critic_1",
        )
        self.target_critic_2 = TD3CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="target_critic_2",
        )

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(
                np.random.normal(scale=self.noise, size=(self.n_actions,))
            ).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(
            self.actor.device
        )
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(
            T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5
        )
        # might break if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = (
                tau * critic_1_state_dict[name].clone()
                + (1 - tau) * target_critic_1_state_dict[name].clone()
            )

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = (
                tau * critic_2_state_dict[name].clone()
                + (1 - tau) * target_critic_2_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        print("saving")
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()


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
            width // 2 + (width // 2) * random.uniform(-0.5, 0.5),
            height // 2 + 0 * (height // 2) * random.uniform(-0.5, 0.5),
        ]
    )


def is_object_close_to_waypoint(object_pos, waypoint_pos, thresh):
    dx = object_pos[0] - waypoint_pos[0]
    dy = object_pos[1] - waypoint_pos[1]
    error = dx * dx + dy * dy
    return error < thresh * thresh, error


class QuadrotorEnv:
    def __init__(self):
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
        self.drone = DroneDynamics(position=np.array([self.width / 2, self.height / 2]))
        self.waypoint_ts = 0
        self.waypoint_ts_thresh = 30
        self.step_count = 0
        self.waypoint_pos = generate_waypoint_position(self.width, self.height)

    def seed(self, seed):
        self.seed_ = seed

    def close():
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

    def step(self, action):
        self.step_count += 1

        reward = 0
        self.drone.set(action)
        self.drone.update(0.01)
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


def td3_agent_loop():
    # env = gym.make('BipedalWalker-v2')
    # env = gym.make('LunarLanderContinuous-v2')
    env = QuadrotorEnv()
    agent = TD3Agent(
        alpha=0.001,
        beta=0.001,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        batch_size=100,
        layer1_size=400,
        layer2_size=300,
        n_actions=env.action_space.shape[0],
    )
    n_games = 10000
    filename = "Quad_" + str(n_games) + ".png"
    figure_file = "plots/" + filename
    make_dir("plots")

    best_score = float("-inf")
    score_history = []

    # agent.load_models()
    # We should save the optims too

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode ",
            i,
            "score %.2f" % score,
            "trailing 100 games avg %.3f" % avg_score,
        )

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(score_history, x, figure_file)


if __name__ == "__main__":
    td3_agent_loop()
