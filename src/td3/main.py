import numpy as np
import torch
import gym
from TD3 import TD3_Agent, ReplayBuffer, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import str2bool, Reward_adapter, evaluate_policy
from buffer import PrioritizedReplayBuffer
from gym import spaces
import random
from rl_plotter.logger import Logger, CustomLogger


"""Hyperparameter Setting"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--EnvIdex", type=int, default=0, help="PV0, Lch_Cv2, Humanv2, HCv2, BWv3, BWHv3"
)
parser.add_argument(
    "--write",
    type=str2bool,
    default=True,
    help="Use SummaryWriter to record the training",
)
parser.add_argument("--render", type=str2bool, default=False, help="Render or Not")
parser.add_argument(
    "--Loadmodel", type=str2bool, default=False, help="Load pretrained model or Not"
)
parser.add_argument("--ModelIdex", type=int, default=30000, help="which model to load")

parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--update_every", type=int, default=50, help="training frequency")
parser.add_argument(
    "--Max_train_steps", type=int, default=5e6, help="Max training steps"
)
parser.add_argument(
    "--save_interval", type=int, default=1e5, help="Model saving interval, in steps."
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=2e3,
    help="Model evaluating interval, in steps.",
)

parser.add_argument(
    "--policy_delay_freq",
    type=int,
    default=1,
    help="Delay frequency of Policy Updating",
)
parser.add_argument("--gamma", type=float, default=0.99, help="Discounted Factor")
parser.add_argument("--net_width", type=int, default=256, help="Hidden net width")
parser.add_argument("--a_lr", type=float, default=1e-4, help="Learning rate of actor")
parser.add_argument("--c_lr", type=float, default=1e-4, help="Learning rate of critic")
parser.add_argument(
    "--batch_size", type=int, default=256, help="batch_size of training"
)
parser.add_argument("--exp_noise", type=float, default=0.15, help="explore noise")
parser.add_argument(
    "--noise_decay", type=float, default=0.998, help="Decay rate of explore noise"
)
opt = parser.parse_args()
print(opt)


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


def main():
    logger = Logger(
        log_dir="/custom_logger/LLd-v2", exp_name="LLd-v2", env_name="myenv", seed=0
    )
    custom_logger = logger.new_custom_logger(
        filename="loss.csv", fieldnames=["episode_score", "mean_score"]
    )
    best_score = float("-inf")
    EnvName = [
        "Pendulum-v0",
        "LunarLanderContinuous-v2",
        "Humanoid-v2",
        "HalfCheetah-v2",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
    ]
    BrifEnvName = [
        "PV0",
        "LLdV2",
        "Humanv2",
        "HCv2",
        "BWv3",
        "BWHv3",
    ]  # Brief Environment Name.
    env_with_dw = [False, True, True, False, True, True]  # dw:die and win
    EnvIdex = opt.EnvIdex
    env = QuadrotorEnv()  # gym.make(EnvName[EnvIdex])
    eval_env = QuadrotorEnv()  # gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # remark: action space【-max,max】
    expl_noise = opt.exp_noise
    max_e_steps = env._max_episode_steps
    print(
        "Env:",
        EnvName[EnvIdex],
        "  state_dim:",
        state_dim,
        "  action_dim:",
        action_dim,
        "  max_a:",
        max_action,
        "  min_a:",
        env.action_space.low[0],
        "  max_e_steps:",
        max_e_steps,
    )

    update_after = 2 * max_e_steps  # update actor and critic after update_after steps
    start_steps = (
        10 * max_e_steps
    )  # start using actor to iterate after start_steps steps

    # Random seed config:
    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    # env._seed(random_seed)
    # eval_env._seed(random_seed)
    np.random.seed(random_seed)

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = " " + timenow[0:13] + "_" + timenow[-2::]
        writepath = "runs/{}".format(BrifEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "env_with_dw": env_with_dw[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": opt.gamma,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "policy_delay_freq": opt.policy_delay_freq,
    }
    if not os.path.exists("model"):
        os.mkdir("model")
    model = TD3_Agent(**kwargs)
    if opt.Loadmodel:
        model.load(BrifEnvName[EnvIdex], opt.ModelIdex)

    # replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, int(1e6))

    if opt.render:
        score = evaluate_policy(env, model, opt.render, turns=10)
        print("EnvName:", BrifEnvName[EnvIdex], "score:", score)
    else:
        total_steps = 0
        score_history = []
        while total_steps < opt.Max_train_steps:
            # (s, _), done, steps, ep_r = env.reset(), False, 0, 0
            s, done, steps, ep_r = env.reset(), False, 0, 0

            """Interact & trian"""
            while not done:
                steps += 1  # steps in one episode

                if total_steps < start_steps:
                    a = env.action_space.sample()
                else:
                    a = (
                        model.select_action(s)
                        + np.random.normal(0, max_action * expl_noise, size=action_dim)
                    ).clip(
                        -max_action, max_action
                    )  # explore: deterministic actions + noise
                s_prime, r, done, _ = env.step(a)
                r = Reward_adapter(r, EnvIdex)

                """Avoid impacts caused by reaching max episode steps"""
                if done and steps != max_e_steps:
                    dw = True  # dw: dead and win
                else:
                    dw = False

                # Reward as initial priority
                replay_buffer.add((s, a, r, s_prime, int(dw)))
                s = s_prime
                ep_r += r

                """train if its time"""
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= update_after and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(replay_buffer)

                """record & log"""
                if total_steps % opt.eval_interval == 0:
                    expl_noise *= opt.noise_decay
                    score = evaluate_policy(eval_env, model, False)
                    if opt.write:
                        writer.add_scalar("ep_r", score, global_step=total_steps)
                        writer.add_scalar(
                            "expl_noise", expl_noise, global_step=total_steps
                        )
                    print(
                        "EnvName:",
                        BrifEnvName[EnvIdex],
                        "steps: {}k".format(int(total_steps / 1000)),
                        "score:",
                        score,
                    )
                total_steps += 1

                """save model"""
                if score > best_score:  # total_steps % opt.save_interval == 0
                    model.save(BrifEnvName[EnvIdex], total_steps, True)
                    best_score = score

            score_history.append(ep_r)
            avg_score = np.mean(score_history[-100:])
            custom_logger.update([ep_r, avg_score], total_steps=total_steps)
            logger.update(score=score_history[-100:], total_steps=total_steps)

        env.close()


if __name__ == "__main__":
    main()
