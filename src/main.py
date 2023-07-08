import os
import sys

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
from screen import Screen
from drone import Drone
from drone_dynamics import DroneDynamics
import numpy as np
import copy
import constants
import pygame
import random
import multiprocessing
from soft_actor_critic.sac import SacAgent
from datetime import datetime
from gym import spaces


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
        self._max_episode_steps = 30000
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
        angle_waypoint = normalize_angle(np.arctan2(dx, dy))

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
            self.drone.position[0] + 20 * constants.DRONE_RADIUS < 0
            or self.drone.position[0] - 20 * constants.DRONE_RADIUS > self.width
            or self.drone.position[1] + 20 * constants.DRONE_RADIUS < 0
            or self.drone.position[1] - 20 * constants.DRONE_RADIUS > self.height
        ) or self.step_count > self._max_episode_steps
        if done:
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

        reward += 1
        reward -= 1e-7 * error_sq

        # Normalize to 0 and 1
        velocity_x = (self.drone.velocity[0] - 100) / 200
        velocity_y = (self.drone.velocity[1] - 100) / 200
        angular_velocity = (self.drone.angular_velocity - 50) / 100
        angle_drone = normalize_angle(self.drone.angle)
        angle_thrust_left = normalize_angle(self.drone.left.get_angle())
        angle_thrust_right = normalize_angle(self.drone.right.get_angle())

        dx = self.waypoint_pos[0] - self.drone.position[0]
        dy = self.waypoint_pos[1] - self.drone.position[1]
        angle_waypoint = normalize_angle(
            np.arctan2(dx, dy)
        )  # dx should be negative I think, but I already trained :P

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


class GeneticAlgorithm:
    def __init__(
        self, population_size, mutation_rate, max_generations, fitness_function, init
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.fitness_function = fitness_function
        self.init = init

    def optimize(self):
        population = self.generate_initial_population()

        for generation in range(self.max_generations):
            # Parallelize fitness scores
            fitness_scores = self.calculate_fitness_parallel(population)
            population, total_fitness = self.select_parents(fitness_scores)
            population = self.cross_over(population)
            self.mutate(population)

            print(f"mean fitness: {total_fitness / self.population_size}")
            print(f"best gains: {fitness_scores[0][0]}")

        best_gains = fitness_scores[0][0]
        return best_gains

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            gains = [random.uniform(0, 5) for _ in range(12)]
            population.append(gains)
        population[0] = copy.deepcopy(self.init)
        population[1] = copy.deepcopy(self.init)
        population[2] = copy.deepcopy(self.init)
        return population

    def calculate_fitness(self, population):
        fitness_scores = []
        for gains in population:
            fitness = self.fitness_function(gains)
            fitness_scores.append((gains, fitness))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores

    def calculate_fitness_parallel(self, population):
        pool = multiprocessing.Pool()  # Create a pool of worker processes
        try:
            fitness_scores = pool.map(
                self.fitness_function, population
            )  # Parallelize fitness calculation
            pool.close()
            pool.join()
            fitness_scores = [
                (population[i], fitness_scores[i]) for i in range(len(population))
            ]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            return fitness_scores
        except:
            pool.close()
            pool.terminate()
            pool.join()
            sys.exit()

    def select_parents(self, fitness_scores):
        selected_parents = []

        # Add the two parents with the highest fitness scores
        selected_parents.append((fitness_scores[0][0], fitness_scores[1][0]))

        total_fitness = sum(score[1] for score in fitness_scores)
        probabilities = [score[1] / total_fitness for score in fitness_scores]
        population_sorted = [parent[0] for parent in fitness_scores]
        for _ in range(self.population_size // 2 - 1):
            parent1 = self.roulette_wheel_selection(population_sorted, probabilities)
            parent2 = self.roulette_wheel_selection(population_sorted, probabilities)
            selected_parents.append((parent1, parent2))

        return selected_parents, total_fitness

    def roulette_wheel_selection(self, population, probabilities):
        cumulative_probabilities = [
            sum(probabilities[: i + 1]) for i in range(len(probabilities))
        ]
        rand_value = random.uniform(0, 1)
        for i, cum_prob in enumerate(cumulative_probabilities):
            if rand_value <= cum_prob:
                return copy.deepcopy(population[i])

    def cross_over(self, selected_parents):
        new_population = []
        for parents in selected_parents:
            parent1, parent2 = parents

            # Can make this random probably
            child1_gains = parent1[:4] + parent2[4:8] + parent1[8:]
            child2_gains = parent2[:4] + parent1[4:8] + parent2[8:]
            new_population.append(child1_gains)
            new_population.append(child2_gains)

        return new_population

    def mutate(self, population):
        for gains in population:
            if random.uniform(0, 1) < self.mutation_rate:
                for index in range(len(gains)):
                    if random.uniform(0, 1) < self.mutation_rate:
                        gains[index] += random.uniform(-0.02, 0.02)
                        gains[index] = max(0, gains[index])


class Gains:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


class PID:
    def __init__(self, gains=[]):
        self.step = 0

        if len(gains):
            self.x_gains = Gains(gains[0], 0, gains[2] * 0.000001)
            self.y_gains = Gains(gains[3], 0, gains[5] * 0.000001)
            self.theta_gains = Gains(gains[6], 0, gains[8] * 0.000001)
            self.angular_gains = Gains(gains[9], 0, gains[11] * 0.000001)
        else:
            self.x_gains = Gains(0, 0, 0)
            self.y_gains = Gains(0.0, 0.0, 0.0)
            self.theta_gains = Gains(0, 0, 0.00)
            self.angular_gains = Gains(0.0, 0, 0.00)

        self.x_int = 0
        self.y_int = 0
        self.theta_int = 0
        self.angular_int = 0

        self.x_error_prev = 0
        self.y_error_prev = 0
        self.theta_error_prev = 0
        self.angular_error_prev = 0

    def reset(self):
        self.x_int = 0
        self.y_int = 0
        self.theta_int = 0
        self.angular_int = 0

        self.x_error_prev = 0
        self.y_error_prev = 0
        self.theta_error_prev = 0
        self.angular_error_prev = 0

    def compute(self, desired_pose, current_pose, angular, velocity, dt):
        # Pose: x, y, theta
        x_error = desired_pose[0] - current_pose[0]
        y_error = desired_pose[1] - current_pose[1]
        if self.step == 0:
            self.x_error_prev = x_error
            self.y_error_prev = y_error

        self.x_int += x_error

        # Negative x error means positive theta (left)
        theta_desired = -(
            self.x_gains.kp * x_error
            + self.x_gains.ki * self.x_int
            + self.x_gains.kd * velocity[0]
        )

        theta_error = theta_desired - current_pose[2]
        self.y_int += y_error
        self.theta_int += theta_error
        if self.step == 0:
            self.theta_error_prev = theta_error

        angular_desired = (
            self.theta_gains.kp * theta_error
            + self.theta_gains.ki * self.theta_int
            + self.theta_gains.kd * (theta_error - self.theta_error_prev) / dt
        )

        angular_error = angular_desired - angular
        self.angular_int += angular_error
        if self.step == 0:
            self.angular_error_prev = angular_error

        thrust_desired = (
            self.y_gains.kp * y_error
            + self.y_gains.ki * self.y_int
            + self.y_gains.kd * velocity[1]
        ) - constants.GRAVITY[1]
        thrust_net_desired = thrust_desired / np.cos(current_pose[2])

        roll_desired = (
            self.angular_gains.kp * angular_error
            + self.angular_gains.ki * self.angular_int
            + self.angular_gains.kd * (angular_error - self.angular_error_prev) / dt
        )

        # Clamp integrals at saturation
        self.x_int = np.clip(self.x_int, -100, 100)
        self.y_int = np.clip(self.y_int, -100, 100)
        self.theta_int = np.clip(self.theta_int, -100, 100)
        self.angular_int = np.clip(self.angular_int, -100, 100)

        F1 = thrust_net_desired - roll_desired
        F2 = thrust_net_desired + roll_desired

        self.step += 1

        # print((F1 - 10000) / 10000, (F2 - 10000) / 10000)

        # print("poses:", desired_pose, current_pose)
        # print("pose errors", y_error, x_error)
        # print(
        #     "theta",
        #     theta_desired,
        #     "angular",
        #     angular_desired,
        #     "thrust",
        #     thrust_desired,
        #     "roll",
        #     roll_desired,
        # )

        return [(F1 - 10000) / 10000, -0.5, (F2 - 10000) / 10000, 0.5]


YELLOW = (255, 255, 0)


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


height = 480
width = 640


def run(render=False, gains=[]):
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

    pygame.init()
    clock = pygame.time.Clock()
    screen = Screen(width=width, height=height) if render else None

    pid = PID(gains)
    center_pose = np.array([width / 2, height / 2])
    drone = Drone(
        screen=screen,
        radius=constants.DRONE_RADIUS,
        position=center_pose,
    )

    waypoint_pos = generate_waypoint_position(width, height)
    waypoint_radius = 10
    waypoint_timer = 0
    episode_timer = 0
    waypoint_timer_threshold = 100  # 5 seconds in milliseconds
    fitness = 0

    running = True
    while running:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        episode_timer += clock.get_time()

        is_out_of_screen = (
            drone.drone.position[0] + constants.DRONE_RADIUS < 0
            or drone.drone.position[0] - constants.DRONE_RADIUS > width
            or drone.drone.position[1] + constants.DRONE_RADIUS < 0
            or drone.drone.position[1] - constants.DRONE_RADIUS > height
        )
        print(drone.drone.position)
        if is_out_of_screen or episode_timer > 50000:
            if not render:
                # print(fitness)
                return fitness
            episode_timer = 0
            waypoint_pos = generate_waypoint_position(width, height)
            waypoint_timer = 0
            fitness = 0
            drone.reset(center_pose)
            drone.render()
            pid.reset()
            continue

        is_close, error_sq = is_object_close_to_waypoint(
            drone.drone.position, waypoint_pos, 2 * waypoint_radius
        )
        print(error_sq)
        if is_close:
            waypoint_timer += clock.get_time()
            if waypoint_timer >= waypoint_timer_threshold:
                waypoint_pos = generate_waypoint_position(width, height)
                waypoint_timer = 0
                fitness += 0.1

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

        # control = pid.compute(
        #     [waypoint_pos[0], waypoint_pos[1]],
        #     [drone.drone.position[0], drone.drone.position[1], drone.drone.angle],
        #     drone.drone.angular_velocity,
        #     drone.drone.velocity,
        #     1 / 60,
        # )
        control = agent.exploit(state)
        drone.set(control)

        # Get the state of all keyboard keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            drone.set([1, 0, 1, 0])
        if keys[pygame.K_LEFT]:
            drone.set([1, 0, -1, 0])
        if keys[pygame.K_RIGHT]:
            drone.set([-1, 0, 1, 0])

        drone.update(0.01)

        fitness += 1 / (error_sq + 1e-10)

        if render:
            screen().fill((0, 0, 0))
            drone.render()

            # Draw the waypoint
            pygame.draw.circle(
                screen(),
                YELLOW,
                screen.to_pygame((int(waypoint_pos[0]), int(waypoint_pos[1]))),
                waypoint_radius,
            )

            pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def fitness_function(gains):
    return run(render=False, gains=gains)


init = [
    3.835030336113285,
    1.0428983674523202,
    3000000.302848277197553,
    3.835030336113285,
    1.0428983674523202,
    200000.302848277197553,
    0.074765286834888,
    0.0162952179653564,
    50.315146563004506,
    0.0003,
    0,
    400,
]


def tune():
    genetic_algorithm = GeneticAlgorithm(
        population_size=50,
        mutation_rate=0.05,
        max_generations=100,
        fitness_function=fitness_function,
        init=init,
    )
    best_gains = genetic_algorithm.optimize()
    return best_gains


if __name__ == "__main__":
    # print(tune())
    run(render=True, gains=init)
