from utils import normalize_angle
from drone.screen import Screen
from drone.drone import Drone
import constants

from gym import spaces
import numpy as np
import pygame
import random

"""
Gym-like environment wrapper
for the 2d drone environment.
"""


def generate_waypoint_position(width, height):
    """
    Generates a random position for a waypoint.

    :param width: Width bounds for the position.
    :param height: Height bounds for the position.
    """
    return np.array(
        [
            width // 2 + (width // 2) * random.uniform(-0.7, 0.7),
            # This should be height // 2 but I trained on
            # this by accident (:cry:)
            height // 2 + 10 * random.uniform(-0.5, 0.5),
        ]
    )


def is_object_close_to_waypoint(object_pos, waypoint_pos, thresh):
    """
    Checks if the supplied object position is
    within {thresh} units of the supplied waypoint position.

    :param object_pos: The object position.
    :param waypoint_pos: The waypoint's position.
    :param thresh: The maximum distance where the object
    is considered close to the waypoint.
    """
    dx = object_pos[0] - waypoint_pos[0]
    dy = object_pos[1] - waypoint_pos[1]
    error = dx * dx + dy * dy
    return error <= thresh * thresh, error


class DroneEnv:
    def __init__(self, render=False):
        """
        Creates a new drone environment.

        :param render: True if the environment should
        be rendered and False otherwise.
        """
        self.render = render
        if self.render:
            pygame.init()
            self.clock = pygame.time.Clock()

        # Set 4 dimensional continuous action space
        self.action_space = spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Distance to waypoint (r), angle_thrust_l, angle_trust_r, angle_D, angle_w,
        # velocity_x, velocity_y, angular velocity,
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
        """
        Sets the environment's seed.
        """
        self.seed_ = seed
        random.seed(seed)

    def close(self):
        """
        Environment destructor (does nothing, but there for interface).
        """
        if self.render:
            pygame.quit()
        return True

    def get_state(self):
        """
        Returns the current state observation.
        """
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

    def reset(self):
        """
        Resets the environment state.
        """
        self.drone.reset(position=np.array([self.width / 2, self.height / 2]))
        self.waypoint_ts = 0
        self.step_count = 0
        self.waypoint_pos = generate_waypoint_position(self.width, self.height)
        return self.get_state()

    def step(self, action):
        """
        Performs one step in the environment with the supplied action.

        :param action: The action to act with. A 4 dimensional vector
        with the form: [left thruster power, left thruster angle,
                        right thruster power, right thruster angle].
        """
        self.step_count += 1

        reward = 0
        self.drone.set(action)
        self.drone.update(0.01)

        if self.render:
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

        state = self.get_state()
        return state, reward, done, None
