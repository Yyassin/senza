import pygame
import constants
import numpy as np

"""
Drone dynamics and state logic.
"""


class Thruster:
    def __init__(
        self,
        angle=0.0,
        target_angle=0.0,
        angle_var_speed=10.0,
        max_angle=constants.PI_2,
        max_power=10000.0,
    ):
        """
        Creates a new drone thruster, which
        has an orientation and thrust power / force.

        :param angle: The orientation of the thruster, relative
        to positive y (up).
        :param target_angle: The desired / set orientation of the
        thruster (we soft update towards this value every timestep).
        :param angle_var_speed: How fast the thruster's angle
        converges to the target.
        :param max_angle: Maximum absolute angle of the thruster, so
        the range is then [-max_angle, max_angle].
        :param max_power: The maximum force/power of the thruster.
        """
        self.max_power = max_power
        self.target_angle = target_angle
        self.angle_var_speed = angle_var_speed
        self.max_angle = max_angle
        self.angle = angle
        self.power_ratio = 0.0
        self.angle_ratio = self.angle / self.max_angle

    def reset(self):
        """
        Resets this thruster's state.
        """
        self.angle = 0.0
        self.power_ratio = 0.0

    def set_angle(self, ratio):
        """
        Sets the thruster's angle within the extents range.

        :param ratio: A value between [-1, 1] that corresponds
        to the factor to multiply with the max angle.
        """
        # Clamp between [-max, +max]
        self.target_angle = self.max_angle * max(-1.0, min(1.0, ratio))

    def get_angle(self):
        """
        Returns the thruster's current angle.
        """
        return self.angle_ratio * self.max_angle

    def set_power(self, ratio):
        """
        Sets the thruster's power.

        :param ratio: The ratio of the max power to set.
        """
        self.power_ratio = max(-1.0, min(1.0, ratio))

    def get_power(self):
        """
        Returns the thruster's power.
        """
        return self.power_ratio * self.max_power

    def update(self, dt):
        """
        Soft updates the thruster's angle
        towards the target angle.

        :param dt: The timestep size.
        """
        self.angle += self.angle_var_speed * (self.target_angle - self.angle) * dt
        self.angle_ratio = self.angle / self.max_angle


class DroneDynamics:
    def __init__(self, radius=20.0, position=np.zeros((2,))):
        """
        Creates a new drone dynamics solver.

        :param radius: The drone central body's radius, used
        to calculate moments.
        :param position: The drone's starting position.
        """
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
        """
        Resets the drone's state.
        :param positon: The position to reset to.
        """
        self.velocity = np.zeros((2,))
        self.angle = 0.0
        self.position = np.copy(position)
        self.angular_velocity = 0.0
        self.left.reset()
        self.right.reset()
        self.set([0, 0, 0, 0])

    def get_thrust(self):
        """
        Calculates and returns the components
        of net thrust in the positive y and x directions. Note
        that we interpret an angle of 0 as positive y.
        """
        angle_left = self.angle + self.left.angle
        angle_right = self.angle + self.right.angle

        unit_left = np.array([-np.sin(angle_left), np.cos(angle_left)])
        unit_right = np.array([-np.sin(angle_right), np.cos(angle_right)])

        thrust_left = self.left.get_power() * unit_left
        thrust_right = self.right.get_power() * unit_right

        return thrust_left + thrust_right

    def get_torque(self):
        """
        Calculates and returns the net torque of the drone.
        """
        thrust_left = self.left.get_power() * np.cos(self.left.angle)
        thrust_right = self.right.get_power() * np.cos(self.right.angle)

        return self.inertia * (thrust_right - thrust_left) * self.thruster_offset

    def update(self, dt):
        """
        Integrates the drone state according to the
        supplied timestep to propogate the state in time.

        :param dt: The size of the timestep.
        """
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
        """
        Sets the drone's control action.

        :param action: The action to set, formatted
        as a 4-element vector containing
        [left thruster power, left thruster angle,
            right thruster power, right thruster angle].
        """
        self.left.set_power(0.5 * (np.clip(action[0], -1, 1) + 1.0))
        self.left.set_angle(action[1])
        self.right.set_power(0.5 * (np.clip(action[2], -1, 1) + 1.0))
        self.right.set_angle(action[3])
