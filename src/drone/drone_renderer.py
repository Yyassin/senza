from drone.drone_dynamics import DroneDynamics
from drone.screen import Screen
import constants

from pygame import Surface, transform
import pygame
import numpy as np

"""
Renderer that draws a drone, according
to current state, to the pygame screen.
"""


class BaseEntity:
    def __init__(self, x: int, y: int):
        """
        Creates a pygame entity with some
        initial position.

        :param x: X coordinate for position.
        :param y: Y coordiante for position.
        """
        self.x = x
        self.y = y


class Rectangle(BaseEntity):
    # Source: https://stackoverflow.com/questions/36510795/rotating-a-rectangle-not-image-in-pygame
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        screen: Screen,
        color: tuple = (0, 255, 0),
    ):
        """
        Creates a new rectangle object (which can be rotated,
        yea --- pygame doesn't let you do that).

        :param x: The rectangle's center x coordinate.
        :param y: The rectangle's center y coordinate.
        :param width: The rectangle's width.
        :param height: The rectangle's height.
        :param screen: The screen where the rectangle should be rendered.
        :param color: The rectangle's color.
        """
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.color = color
        self.screen = screen
        self.rotation = 0

        # The rectangle is a surface itself
        self.surface = Surface((width, height))
        self.surface.set_colorkey((0, 0, 0))
        self.surface.fill(color)
        self.rect = self.surface.get_rect()

    def display(self, angle=None):
        """
        Renders the rectangle on the screen.

        :param angle: The angle (relative to positive x) to
        render the rectangle at.
        """
        # updating values
        self.surface.fill(
            self.color
        )  # refill the surface color if you change it somewhere in the program
        self.rect = self.surface.get_rect()
        self.rect.center = (self.x, self.y)

        # renderer
        if angle is not None:
            self.rotation = angle

        old_center = self.rect.center
        new = transform.rotate(self.surface, np.rad2deg(self.rotation))
        self.rect = new.get_rect()
        self.rect.center = old_center
        self.screen().blit(new, self.rect)


class DroneRenderer:
    def __init__(self, drone: DroneDynamics, screen: Screen):
        """
        Creates a new drone renderer that renders the supplied
        drone object to the supplied screen.

        :param drone: The drone to render.
        :param screen: The screen to render the drone on.
        """
        self.drone = drone
        self.radius = 20
        self.color = (255, 255, 255)
        self.center_line_length = self.radius * 2
        self.center_line_color = (255, 0, 0)
        self.thruster_offset = 40

        self.thruster_width = 10
        self.thruster_height = 20
        self.power_bar_height = 50
        self.thruster_color = (0, 255, 0)

        self.screen = screen

    def render(self):
        """
        Draws the drone on the screen.
        """
        self.render_body()
        self.render_left()
        self.render_right()

    def render_body(self):
        """
        Renders the drone body.
        """
        drone_x, drone_y = self.drone.position[0], self.drone.position[1]
        drone_center = (drone_x, drone_y)

        center_line_start = drone_center
        center_line_end = (
            drone_x + self.radius * np.cos(self.drone.angle + constants.PI_2),
            drone_y + self.radius * np.sin(self.drone.angle + constants.PI_2),
        )

        # Draw the drone body
        pygame.draw.circle(
            self.screen(), self.color, self.screen.to_pygame(drone_center), self.radius
        )
        # And center line
        pygame.draw.line(
            self.screen(),
            self.center_line_color,
            self.screen.to_pygame(center_line_start),
            self.screen.to_pygame(center_line_end),
            2,
        )

    def render_thruster(
        self, center_x, center_y, angle, width, height, power_ratio, color
    ):
        """
        Generic method to render a single drone thruster.

        :param center_x: X coordinate of the thruster's center.
        :param center_y: Y coordinate of the thruster's center.
        :param angle: The thruster's angle relative to positive y.
        :param width: The thruster's width.
        :param height: The thruster's height.
        :param power_ratio: The ratio of power applied by the thruster.
        :param color: The thruster's color.
        """
        power_x, power_y = self.screen.to_pygame(
            (
                center_x
                - ((height / 2) + power_ratio * self.power_bar_height / 2)
                * np.sin(self.drone.angle + angle),
                center_y
                + ((height / 2) + power_ratio * self.power_bar_height / 2)
                * np.cos(self.drone.angle + angle),
            )
        )
        center_x, center_y = self.screen.to_pygame((center_x, center_y))
        thruster = Rectangle(center_x, center_y, width, height, self.screen, color)
        thruster_power = Rectangle(
            power_x,
            power_y,
            2,
            power_ratio * self.power_bar_height,
            self.screen,
            (255, 255, 0),
        )
        thruster.display(self.drone.angle + angle)
        thruster_power.display(self.drone.angle + angle)

    def render_left(self):
        """
        Renders the left thruster.
        """
        drone_x, drone_y = self.drone.position[0], self.drone.position[1]

        left_thruster_x = drone_x - self.thruster_offset * np.cos(self.drone.angle)
        left_thruster_y = drone_y - self.thruster_offset * np.sin(self.drone.angle)
        self.render_thruster(
            left_thruster_x,
            left_thruster_y,
            self.drone.left.get_angle(),
            self.thruster_width,
            self.thruster_height,
            self.drone.left.power_ratio,
            self.thruster_color,
        )

    def render_right(self):
        """
        Renders the right thruster.
        """
        drone_x, drone_y = self.drone.position[0], self.drone.position[1]

        right_thruster_x = drone_x + self.thruster_offset * np.cos(self.drone.angle)
        right_thruster_y = drone_y + self.thruster_offset * np.sin(self.drone.angle)
        self.render_thruster(
            right_thruster_x,
            right_thruster_y,
            self.drone.right.get_angle(),
            self.thruster_width,
            self.thruster_height,
            self.drone.right.power_ratio,
            (255, 0, 0),
        )
