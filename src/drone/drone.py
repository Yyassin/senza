from drone.drone_renderer import DroneRenderer
from drone.drone_dynamics import DroneDynamics
from drone.screen import Screen

import numpy as np

"""
Encapsulates drone dynamics and rendering
into one common interface.
"""


class Drone:
    def __init__(
        self, screen: Screen, radius: float = 20.0, position: np.array = np.zeros((2,))
    ):
        """
        Creates a new drone.

        :param screen: The renderer screen.
        :param radius: Radius of the central drone body.
        :param position: The drone's starting and reset position.
        """
        self.drone = DroneDynamics(radius, position)
        if screen is not None:
            self.drone_renderer = DroneRenderer(self.drone, screen)

    def reset(self, position=np.zeros((2,))):
        """
        Resets the drone state.

        :param position: The position to reset to.
        """
        self.drone.reset(position)

    def update(self, dt):
        """
        Performs a single timestep
        state propogation on the drone.

        :param dt: The length of the timestep.
        """
        self.drone.update(dt)

    def set(self, action: np.array = np.zeros((4,))):
        """
        Sets the drone's next action.

        :param action: The action to set, formatted
        as a 4-element vector containing
        [left thruster power, left thruster angle,
            right thruster power, right thruster angle].
        """
        self.drone.set(action)

    def render(self):
        """
        Renders the current drone state
        on the screen.
        """
        self.drone_renderer.render()
