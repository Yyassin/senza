from drone_renderer import DroneRenderer
from drone_dynamics import DroneDynamics
from screen import Screen
import numpy as np


class Drone:
    def __init__(
        self, screen: Screen, radius: float = 20.0, position: np.array = np.zeros((2,))
    ):
        self.drone = DroneDynamics(radius, position)
        if screen is not None:
            self.drone_renderer = DroneRenderer(self.drone, screen)

    def reset(self, position=np.zeros((2,))):
        self.drone.reset(position)

    def update(self, dt):
        self.drone.update(dt)

    def set(self, action: np.array = np.zeros((4,))):
        self.drone.set(action)

    def render(self):
        self.drone_renderer.render()
