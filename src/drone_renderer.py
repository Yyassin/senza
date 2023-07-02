from pygame import Surface, transform
import pygame
import numpy as np
from drone_dynamics import DroneDynamics
from screen import Screen
import constants

class BaseEntity:
    def __init__(self, x: int, y: int):
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
        self.drone = drone
        # Consider passing some of these in
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
        self.render_body()
        self.render_left()
        self.render_right()

    def render_body(self):
        drone_x, drone_y = self.drone.position[0], self.drone.position[1]
        drone_center = (drone_x, drone_y)

        center_line_start = drone_center
        center_line_end = (
            drone_x + self.radius * np.cos(self.drone.angle + constants.PI_2),
            drone_y + self.radius * np.sin(self.drone.angle + constants.PI_2),
        )

        # Draw the drone body
        pygame.draw.circle(self.screen(), self.color, self.screen.to_pygame(drone_center), self.radius)
        # And center line
        pygame.draw.line(
            self.screen(), self.center_line_color, self.screen.to_pygame(center_line_start), self.screen.to_pygame(center_line_end), 2
        )

    def render_thruster(self, center_x, center_y, angle, width, height, power_ratio, color):
        power_x, power_y = self.screen.to_pygame((
            center_x - ((height / 2) + power_ratio * self.power_bar_height / 2) * np.sin(self.drone.angle + angle), 
            center_y + ((height / 2) + power_ratio * self.power_bar_height / 2) * np.cos(self.drone.angle + angle)
        ))
        center_x, center_y = self.screen.to_pygame((center_x, center_y))
        thruster = Rectangle(center_x, center_y, width, height, self.screen, color)
        thruster_power = Rectangle(
            power_x,
            power_y,
            2, 
            power_ratio * self.power_bar_height, 
            self.screen, 
            (255, 255, 0)
        )
        thruster.display(self.drone.angle + angle)
        thruster_power.display(self.drone.angle + angle)

    def render_left(self):
        # Can pass this in
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
        # Can pass this in
        drone_x, drone_y = self.drone.position[0], self.drone.position[1]

        right_thruster_x = drone_x + self.thruster_offset * np.cos(self.drone.angle)
        right_thruster_y = drone_y + self.thruster_offset * np.sin(self.drone.angle)
        self.render_thruster(
            right_thruster_x,
            right_thruster_y,
            self.drone.left.get_angle(),
            self.thruster_width,
            self.thruster_height,
            self.drone.right.power_ratio,
            (255, 0, 0),
        )

if __name__ == "__main__":
    pygame.init()
    screen = Screen()
    clock = pygame.time.Clock()

    drone = DroneDynamics()
    drone.position = np.array([screen.width / 2, screen.height / 2])
    drone.angle = 0
    drone_renderer = DroneRenderer(drone, screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        drone.set([-1, 0, -1, 0])

        # Get the state of all keyboard keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            drone.set([1, 0, 1, 0])
        if keys[pygame.K_LEFT]:
            drone.set([1, 0, -1, 0])
        if keys[pygame.K_RIGHT]:
            drone.set([-1, 0, 1, 0])

        screen().fill((0, 0, 0))

        drone.update(0.01)

        drone_renderer.render()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()