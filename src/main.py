from screen import Screen
from drone import Drone
import numpy as np
import constants
import pygame

def main():
    pygame.init()
    screen = Screen()
    clock = pygame.time.Clock()

    drone = Drone(
        screen=screen,
        radius=constants.DRONE_RADIUS, 
        position=np.array([screen.width / 2, screen.height / 2])
    )

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
        drone.render()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()