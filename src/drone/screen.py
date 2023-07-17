import pygame

"""
Wrapper around Pygame's screen
"""


class Screen:
    def __init__(self, width=640, height=480):
        """
        Wrapper around the pygame screen

        :param width: The screen width in pixels
        :param height: The screen height in pixels
        """
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))

    def __call__(self):
        """
        Returns a reference to the pygame screen
        """
        return self.screen

    def to_pygame(self, coords):
        """
        Converts regular 2D cartesian coordinates
        into pygame coordinates (lower-left => top left, and
        y increasing downwards).

        :param coords: A tuple (x, y) of the cartesian coordinate
        to convert.
        """
        return (coords[0], self.height - coords[1])
