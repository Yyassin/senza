import pygame 

class Screen():
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))

    def __call__(self):
        return self.screen

    def to_pygame(self, coords):
        """Convert coordinates into pygame coordinates (lower-left => top left)."""
        return (coords[0], self.height - coords[1])