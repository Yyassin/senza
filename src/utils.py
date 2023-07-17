"""
Generally useful utility functions.
"""


def normalize_angle(angle):
    """
    Normalizes the provided angle
    into the range [-180, 180].

    :param angle: The angle to normalize.
    """
    normalized_angle = ((angle + 180) % 360) - 180
    normalized_angle /= 180
    return normalized_angle
