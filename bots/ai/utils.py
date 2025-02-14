"""
This file provides important constants, as well as the Utils class, which bundles all utility functions in a static class
"""

import numpy as np

# Set this to False when shipping
DEBUG = True

# Constants for tiles types
EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2
# The UNKNOWN type is used for both exploration and tile type (see Observation class)
UNKNOWN = -1

# Actions are represented by a number from 0 to 5
CENTER = 0
LEFT = 4
DOWN = 3
RIGHT = 2
UP = 1
SAP = 5


class Utils:
    @staticmethod
    def match(
        old: np.ndarray, old_mask: np.ndarray, new: np.ndarray, new_mask: np.ndarray
    ) -> int:
        """
        This function is used to determine the movement direction of asteroid tiles
        It tries out all possible directions of movement, and returns the one that matches
        """
        for i in [-1, 0, 1]:
            # Move both mask and table in direction i
            arr = np.roll(old, (i, -i), axis=(1, 0))
            mask = np.roll(old_mask, (i, -i), axis=(1, 0))
            # If the direction is top-right, mark bottom row and left column as unknown
            if i == 1:
                mask[-1, :] = False
                mask[:, 0] = False
            # If the direction is bottom-left, mark top row and right column as unknown
            if i == -1:
                mask[0, :] = False
                mask[:, -1] = False
            # If the direction is 0, do nothing
            mask = np.logical_and(mask, new_mask)
            # Checks if all visible tiles match
            if np.all(arr[mask] == new[mask]):
                return i
        # Debug code: if no direction is fitting, something went wrong
        if DEBUG:
            old.tofile("debug/old.txt", sep=" ")
            new.tofile("debug/new.txt", sep=" ")
            old_mask.tofile("debug/old_mask.txt", sep=" ")
            new_mask.tofile("debug/new_mask.txt", sep=" ")
        raise Exception("No matching shift")

    @staticmethod
    def dist(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """
        Returns Manhattan distance from pos1 to pos2
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return np.abs(x1 - x2) + np.abs(y1 - y2)

    @staticmethod
    def direction(f: tuple[int, int], t: tuple[int, int]) -> tuple[int, int]:
        """
        Naive method for choosing direction toward goal(from f to t):
        return a tuple of the two directions in which we have to move,
        ordered by which difference is greater
        This should be replaced by the A-star pathfinder
        """
        xf, yf = f
        xt, yt = t
        dx, dy = xt - xf, yt - yf
        xdir = RIGHT if dx > 0 else LEFT if dx < 0 else 0
        ydir = DOWN if dy > 0 else UP if dy < 0 else 0
        if np.abs(dx) > np.abs(dy):
            return xdir, ydir
        return ydir, xdir

    @staticmethod
    def move(pos: tuple[int, int], d: int) -> tuple[int, int]:
        """
        Returns the resulting position after executing action d at position pos
        """
        x, y = pos
        x, y = int(x), int(y)
        if d == CENTER:
            return x, y
        elif d == LEFT:
            return x - 1, y
        elif d == DOWN:
            return x, y + 1
        elif d == RIGHT:
            return x + 1, y
        elif d == UP:
            return x, y - 1
        raise ValueError("Invalid direction")

    @staticmethod
    def in_bounds(square: tuple[int, int]) -> bool:
        """
        Return true iff the position is in bounds
        """
        x, y = square
        return 0 <= x < 24 and 0 <= y < 24

    @staticmethod
    def squared_dist(f: tuple[int, int], t: tuple[int, int]) -> int:
        """
        Returns the squared Euclidean distance between two points
        """
        xf, yf = f
        xt, yt = t
        return (xt - xf) ** 2 + (yt - yf) ** 2

    @staticmethod
    def tofile(filename: str, arr: np.ndarray):
        """
        Debug function used to save a numpy array to a file
        """
        np.savetxt(filename, arr, fmt="%s")
