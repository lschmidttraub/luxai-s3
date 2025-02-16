"""
This file provides important constants, as well as the Utils class, which bundles all utility functions in a static class
"""

import numpy as np
import math

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
    def match_shift(
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
    def squared_dist(f: tuple[int, int], t: tuple[int, int]) -> int:
        """
        Returns the squared Euclidean distance between two points
        """
        xf, yf = f
        xt, yt = t
        return (xt - xf) ** 2 + (yt - yf) ** 2

    @staticmethod
    def euclidean_dist(f: tuple[int, int], t: tuple[int, int]) -> float:
        """
        Returns the Euclidean distance
        """
        return math.sqrt(Utils.squared_dist(f, t))

    @staticmethod
    def max_dist(f: tuple[int, int], t: tuple[int, int]) -> int:
        """
        Return the Chebyshev distance between two points
        """
        xf, yf = f
        xt, yt = t
        return max(np.abs(xf - xt), np.abs(yf - yt))

    @staticmethod
    def position_mask(pos: tuple[int, int], radius: int):
        """
        returns an array of all positions in a mask of width 2*radius+1 centered around pos
        """
        x, y = pos
        return [
            (i, j)
            for i in range(y - radius, y + radius + 1)
            for j in range(x - radius, x + radius + 1)
            if Utils.in_bounds((i, j))
        ]

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
    def symmetric(pos: tuple[int, int]):
        """
        Returns the position symmetric to pos relative to the board's axis of symmetry
        """
        x, y = pos
        return (23 - y, 23 - x)

    @staticmethod
    def add_sym(pos: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Adds the symmetric images of each position to the list and removes duplicates
        """
        return list(set(pos + list(map(Utils.symmetric, pos))))

    @staticmethod
    def is_symmetric(arr: np.ndarray) -> None:
        for i in range(24):
            for j in range(24):
                if not math.isclose(arr[i, j], arr[Utils.symmetric((i, j))]):
                    Utils.tofile("debug/probs.txt", arr)
                    raise Exception(
                        f"array not symmetric: {arr[i,j]}!={arr[Utils.symmetric((i, j))]}"
                    )

    @staticmethod
    def is_top_half(pos: tuple[int, int]) -> bool:
        x, y = pos
        return x + y <= 23

    @staticmethod
    def prob_mult(a: float, b: float) -> float:
        """
        Combines the probabilities of two overlapping relic tiles, by multiplying the probability that
        the tile isn't a relic tiles in both areas to find the new probability that the tile isn't a relic
        tile (this way we never get probabilities over 1)
        """
        return 1 - (1 - a) * (1 - b)

    @staticmethod
    def tofile(filename: str, arr: np.ndarray):
        """
        Debug function used to save a numpy array to a file
        """
        np.savetxt(filename, arr, fmt="%s")

    @staticmethod
    def poisson_binomial(P: np.ndarray, k: int) -> tuple[float, np.ndarray]:
        """
        returns probabilities of getting k given the i-th entry being positive for each 0<=i<n
        as well as probability of getting exactly k positive results
        DP[i, j, l] = probability of l positives for the first i tests, disregarding the j-th test
        """
        if P.ndim != 1:
            raise Exception("Invalid array dimension: ", P.shape)
        if k == 0:
            return 1, np.zeros(len(P))
        n = len(P)
        DP = np.zeros((n + 1, n + 1, k + 1))
        DP[0, :, 0] = 1
        for i in range(1, n + 1):
            for j in range(0, n + 1):
                if i - 1 == j:
                    DP[i, j, 0] = DP[i - 1, j, 0]
                else:
                    DP[i, j, 0] = DP[i - 1, j, 0] * (1 - P[i - 1])
                for l in range(1, k + 1):
                    if i - 1 == j:
                        DP[i, j, l] = DP[i - 1, j, l]
                    else:
                        DP[i, j, l] = (
                            DP[i - 1, j, l] * (1 - P[i - 1])
                            + DP[i - 1, j, l - 1] * P[i - 1]
                        )
        # we sometimes get a vanishingly small probability of getting exactly k positives
        # this is problematic, as we divide by said probability when using the Bayes formula
        # We thus select a small epsilon to avoid minuscule
        epsilon = 1e-4
        return max(DP[n, n, k], epsilon), DP[n, :-1, k - 1]

    @staticmethod
    def bernoulli_binomial(p: float, k: int, n: int) -> float:
        """
        Return the probability of get k in a binomial distribution with n runs
        """
        return math.comb(n, k) * p**k * (1 - p) ** (n - k)

    @staticmethod
    def bayes(a, b, b_given_a):
        """
        Bayes' formula
        """
        return b_given_a * a / b
