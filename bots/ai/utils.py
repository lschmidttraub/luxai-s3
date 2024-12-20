import numpy as np

EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

CENTER = 0
LEFT = 4
DOWN = 3
RIGHT = 2
UP = 1
SAP = 5


def match(
    old: np.ndarray, old_mask: np.ndarray, new: np.ndarray, new_mask: np.ndarray
) -> int:
    for i in range(-1, 2):
        arr = np.roll(old, (i, -i), axis=(1, 0))
        mask = np.roll(old_mask, (i, -i), axis=(1, 0))
        mask = mask & new_mask
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        if np.all(arr[mask] == new[mask]):
            return i
    old.tofile("old.txt", sep=" ")
    new.tofile("new.txt", sep=" ")
    old_mask.tofile("old_mask", sep=" ")
    new_mask.tofile("new_mask", sep=" ")
    raise Exception("No matching shift")


def dist(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    x1, y1 = pos1
    x2, y2 = pos2
    return np.abs(x1 - x2) + np.abs(y1 - y2)


def direction(f: tuple[int, int], t: tuple[int, int]) -> tuple[int, int]:
    xf, yf = f
    xt, yt = t
    dx, dy = xt - xf, yt - yf
    xdir = RIGHT if dx > 0 else LEFT if dx < 0 else 0
    ydir = DOWN if dy > 0 else UP if dy < 0 else 0
    if np.abs(dx) > np.abs(dy):
        return xdir, ydir
    return ydir, xdir


def move(pos: tuple[int, int], d: int) -> tuple[int, int]:
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


def in_bounds(square: tuple[int, int]) -> bool:
    x, y = square
    return 0 <= x < 24 and 0 <= y < 24


def shortest_path(pos: tuple[int, int], vision: np.ndarray) -> list[tuple[int, int]]:
    return []
