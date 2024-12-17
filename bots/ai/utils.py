import numpy as np

EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

CENTER = 0
LEFT = 1
DOWN = 2
RIGHT = 3
UP = 4


def match(old, mask, new) -> int:
    dirs = [0, 1, -1]
    for i in dirs:
        arr = np.roll(old, (i, -i), axis=(1, 0))
        mask = old.mask | new.mask
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        if np.all(arr[~mask] == new[~mask]):
            return i
    return 0


def dist(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    x1, y1 = pos1
    x2, y2 = pos2
    return np.abs(x1 - x2) + np.abs(y1 - y2)


def direction(f: tuple[int, int], t: tuple[int, int]) -> tuple[int, int]:
    if f == t:
        return 0, 0
    xf, yf = f
    xt, yt = t
    dx, dy = xt - xf, yt - yf
    xdir = 2 if dx > 0 else 4 if dx < 0 else 0
    ydir = 3 if dy > 0 else 1 if dy < 0 else 0
    if np.abs(xdir) > np.abs(ydir):
        return xdir, ydir
    return ydir, xdir


def move(pos: tuple[int, int], d: int) -> tuple[int, int]:
    x, y = pos
    if d == CENTER:
        return pos
    elif d == LEFT:
        return x, y - 1
    elif d == DOWN:
        return x + 1, y
    elif d == RIGHT:
        return x, y + 1
    elif d == UP:
        return x - 1, y
    raise ValueError("Inputted and invalid direction")
