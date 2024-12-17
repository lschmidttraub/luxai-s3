import numpy as np

EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

CENTER = 0
LEFT = 4
DOWN = 3
RIGHT = 2
UP = 1


def match(old, mask, new) -> int:
    dirs = [0, 1, -1]
    for i in dirs:
        arr = np.roll(old, (i, -i), axis=(1, 0))
        mask = mask | new.mask
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


def in_bounds(x: int, y: int) -> bool:
    return x >= 0 and y >= 0 and x < 24 and y < 24

def shortest_path(pos:tuple[int,int], vision:np.ndarray)->list[tuple[int,int]]:
    return []
