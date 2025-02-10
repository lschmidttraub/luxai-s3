import numpy as np
from utils import *
from heapq import heappush, heappop
from collections import defaultdict


class PathFinding:
    def __init__(self, W, H):
        self.cells
        self.drift_dir = 0
        self.drift_steps = 0
        self.W = W
        self.H = H

    def update_map(self, new_map: np.ndarray):
        self.cells: np.ndarray = new_map

    def dynamic_Astar(
        self, start: tuple[int, int], goal: tuple[int, int], step: int
    ) -> list[tuple[int, int]]:
        heap = []

        for i in range(self.W):
            for j in range(self.H):
                if self.cells[(i, j)] != -1:
                    heappush(heap, (dist((i, j), start), (i, j)))
        if self.cells[start] == -1:
            heappush(heap, (0, start))

        def heuristic(self, pos: tuple[int, int]):
            return dist(pos, goal)

        def is_accessible(square):
            return self.cells[square] != ASTEROID_TILE

        parent = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0
        while heap:
            _, pos = heappop(heap)
            if pos == goal:
                return self.reconstruct_path(parent, pos)
            if self.drift_steps and not step % self.drift_steps:
                self.cells = np.roll(
                    self.cells, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                # we assume rollover tiles are asteroids to avoid any mistakes
                if self.drift_dir == 1:
                    self.cells[-1, :] = ASTEROID_TILE
                    self.cells[:, 0] = ASTEROID_TILE
                elif self.drift_dir == -1:
                    self.cells[0, :] = ASTEROID_TILE
                    self.cells[:, -1] = ASTEROID_TILE

        return []

    def get_neighbors(self, pos):
        x, y = pos
        return [
            (x + i, y + j)
            for i in [-1, 1]
            for j in [-1, 1]
            if in_bounds((x + i, y + j)) and self.cells[i][j] != ASTEROID_TILE
        ]

    def reconstruct_path(self, parent, cell) -> list:
        path = []
        while cell in parent:
            path.append(cell)
            cell = parent[cell]
        # it's more convenient to store path in reversed path
        return path


class Cell:
    def __init__(self, pos: tuple[int, int], is_asteroid=False):
        self.x, self.y = pos
        self.accessible = ~is_asteroid
        self.f = float("inf")
        self.g = float("inf")
