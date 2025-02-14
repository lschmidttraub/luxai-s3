"""
Implementation of the A-star algorithm to find the best path a unit can 
take to get to its target location.
Technically, this isn't a total solution, as we don't allow for the unit 
to wait or access a cell at a later time, which might lead to a faster path, 
since come cells move.
However, I think it is more efficient to just compute a "naive" solution, and simply recompute later on if no solution if no path is found immediately.
"""

import numpy as np
from utils import *
from heapq import heappush, heappop
from collections import defaultdict


class PathFinding:
    @staticmethod
    def A_star(
        start: tuple[int, int],
        goal: tuple[int, int],
        start_map: np.ndarray,
        start_step: int = 0,
        drift_dir: int = 0,
        drift_steps: int = 0,
    ) -> list[tuple[int, int]] | None:
        """
        Return array of positions needed to go through to get to target
        """
        if start == goal:
            return []
        # The heap is a minHeap in which the value is the distance + some heuristic
        # and the key is a position and a step (to know when to shift)
        heap = []
        heappush(heap, (0, start, start_step))

        # heuristic function = Manhattan distance
        heuristic = lambda pos: Utils.dist(pos, goal)

        # Maintain parent dictionary for backtracking
        parent = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0

        # use set to remember which squares were already visited
        visited = set()
        while heap:
            _, pos, step = heappop(heap)
            if pos in visited:
                continue
            else:
                visited.add(pos)
            if pos == goal:
                return PathFinding.reconstruct_path(parent, pos)
            # Since tiles aren't necessarily processed in ascending step order,
            # we need to create a shifted copy of our map each time we pop from the heap
            if drift_steps:
                shift = (step // drift_steps - start_step // drift_steps) * drift_dir
            else:
                shift = 0
            shifted_map = np.roll(start_map, (shift, -shift), axis=(1, 0))
            # rollover tiles are UNKNOWN
            if drift_dir == 1:
                shifted_map[-shift:, :] = UNKNOWN
                shifted_map[:, : shift - 1] = UNKNOWN
            elif drift_dir == -1:
                shifted_map[: -shift - 1, :] = UNKNOWN
                shifted_map[:, shift:] = UNKNOWN

            for nb in PathFinding.get_neighbors(pos, shifted_map):
                if g_score[nb] == float("inf"):
                    g_score[nb] = g_score[pos] + 1
                    heappush(
                        heap,
                        (
                            g_score[nb] + heuristic(nb)
                            # Since we would rather not take unknown tiles, we incur a penalty to taking
                            # unknown tiles (this penalty can be changed, I just think 10 seems reasonable)
                            + (10 if shifted_map[nb] == UNKNOWN else 0),
                            nb,
                            step + 1,
                        ),
                    )
                    parent[nb] = pos
        # If the tile isn't accessible, we return None
        return None

    @staticmethod
    def get_neighbors(
        pos: tuple[int, int], shifted_map: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Returns accessible neighbors of the given cell.
        We avoid asteroid tiles
        """
        x, y = pos
        return [
            (x + i, y + j)
            for i, j in [(0, -1), (0, 1), (-1, 0), (1, 0)]
            if Utils.in_bounds((x + i, y + j))
            and shifted_map[x + i, y + j] != ASTEROID_TILE
        ]

    @staticmethod
    def reconstruct_path(parent: dict, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Returns list of positions used to get to cell, in reversed order
        """
        path = [cell]
        while cell in parent:
            cell = parent[cell]
            path.append(cell)
        # it's more convenient to store path in reversed order
        # as we can then pop from the back, which is faster
        return path
