import numpy as np
from observation import Observation
from utils import *


class Strategy:
    def __init__(self, observation: Observation):
        self.obs = observation
        self.relic_tile_mask: np.ndarray = np.zeros((24, 24)).astype(bool)
        self.relic_tile_probs: np.ndarray = np.zeros((24, 24))

    def choose_action(self) -> np.ndarray:
        actions = np.zeros((self.obs.params["max_units"], 3), dtype=int)
        units = self.obs.units
        relic_nodes = self.obs.relic_nodes
        vision = self.obs.vision
        for u_id, (pos, energy) in units.items():
            if len(relic_nodes) > 0:
                m_relic = relic_nodes[0]
                m_dist = 100
                for relic in relic_nodes:
                    if dist(pos, relic) < m_dist:
                        m_relic = relic
                        m_dist = dist(pos, m_relic)
                dirs = direction(pos, m_relic)
                if m_dist <= 4:
                    actions[u_id][0] = np.random.randint(0, 5)
                else:
                    actions[u_id][0] = self.choose_dir(pos, dirs)
            else:
                actions[u_id][0] = self.choose_dir(pos, self.explore_dir(pos))

        return actions

    def update_potential_relic_tiles(self):
        pt_diff = self.obs.pt_diff
        pos1 = [pos for _, (pos, e) in self.obs.units.items()]
        pos2 = [pos for _, (pos, e) in self.obs.enemy_units.items()]
        if ~pt_diff[0]:
            self.relic_tile_mask[pos1] = True
        if ~pt_diff[1]:
            self.relic_tile_mask[pos2] = True

    def explore_dir(self, pos: tuple[int, int]) -> tuple[int, int]:
        nearest_unexplored = (12, 12)
        x, y = pos
        for i in range(10):
            for j in range(10):
                for m in [-1, 1]:
                    for n in [-1, 1]:
                        a, b = x + m * i - 10, y + n * j - 10
                        if in_bounds(a, b) and self.obs.exploration[(a, b)] == -1:
                            return direction(pos, (a, b))

        return direction(pos, (12, 12))

    def choose_dir(self, pos: tuple[int, int], d: tuple[int, int]) -> int:
        vision = self.obs.vision
        if vision.shape != (24, 24):
            raise Exception("graalhhh")
        square = move(pos, d[0])
        if not isinstance(square, tuple):
            raise Exception("invalid move: ", square)
        if vision[square] != ASTEROID_TILE:
            return d[0]
        elif vision[(0, 0)] != ASTEROID_TILE:
            return d[1]
        return 0

    def eval(self) -> float:
        return 0
