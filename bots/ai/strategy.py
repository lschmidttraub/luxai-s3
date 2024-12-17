import numpy as np
from observation import Observation
from utils import *


class Strategy:
    def __init__(self, observation: Observation):
        self.obs = observation
        self.relic_tile_mask: np.ndarray = np.zeros((24, 24)).astype(bool)
        self.relic_tile_probs: np.ndarray = np.zeros((24, 24))

    def choose_action(self) -> list[list[int]]:
        actions = [[0] * 3] * 16
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
                action = 0
                if m_dist < 4:
                    action = np.random.randint(0, 5)
                elif vision[move(pos, dirs[0])] != ASTEROID_TILE:
                    action = dirs[0]
                elif vision[move(pos, dirs[1])] == ASTEROID_TILE:
                    action = dirs[1]
                actions[u_id][0] = action

        return actions

    def update_potential_relic_tiles(self):
        pt_diff = self.obs.pt_diff
        pos1 = [pos for _, (pos, e) in self.obs.units.items()]
        pos2 = [pos for _, (pos, e) in self.obs.enemy_units.items()]
        if ~pt_diff[0]:
            self.relic_tile_mask[pos1] = True
        if ~pt_diff[1]:
            self.relic_tile_mask[pos2] = True

    def eval(self) -> float:
        return 0
