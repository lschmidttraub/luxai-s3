import numpy as np
from observation import Observation
from utils import *
from roles.scout import Scout
from roles.attacker import Attacker
from roles.miner import Miner


class Strategy:
    def __init__(self, observation: Observation):
        self.obs = observation
        H, W = self.obs.H, self.obs.W
        self.relic_tile_mask: np.ndarray = np.zeros((W, H)).astype(bool)
        self.relic_tile_probs: np.ndarray = np.zeros((W, H))
        # create arrays of different roles
        self.scouts: list[int] = []
        self.attackers: list[int] = []
        self.miners: list[int] = []
        self.unit_strats = dict
        self.all_relics_discovered = False

    def choose_action(self) -> np.ndarray:
        units = self.obs.units
        relic_nodes = self.obs.relic_nodes
        vision = self.obs.vision
        actions = self.choose_sap()
        for u_id, (pos, energy) in units.items():
            if actions[u_id][0]:
                continue
            if len(relic_nodes) > 0:
                m_relic = next(iter(relic_nodes))
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
        W, H = self.obs.W, self.obs.H
        nearest_unexplored = (W // 2, H // 2)
        x, y = pos
        m_dist = max(W, H)
        for d in range(m_dist):
            for i in range(d + 1):
                j = d + 1 - i
                for m in [-1, 1]:
                    for n in [-1, 1]:
                        a, b = x + m * i, y + n * j
                        if in_bounds(a, b) and self.obs.exploration[(a, b)] == -1:
                            return direction(pos, (a, b))

        return direction(pos, nearest_unexplored)

    def choose_dir(self, pos: tuple[int, int], d: tuple[int, int]) -> int:
        vision = self.obs.vision
        if vision.shape != (24, 24):
            raise Exception("graalhhh")
        sq1 = move(pos, d[0])
        sq2 = move(pos, d[1])
        if in_bounds(sq1[0], sq1[1]) and vision[sq1] != ASTEROID_TILE:
            return d[0]
        elif in_bounds(sq2[0], sq2[1]) and vision[sq2] != ASTEROID_TILE:
            return d[1]
        return 0

    def choose_sap(self) -> np.ndarray:
        actions = np.zeros((self.obs.max_units, 3), dtype=int)
        units_in_range = [[]] * self.obs.max_units
        # attackers = [[]] * self.obs.max_units
        for u_id, (pos, energy) in self.obs.units.items():
            if energy < self.obs.sap_cost:
                continue
            for e_id, (e_pos, e_energy) in self.obs.enemy_units.items():
                if dist(e_pos, pos) <= self.obs.sap_range:
                    actions[u_id] = [5, e_pos[0], e_pos[1]]
                    # units_in_range[u_id].append(e_pos)
                    # attackers[e_id].append(u_id)

        return actions

    def update_roles(self):
        if not self.all_relics_discovered and self.obs.found_all_relics():
            self.all_relics_discovered = True

        n_units = len(self.obs.units)
        n_scouts = 0 if self.all_relics_discovered else n_units // 3
        n_attackers = n_units // 3
        n_miners = n_units - n_scouts - n_attackers

        def remove_dead_units(l, expected_length):
            for i, item in enumerate(l):
                if not item in l:
                    self.scouts.remove(i)

            while len(l) > expected_length:
                l.remove(-1)

        remove_dead_units(self.scouts, n_scouts)
        remove_dead_units(self.attackers, n_attackers)
        remove_dead_units(self.miners, n_miners)

        def add_new_units(l, expected_length):
            for u_id in self.obs.units:
                if len(l) == expected_length:
                    return
                if not (
                    u_id in self.scouts or u_id in self.attackers or u_id in self.miners
                ):
                    l.append(u_id)

        add_new_units(self.scouts, n_scouts)
        add_new_units(self.attackers, n_attackers)
        add_new_units(self.miners, n_miners)

    def eval(self) -> float:
        return 0
