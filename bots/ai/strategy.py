import numpy as np
from observation import Observation
from utils import *
from roles.scout import Scout
from roles.attacker import Attacker
from roles.miner import Miner
from roles.unit import Unit


class Strategy:
    def __init__(self, observation: Observation):
        self.obs = observation
        H, W = self.obs.H, self.obs.W
        self.relic_tile_mask: np.ndarray = np.zeros((W, H)).astype(bool)
        self.relic_tile_probs: np.ndarray = np.zeros((W, H))
        # create arrays of different roles
        self.unit_roles: dict = {
            "scout": Scout(self.obs),
            "attacker": Attacker(self.obs),
            "miner": Miner(self.obs),
        }
        self.all_relics_discovered = False

    def choose_action(self) -> np.ndarray:
        self.update_roles()
        actions = np.zeros((self.obs.max_units, 3), dtype=int)
        self.unit_roles["scout"].choose_action(actions)
        self.unit_roles["attacker"].choose_action(actions)
        self.unit_roles["miner"].choose_action(actions)

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
                        square = (x + m * i, y + n * j)
                        if in_bounds(square) and self.obs.exploration[square] == -1:
                            return direction(pos, square)

        return direction(pos, nearest_unexplored)

    def choose_dir(self, pos: tuple[int, int], d: tuple[int, int]) -> int:
        vision = self.obs.vision
        if vision.shape != (24, 24):
            raise Exception("graalhhh")
        sq1 = move(pos, d[0])
        sq2 = move(pos, d[1])
        if in_bounds(sq1) and vision[sq1] != ASTEROID_TILE:
            return d[0]
        elif in_bounds(sq2) and vision[sq2] != ASTEROID_TILE:
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
        if not n_units:
            self.unit_roles["scout"].update_units([])
            self.unit_roles["attacker"].update_units([])
            self.unit_roles["miner"].update_units([])
            return

        # these formulas are probably super shitty, need to improve
        scout_prop = self.obs.undiscovered_count() / (self.obs.H * self.obs.W)
        attacker_prop = len(self.obs.enemy_units) / n_units * 1 / 3
        miner_prop = (
            (len(self.obs.relic_nodes) + len(self.obs.relic_tiles))
            * 1
            / self.obs.max_units
        )
        total = scout_prop + attacker_prop + miner_prop
        scout_prop /= total
        attacker_prop /= total
        miner_prop /= total

        # rounding products to the nearest integer (instead of rounding down) helps us
        # avoid the edge case where n_miners = 1 but no relic nodes have been discovered (inshallah)
        n_scouts = int(round(n_units * scout_prop))
        n_attackers = int(round(n_units * attacker_prop))
        n_miners = n_units - n_scouts - n_attackers

        scouts = self.unit_roles["scout"].units
        attackers = self.unit_roles["attacker"].units
        miners = self.unit_roles["miner"].units

        def remove_dead_units(l, expected_length):
            for u_id in l:
                if not u_id in self.obs.units:
                    l.remove(u_id)

            while len(l) > expected_length:
                l.pop(-1)

        remove_dead_units(scouts, n_scouts)
        remove_dead_units(attackers, n_attackers)
        remove_dead_units(miners, n_miners)

        def add_new_units(l, expected_length):
            for u_id in self.obs.units:
                if len(l) == expected_length:
                    return
                if not (u_id in scouts or u_id in attackers or u_id in miners):
                    l.append(u_id)

        add_new_units(scouts, n_scouts)
        add_new_units(attackers, n_attackers)
        add_new_units(miners, n_miners)

        self.unit_roles["scout"].update_units(scouts)
        self.unit_roles["attacker"].update_units(attackers)
        self.unit_roles["miner"].update_units(miners)

    def eval(self) -> float:
        return 0
