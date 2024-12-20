from roles.unit import Unit
import numpy as np
from utils import *


class Attacker(Unit):
    def choose_action(self, actions: np.ndarray) -> None:
        # this is still a very naive approach: simply attack the first enemy you see
        # ideas for improvement: create a map of enemy units, to track clusters + split attackers according to cluster sizes
        for u_id in self.units:
            if not len(self.obs.enemy_units):
                raise Exception("Assigned attacker without any enemy units")
            pos, energy = self.obs.units[u_id]

            # units_in_range[u_id].append(e_pos)
            # attackers[e_id].append(u_id)
            has_target = False
            for _, (e_pos, e_energy) in self.obs.enemy_units.items():
                if dist(pos, e_pos) == 1 and energy > e_energy:
                    actions[u_id][0] = direction(pos, e_pos)
                    has_target = True
                if dist(pos, e_pos) <= self.obs.sap_range:
                    has_target = True
                    if energy < self.obs.sap_cost:
                        actions[u_id][0] = self.find_closest_energy_square(pos)
                    else:
                        e_x, e_y = e_pos
                        actions[u_id] = [5, e_x, e_y]
                    break

            if not has_target:
                m_pos = None
                m_dist = 0
                for _, (e_pos, e_energy) in self.obs.enemy_units.items():
                    if m_pos is None or dist(pos, e_pos) < m_dist:
                        m_dist = dist(pos, e_pos)
                        m_pos = e_pos
                if m_pos is None:
                    raise Exception(
                        "Assigned attacker without any enemy units (should have been checked earlier)"
                    )
                actions[u_id][0] = self.choose_dir(pos, direction(pos, m_pos))

    def find_closest_energy_square(self, pos: tuple[int, int]) -> int:
        W, H = self.obs.W, self.obs.H
        x, y = pos
        m_dist = max(W, H)
        for d in range(m_dist):
            for i in range(d + 1):
                j = d + 1 - i
                for m in [-1, 1]:
                    for n in [-1, 1]:
                        square = (x + m * i, y + n * j)
                        if (
                            in_bounds(square)
                            and self.obs.exploration[square] == 0
                            and self.obs.energy[square] > 0
                        ):
                            return self.choose_dir(pos, direction(pos, square))
        return np.random.randint(0, 5)
