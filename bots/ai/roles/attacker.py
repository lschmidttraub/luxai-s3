"""
The Attacker roles wants to use sap actions to kill enemy units
"""

from roles.unit import Unit, Units
import numpy as np
from utils import *


class Attacker(Unit):
    pass


class Attackers(Units):
    def choose_actions(self, actions: np.ndarray) -> None:
        # this is still a very naive approach: simply attack the first enemy you see
        # ideas for improvement: create a map of enemy units, to track clusters + split attackers according to cluster sizes
        if len(self.units) and not len(self.obs.enemy_units):
            raise Exception("Assigned attacker without any enemy units")
        for unit in self.units:
            # units_in_range[u_id].append(e_pos)
            # attackers[e_id].append(u_id)
            has_target = False
            # We probably should consider how much energy opposing units have, as well as to which tile they might move
            # (sap actions are computed after move actions)
            for _, (e_pos, e_energy) in self.obs.enemy_units.items():
                # check if the enemy unit is within range
                if Utils.dist(unit.pos, e_pos) <= self.obs.sap_range:
                    has_target = True
                    if unit.energy < self.obs.sap_cost:
                        # if we don't have enough energy, our goal becomes gathering energy
                        unit.target = self.find_closest_energy_square(unit.pos)
                        self.calc_future_actions(unit)
                        actions[unit.id][0] = unit.next_action(self.obs)

                    else:
                        # otherwise, attack
                        e_x, e_y = e_pos
                        actions[unit.id] = [5, e_x, e_y]
                    break
            # if no enemy unit is within range, move towards the closest enemy unit
            if not has_target:
                unit.target = self.find_closest_enemy(unit.pos)
                self.calc_future_actions(unit)
                actions[unit.id][0] = unit.next_action(self.obs)

    def find_closest_energy_square(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        if the unit is in a neihborhood with positive energy values, move to neihboring cell with highest
        energy level. Otherwise, find the closest cell with a positive energy value
        """
        m_dist = float("inf")
        x, y = pos
        m_pos = pos
        # find neighbor with highest energy level
        for i, j in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            if (
                Utils.in_bounds((x + i, y + j))
                and self.obs.energy[(x + i, y + j)] > self.obs.energy[m_pos]
            ):
                m_pos = (x + i, y + j)
        # If this energy level is positive, move to this tile
        if self.obs.energy[m_pos] > 0:
            return m_pos

        # Otherwise, we find the closest tile with positive energy
        for idx in np.where(self.obs.energy > 0):
            x, y = idx[0], idx[1]

            d = Utils.dist(pos, (x, y))
            if m_pos is None or d < m_dist:
                m_dist = d
                m_pos = (x, y)

        return m_pos

    def find_closest_enemy(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        returns the location of the closest enemy unit
        """
        m_pos = None
        m_dist = 0
        for _, (e_pos, _) in self.obs.enemy_units.items():
            if m_pos is None or Utils.dist(pos, e_pos) < m_dist:
                m_dist = Utils.dist(pos, e_pos)
                m_pos = e_pos
        if m_pos is None:
            raise Exception(
                "Assigned attacker without any enemy units (should have been checked earlier)"
            )
        return m_pos
