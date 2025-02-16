""" 
The job of scout units is to explore the map
"""

from roles.unit import Unit, Units
import numpy as np
from utils import *
from observation import Observation


class Scout(Unit):
    pass


class Scouts(Units):
    def __init__(self, obs: Observation):
        super().__init__(obs)
        # scouts prefer to explore unknown tiles so we choose a negative penalty
        self.unknown_penalty = -1

    def choose_actions(self, actions: np.ndarray) -> None:
        if len(self.units) and not self.obs.undiscovered_count():
            raise Exception(
                "Assigned scout role when all squares were explored",
                self.obs.undiscovered_count(),
            )
        for unit in self.units:
            # if the unit has no more actions left, or its target has already been explored,
            # we recompute its future_actions attribute
            unreachable = set()
            unit.target = self.closest_unexplored(unit.pos, unreachable)
            while not self.calc_path(unit):
                unreachable.add(unit.target)
                unit.target = self.closest_unexplored(unit.pos, unreachable)
            self.calc_future_actions(unit)
            if DEBUG:
                """with open("scout_actions.txt", "a") as file:
                file.write(
                    str(unit.id)
                    + " "
                    + str(unit.future_actions[-1])
                    + " "
                    + str(unit.pos)
                    + " "
                    + str(unit.target)
                    + "\n"
                )"""

            actions[unit.id][0] = unit.next_action(self.obs)

    def closest_unexplored(
        self, pos: tuple[int, int], unreachable: set
    ) -> tuple[int, int]:
        """
        Return the position of the closest unexplored tile to pos
        We also prefer to choose tiles that are closer to the center of the map
        """
        m_dist = 0
        m_pos = None
        middle = (self.obs.H // 2, self.obs.W // 2)
        # This could maybe be done more efficiently, but you usually want to
        # avoid for loops in python
        for idx in np.dstack(
            np.where(
                np.logical_and(self.obs.exploration == UNKNOWN, ~self.obs.nebula_mask)
            )
        )[0]:
            # we have to convert positions to tuples to avoid errors during the A-star algorithm
            tuple_idx = tuple(idx)
            if tuple_idx in unreachable:
                continue
            d = Utils.dist(pos, tuple_idx)
            # if unexplored unit is in sensor range, it is a nebula tile and should be ignored
            if (
                m_pos is None
                or self.obs.sensor_range < d < m_dist
                or (
                    d == m_dist
                    and Utils.squared_dist(tuple_idx, middle)
                    < Utils.squared_dist(m_pos, middle)
                )
            ):
                m_dist = d
                m_pos = tuple_idx
        if m_pos is None:
            raise Exception(
                "Assigned scout role when all squares were explored. explore ratio (Should have been checked earlier)"
            )

        return m_pos
