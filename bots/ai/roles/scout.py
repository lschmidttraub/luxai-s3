""" 
The job of scout units is to explore the map
"""

from logging import debug
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
        self.unknown_penalty = 0

        # list of tiles to be taken as targets
        self.target_tiles = {(23, 0), (0, 23), (6, 17), (17, 6), (5, 5), (18, 18)}
        self.unit_targets = set()

    def choose_actions(self, actions: np.ndarray) -> None:
        if len(self.units) and not self.obs.undiscovered_count():
            raise Exception(
                "Assigned scout role when all squares were explored",
                self.obs.undiscovered_count(),
            )

        self.target_tiles = {
            target
            for target in self.target_tiles
            if self.obs.exploration[target] == UNKNOWN
        }

        self.unit_targets = {
            unit.target for unit in self.units if not unit.action_invalid()
        }

        for unit in self.units:
            """
            # if the unit has no more actions left, or its target has already been explored,
            # we recompute its future_actions attribute
            unreachable = set()
            unit.target = self.closest_unexplored(unit.pos, unreachable)
            while not self.calc_path(unit):
                unreachable.add(unit.target)
                unit.target = self.closest_unexplored(unit.pos, unreachable)
            self.calc_future_actions(unit)

            actions[unit.id][0] = unit.next_action(self.obs)
            """
            if unit.action_invalid() or self.obs.exploration[unit.target] != UNKNOWN:
                for tile in self.target_tiles:
                    if not tile in self.unit_targets:
                        unit.target = tile
                        self.calc_future_actions(unit)
                        if not unit.action_invalid():
                            self.unit_targets.add(tile)
                            if unit.target == unit.pos:
                                raise Exception(
                                    f"first {unit.pos} : {self.target_tiles}"
                                )
                            break
                if unit.action_invalid() or self.obs.exploration[unit.pos] != UNKNOWN:
                    unit.target = self.closest_unexplored(unit.pos)
                    self.calc_future_actions(unit)
                    if unit.target == unit.pos:
                        raise Exception(f"yep. {unit.pos} : {unit.target}")
                    if not unit.future_actions:
                        unit.future_actions = [CENTER]
            actions[unit.id][0] = unit.next_action()
            """
            if DEBUG:
                with open("debug/targets.txt", "a") as file:
                    file.write(
                        f"Scout : {self.obs.step} : {unit.id} : {unit.target} : {actions[unit.id][0]} : {unit.future_actions}\n"
                    )
                """

    def closest_unexplored(
        self, pos: tuple[int, int], unreachable: set = set()
    ) -> tuple[int, int]:
        """
        Return the position of the closest unexplored tile to pos
        We also prefer to choose tiles that are closer to the center of the map
        """
        m_dist = 0
        m_pos = None
        middle = (self.obs.H // 2, self.obs.W // 2)

        for p in np.ndindex(self.obs.shape):
            if (
                self.obs.exploration[p] == UNKNOWN
                and not self.obs.nebula_mask[p]
                and not p in unreachable
            ):
                d = Utils.dist(pos, p)
                # if unexplored unit is in sensor range, it is a nebula tile and should be ignored
                if (
                    m_pos is None
                    or d < m_dist
                    or (
                        d == m_dist
                        and Utils.squared_dist(p, middle)
                        < Utils.squared_dist(m_pos, middle)
                    )
                ):
                    m_dist = d
                    m_pos = p

        if pos == m_pos:
            Utils.heatmap("debug/exploration.jpg", self.obs.exploration)
            Utils.heatmap("debug/nebula.jpg", self.obs.nebula_mask)
            Utils.heatmap("debug/prev_mask.jpg", self.obs.prev_mask)

            raise Exception(
                f"unknown : {pos} : {self.obs.exploration[pos]} : {self.obs.nebula_mask[pos]}"
            )
        if m_pos is None:
            # If everything is explored, just explore the tiles with the oldest exploration value again
            m_score = 0
            for x, y in Utils.position_mask(pos, 4):
                if (x, y) != pos:
                    score = self.obs.exploration[x, y] / Utils.dist(pos, (x, y))
                    if score > m_score:
                        m_score = score
                        m_pos = (x, y)
        if m_pos is None:
            raise Exception(
                "Assigned scout role when all squares were explored. explore ratio (Should have been checked earlier)"
            )
        if pos == m_pos:
            Utils.tofile("debug/exploration.txt", self.obs.exploration)
            raise Exception("explored")

        return m_pos
