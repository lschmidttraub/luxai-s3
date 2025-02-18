"""
Unit and Units are abstract classes meant to serve as templates for each respective role.
The Unit class handles the information linked to a single unit: position, energy, future actions etc...
The Units class handles a list of units, namely all units of a given role, and coordinates them to 
act in a coherent way
"""

from abc import ABC, abstractmethod
import numpy as np
from observation import Observation
from utils import *
from pathfinding import PathFinding


class Unit(ABC):
    def __init__(
        self,
        u_id: int,
        pos: tuple[int, int],
        energy: int,
        obs: Observation,
        target: tuple[int, int] | None = None,
    ):
        # a more accurate name might be future_move_actions, since this array only tracks movement
        # actions used to get to a particular target
        self.future_actions = []
        self.id = u_id
        self.pos = pos
        self.energy = energy
        self.obs = obs
        self.target = target

    def next_action(self) -> int:
        """
        returns the next movement action (assumes that we want to move, and that
        we have a target as well as a non-empty list of future actions)
        """
        if not self.future_actions:
            raise Exception(
                f"Called next_action on unit with no future actions : {self.pos} : {self.target}"
            )
        if self.energy < self.obs.move_cost:
            # If the unit can't afford to move, we simply don't move
            with open("center.txt", "a") as file:
                file.write(
                    f"{self.obs.step} : {self.id} : {self.obs.move_cost} : {self.energy}"
                )
            return CENTER
        # the next movement action will simply be the last element of the future_actions list
        action = self.future_actions.pop()
        # update the position of the unit after movement
        self.pos = Utils.move(self.pos, action)
        return action

    def action_invalid(self) -> bool:
        """
        return True iff the future_actions list has to recomputed, i.e. if future_actions is empty, the next actions in CENTER,
        or the next field is an asteroid tile
        """
        return (
            self.target is None
            or not self.future_actions
            or self.future_actions[-1] == CENTER
            or self.obs.vision[Utils.move(self.pos, self.future_actions[-1])]
            == ASTEROID_TILE
        )


class Units(ABC):
    def __init__(self, obs: Observation):
        # We purposefully omit the type of list entries, to avoid having to use generics (duck typing)
        self.units: list = []
        self.obs = obs
        # penalty for taking unknown tiles during pathfinding
        self.unknown_penalty: int = 3

    @abstractmethod
    def choose_actions(self, actions: np.ndarray) -> None:
        """
        Takes an array of actions and fills the corresponding to its units with appropriate actions
        """
        pass

    def calc_future_actions(self, unit: Unit) -> None:
        """
        Calculates the future_actions attribute of unit
        """
        path = self.calc_path(unit)
        unit.future_actions = []
        if path is None:
            # if no path is available or the unit is on its target, do nothing
            unit.future_actions = []
        elif not path:
            unit.future_actions = [CENTER]
        else:
            curr_square = path.pop()
            while path:
                next_square = path.pop()
                # to figure out which action is necessary to go from one cell to the next,
                # we just take the first component of the direction function,
                # since the difference in the other direction will be 0
                unit.future_actions.append(Utils.direction(curr_square, next_square)[0])
                curr_square = next_square
            # we once again prefer to store future actions in reversed order (more efficient popping)
            unit.future_actions.reverse()

    def update_units(self, units: list) -> None:
        """
        Updates the list of units
        """
        self.units = units

    def calc_path(self, unit: Unit) -> list[tuple[int, int]] | None:
        """
        Calculates the path a unit must take to get to its target
        """
        if unit.target is None:
            raise Exception("Unit has no target")

        return PathFinding.A_star(
            unit.pos,
            unit.target,
            self.obs.vision,
            self.obs.step,
            self.obs.drift_dir,
            self.obs.drift_steps,
            self.unknown_penalty,
        )

    def partition(self) -> dict:
        """
        Partitions the grid in a Voronoi-esque partition, where each position is associated to the closest unit
        """
        if not self.units:
            return {}
        partition = {unit.id: [] for unit in self.units}

        def closest_unit(pos: tuple[int, int]) -> int:
            m_unit = self.units[0]
            for unit in self.units[1:]:
                if Utils.dist(pos, unit.pos) < Utils.dist(pos, m_unit):
                    m_unit = unit
            return m_unit.id

        for x in range(24):
            for y in range(24):
                partition[closest_unit((x, y))].append((x, y))

        return partition
