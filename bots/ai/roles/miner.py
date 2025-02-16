from pathfinding import PathFinding
from roles.unit import Units, Unit
import numpy as np
from utils import *
from observation import Observation
from pathfinding import PathFinding


class Miner(Unit):
    def __init__(
        self,
        u_id: int,
        pos: tuple[int, int],
        energy: int,
        target: tuple[int, int] | None = None,
    ):
        super().__init__(u_id, pos, energy, target)


class Miners(Units):
    def __init__(self, obs: Observation):
        super().__init__(obs)
        self.unit_search_radius = 10
        self.unit_targets = set()
        self.relic_tile_probs: np.ndarray

    def choose_actions(self, actions: np.ndarray) -> None:
        if len(self.units) and not len(self.obs.relic_nodes):
            raise Exception("Use of miner units without any discovered relics")
        # we exclude the tiles that the units are already on, to avoid units constantly
        # exchanging each others tiles
        self.unit_targets = set(
            [unit.pos for unit in self.units] + [unit.target for unit in self.units]
        )
        needs_target = np.ones(len(self.units), dtype=bool)
        while needs_target.any():
            for unit in needs_target:
                pass
        for unit in self.units:
            # the idea is that we want to move units to the tiles with the highest probabilities of being
            # relic tiles, but we don't want them all to go to the same tile
            unreachable = set()
            new_target = self.max_prob_in_range(
                unit, self.unit_search_radius, unreachable
            )
            # if the target is the same and the unit already has a path of action, we don't need
            # to recompute the future actions
            while unit.pos != new_target and unit.future_actions is None:
                unreachable.add(new_target)
                new_target = self.max_prob_in_range(
                    unit, self.unit_search_radius, unreachable
                )
            else:
                unit.target = new_target
                self.calc_future_actions(unit)
            actions[unit.id][0] = unit.next_action(self.obs)
            self.unit_targets.add(new_target)

    def update_relic_probs(self, new_probs: np.ndarray) -> None:
        self.relic_tile_probs = new_probs

    def distribute_relics(self) -> None:
        """
        We don't want to send all nodes to the same relic tile, so we use this function
        INCOMPLETE
        """
        relics = self.obs.relic_nodes

    def closest_relic_node(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        returns the closest relic node
        """
        m_relic = None
        m_dist = 0
        for relic in self.obs.relic_nodes:
            if m_relic is None or Utils.dist(pos, relic) < m_dist:
                m_relic = relic
                m_dist = Utils.dist(pos, m_relic)
        if m_relic is None:
            raise Exception(
                "Use of miners without relics (this should have been have been checked earlier)"
            )
        return m_relic

    def max_prob_in_range(
        self, unit: Unit, radius: int, unreachable: set
    ) -> tuple[int, int]:
        """
        Returns the square in a certain perimeter around pos with the maximum probability of being a relictile,
        and that also isn't already selected as a target by other tiles
        """
        mask = Utils.position_mask(unit.pos, radius) + (
            [unit.target] if unit.target is not None else []
        )
        # The unit should always have prioritary access to the tile it is currently on as a target
        m_pos = unit.pos
        for pos in PathFinding.get_neighbors(unit.pos, self.obs.vision):
            if (
                not pos in unreachable
                and self.relic_tile_probs[pos] < self.relic_tile_probs[m_pos]
            ):
                m_pos = pos
        if self.relic_tile_probs[m_pos] > 0.025:
            return m_pos

        for p in mask:
            if (
                not p in self.unit_targets
                and not p in unreachable
                and self.relic_tile_probs[p] < self.relic_tile_probs[m_pos]
            ):
                m_pos = p
        return m_pos
