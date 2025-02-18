from pathfinding import PathFinding
from roles.unit import Units, Unit
import numpy as np
from utils import *
from observation import Observation
from pathfinding import PathFinding


class Miner(Unit):
    pass


class Miners(Units):
    def __init__(self, obs: Observation):
        super().__init__(obs)
        self.unit_search_radius = 10
        self.relic_tile_probs: np.ndarray

    def choose_actions(self, actions: np.ndarray) -> None:
        if len(self.units) and not len(self.obs.relic_nodes):
            raise Exception("Use of miner units without any discovered relics")

        # weight of the probability (higher means that probability is taken more into account)
        alpha = 40

        def score(unit: Miner, pos: tuple[int, int]) -> float:
            # score function: smaller is better (so then it's easier to extract maximum)
            return Utils.dist(unit.pos, pos) - alpha * self.relic_tile_probs[pos]

        # of course it would be nice to not have to this every iteration, but this is fine for now
        needs_target = set(self.units)
        taken = set()
        scores = [
            (score(unit, (i, j)), unit, (i, j))
            for unit in needs_target
            for i, j in np.ndindex((24, 24))
            if self.relic_tile_probs[i, j] >= 0.1
        ]

        scores.sort(key=lambda x: x[0])

        # the idea is that we want to move units to the tiles with the highest probabilities of being
        # relic tiles, but we don't want them all to go to the same tile
        for _, unit, pos in scores:
            if unit in needs_target and pos not in taken:
                unit.target = pos
                self.calc_future_actions(unit)
                if not unit.action_invalid():
                    needs_target.remove(unit)
                    taken.add(pos)
                    if DEBUG:
                        with open("debug/targets.txt", "a") as file:
                            file.write(
                                f"Miner : {self.obs.step} : {unit.id} : {unit.target} : {unit.future_actions}\n"
                            )
                    actions[unit.id][0] = unit.next_action()

        """
        for unit in self.units:
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
        """

    def update_relic_probs(self, new_probs: np.ndarray) -> None:
        """
        We have to pass the relic tile probabilities this way, as importing the Strategy class would result in a cyclic import
        """
        self.relic_tile_probs = new_probs

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
        if self.relic_tile_probs[m_pos] > 0.02:
            return m_pos

        for p in mask:
            if (
                not p in unreachable
                and self.relic_tile_probs[p] < self.relic_tile_probs[m_pos]
            ):
                m_pos = p
        return m_pos
