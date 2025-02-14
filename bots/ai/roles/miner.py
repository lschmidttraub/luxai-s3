from roles.unit import Units, Unit
import numpy as np
from utils import *
from observation import Observation


class Miner(Unit):
    def __init__(
        self,
        u_id: int,
        pos: tuple[int, int],
        energy: int,
        target: tuple[int, int] | None = None,
    ):
        super().__init__(u_id, pos, energy, target)
        self.is_on_relic_tile = False


class Miners(Units):
    def __init__(self, obs: Observation):
        super().__init__(obs)

    def choose_actions(self, actions: np.ndarray) -> None:
        if len(self.units) and not len(self.obs.relic_nodes):
            raise Exception("Use of miner units without any discovered relics")
        for unit in self.units:
            # At the moment, this never comes into play, as we have yet to implement relic tile identification
            if unit.is_on_relic_tile:
                actions[unit.id][0] = CENTER
                continue

            if not unit.target or not unit.future_actions:
                unit.target = self.closest_relic_node(unit.pos)
                self.calc_future_actions(unit)

            # going all the way to a relic node has no use, so we move randomly once we are within the
            # perimeter where relic tiles can appear
            if Utils.dist(unit.pos, unit.target) <= 4:
                # needs to be fixed asap (i.e. we need to implement relic tile handling)
                actions[unit.id][0] = np.random.randint(0, 5)
            else:
                actions[unit.id][0] = unit.next_action(self.obs)

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
