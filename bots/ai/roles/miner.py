from roles.unit import Unit
import numpy as np
from utils import *


class Miner(Unit):
    def choose_action(self, actions: np.ndarray) -> None:
        relic_nodes = self.obs.relic_nodes
        for u_id in self.units:
            pos, _ = self.obs.units[u_id]
            if not len(relic_nodes):
                raise Exception("Use of miner units without any discovered relics")
            # not very elegant, would be nice if could fix but accessing the first element of set
            # with an iterator is worse imo
            m_relic = None
            m_dist = 0
            for relic in relic_nodes:
                if m_relic is None or dist(pos, relic) < m_dist:
                    m_relic = relic
                    m_dist = dist(pos, m_relic)
            if m_relic is None:
                raise Exception(
                    "Use of miners without relics (this should have been have been checked earlier)"
                )
            dirs = direction(pos, m_relic)
            if m_dist <= 4:
                # needs to be fixed asap
                actions[u_id][0] = np.random.randint(0, 5)
            else:
                actions[u_id][0] = self.choose_dir(pos, dirs)
