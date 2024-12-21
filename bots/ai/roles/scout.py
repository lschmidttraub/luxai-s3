from roles.unit import Unit
import numpy as np
from utils import *


class Scout(Unit):
    def choose_action(self, actions: np.ndarray) -> None:
        # Ideally we would implement some variation on A-star to find the shortest path
        # from a to b, taking into account the movement of asteroids, so that we would
        # only have to compute the path once
        if len(self.units) and self.obs.undiscovered_count() == 0:
            raise Exception(
                "Assigned scout role when all squares were explored",
                self.obs.undiscovered_count(),
            )
        for u_id in self.units:
            pos, energy = self.obs.units[u_id]
            dirs = direction(pos, self.closest_unexplored(pos))
            # with open("scout_actions.txt", "a") as file:
            #   file.write(str(dirs) + "\n")
            actions[u_id][0] = self.choose_dir(pos, dirs)

    def closest_unexplored(self, pos: tuple[int, int]) -> tuple[int, int]:
        m_dist = 0
        m_pos = None
        # This could maybe be done more efficiently, but you usually want to
        # avoid for loops in python
        for idx in np.dstack(np.where(self.obs.exploration == -1))[0]:
            x, y = int(idx[0]), int(idx[1])
            d = dist(pos, (x, y))
            # if unexplored unit is in sensor range, it is a nenula tile and should be ignored
            if m_pos is None or self.obs.sensor_range < d < m_dist:
                m_dist = d
                m_pos = (x, y)
        if m_pos is None:
            raise Exception(
                "Assigned scout role when all squares were explored. explore ratio (Should have been checked earlier)"
            )
        return m_pos
