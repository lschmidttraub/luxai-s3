from abc import ABC, abstractmethod
import numpy as np
from observation import Observation
from utils import *


class Unit(ABC):
    def __init__(self, obs: Observation):
        self.units: list[int] = []
        self.obs = obs

    @abstractmethod
    def choose_action(self, actions: np.ndarray) -> None:
        pass

    def update_units(self, units: list[int]) -> None:
        self.units = units

    def choose_dir(self, pos: tuple[int, int], d: tuple[int, int]) -> int:
        vision = self.obs.vision
        if vision.shape != (24, 24):
            raise Exception("graalhhh")
        sq1 = move(pos, d[0])
        sq2 = move(pos, d[1])
        if in_bounds(sq1) and vision[sq1] != ASTEROID_TILE:
            return d[0]
        elif in_bounds(sq2) and vision[sq2] != ASTEROID_TILE:
            return d[1]
        return np.random.randint(0, 5)
