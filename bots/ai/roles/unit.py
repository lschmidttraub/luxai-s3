from abc import abstractmethod
import numpy as np


class Unit:
    def __init__(self):
        self.units = []

    @abstractmethod
    def choose_action(self) -> dict[int, list[int]]:
        pass

    @abstractmethod
    def update_units(self, units: list[int]) -> None:
        self.units = units
