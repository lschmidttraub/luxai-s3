import numpy as np
import numpy.ma as ma
import json
from utils import *


class Observation:
    def __init__(self, player: int, env_config: dict):
        self.player = player
        self.step: int
        self.pts: tuple[int, int] = (0, 0)
        self.pt_diff: tuple[int, int]
        self.player: int
        self.units: dict[int, tuple[tuple[int, int], int]] = {}
        self.enemy_units: dict[int, tuple[tuple[int, int], int]] = {}
        self.energy: ma.MaskedArray = ma.array(
            np.zeros((24, 24)), mask=np.ones((24, 24)).astype(bool)
        )
        self.vision: np.ndarray = np.zeros((24, 24))
        self.exploration = np.full((24, 24), -1)
        # Relic nodes are the tiles on the map, whilst relic tiles are the map tiles that give points
        self.relic_tiles: list[tuple[int, int]] = []
        self.relic_nodes: list[tuple[int, int]] = []
        self.relic_tile_mask: np.ndarray = np.zeros((24, 24)).astype(bool)
        self.params: dict[str, int] = env_config

    def update_observation(self, step: int, obs: dict) -> None:
        self.step = step
        units = obs["units"]
        unit_mask = obs["units_mask"]
        self.units = self.calc_units(
            units["position"][self.player],
            units["energy"][self.player],
            unit_mask[self.player],
        )
        opp = ~self.player
        self.enemy_units = self.calc_units(
            units["position"][opp], units["energy"][opp], unit_mask[opp]
        )
        vision_mask = obs["sensor_mask"]
        map_features = obs["map_features"]
        new_energy = ma.masked_array(
            map_features["energy"], mask=np.invert(vision_mask)
        )
        new_tiles = ma.masked_array(
            map_features["tile_type"], mask=np.invert(vision_mask)
        )
        self.update_vis(new_tiles)
        self.update_exploration(vision_mask)
        self.relic_nodes = [
            (n[0], n[1])
            for n, m in zip(obs["relic_nodes"], obs["relic_nodes_mask"])
            if m
        ]
        new_point = obs["team_points"]

        pts = obs["team_points"]
        self.pt_diff = pts - self.pts
        self.pts = pts

    def update_vis(self, new_tiles) -> None:
        i = match(self.vision, self.exploration != 0, new_tiles)
        self.vision = np.roll(self.vision.data, (i, -i), axis=(1, 0))
        if i:
            # if squares shifted, change new row and column to unexplored
            self.exploration[-(i == 1), :] = -1
            self.exploration[:, -(i == -1)] = -1
        mask = new_tiles.mask
        self.vision = np.where(mask, self.vision, new_tiles.data)

    def calc_units(self, pos, energy, mask) -> dict[int, tuple[tuple[int, int], int]]:
        return {i: (p, e) for i, (m, p, e) in enumerate(zip(mask, pos, energy)) if m}

    def update_exploration(self, vision_mask) -> None:
        self.exploration = self.exploration + (self.exploration != -1)
        self.exploration[vision_mask] = 0
