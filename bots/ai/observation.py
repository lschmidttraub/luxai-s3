import numpy as np
import numpy.ma as ma
import json
from utils import *


class Observation:
    def __init__(self, player: int, env_config: dict):
        self.player = player
        params: dict[str, int] = env_config
        self.max_units: int = params["max_units"]
        self.match_count: int = params["match_count_per_episode"]
        self.max_steps: int = params["max_steps_in_match"]
        self.H: int = params["map_height"]
        self.W: int = params["map_width"]
        self.move_cost: int = params["unit_move_cost"]
        self.sap_cost: int = params["unit_sap_cost"]
        self.sap_range: int = params["unit_sap_range"]
        self.sensor_range: int = params["unit_sensor_range"]

        self.step: int
        self.pts: tuple[int, int] = (0, 0)
        self.pt_diff: tuple[int, int]
        self.player: int
        self.units: dict[int, tuple[tuple[int, int], int]] = {}
        self.enemy_units: dict[int, tuple[tuple[int, int], int]] = {}
        self.energy: np.ndarray = np.zeros((self.W, self.H), dtype=int)
        self.vision: np.ndarray = np.full((self.W, self.H), -1, dtype=int)
        self.exploration: np.ndarray = np.full((self.W, self.H), -1)
        # Relic nodes are the tiles on the map, whilst relic tiles are the map tiles that give points
        self.relic_tiles: set[tuple[int, int]] = set({})
        self.relic_nodes: set[tuple[int, int]] = set({})
        self.relic_tile_mask: np.ndarray = np.zeros((self.W, self.H)).astype(bool)

        self.drift_steps = 0
        self.drift_dir = 0

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
        vision_mask = np.array(obs["sensor_mask"], dtype=bool)
        map_features = obs["map_features"]
        self.energy = map_features["energy"]
        new_tiles = np.array(map_features["tile_type"], dtype=int)
        self.update_vis(new_tiles, vision_mask)
        self.update_exploration(vision_mask)
        self.relic_nodes = {
            pair
            for n, m in zip(obs["relic_nodes"], obs["relic_nodes_mask"])
            if m
            for pair in [(n[0], n[1]), (n[1], n[0])]
        }

        pts = obs["team_points"]
        self.pt_diff = pts - self.pts
        self.pts = pts

    def update_vis(self, new_tiles, new_mask) -> None:
        if ~self.drift_steps and (self.step == 21 or self.step == 41):
            i = match(self.vision, self.exploration == 0, new_tiles, new_mask)
            if i:
                self.drift_steps = self.step - 1
                self.drift_dir = i
        if self.drift_steps:
            if self.step % self.drift_steps == 1:
                self.vision = np.roll(
                    self.vision, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                self.exploration = np.roll(
                    self.exploration, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                # if squares shifted, change new row and column to unexplored
                if self.drift_dir == 1:
                    self.exploration[-1, :] = -1
                    self.exploration[:, 0] = -1
                elif self.drift_dir == -1:
                    self.exploration[0, :] = -1
                    self.exploration[:, -1] = -1
                joined_mask = np.logical_and(self.exploration == 0, new_mask)
                if not np.all(self.vision[joined_mask] == new_tiles[joined_mask]):
                    tofile("vision.txt", self.vision)
                    tofile("new.txt", new_tiles)
                    tofile("mask.txt", joined_mask)
                    tofile("sensor.txt", new_mask)
                    tofile("exploration.txt", self.exploration)

                    raise Exception("Discrepancy between predicted and observed shift")
        self.vision = np.where(new_mask, new_tiles, self.vision)

    def calc_units(self, pos, energy, mask) -> dict[int, tuple[tuple[int, int], int]]:
        return {i: (p, e) for i, (m, p, e) in enumerate(zip(mask, pos, energy)) if m}

    def update_exploration(self, vision_mask) -> None:
        self.exploration = self.exploration + (self.exploration != -1)
        self.exploration[vision_mask] = 0

    def found_all_relics(self) -> bool:
        if len(self.units) == self.max_units:
            return True
        return self.undiscovered_count == 0

    def undiscovered_count(self) -> int:
        # higher ratio means more of the map is undiscovered
        return np.sum(self.exploration == -1)
