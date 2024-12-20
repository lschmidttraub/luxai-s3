import numpy as np
import numpy.ma as ma
import json
from utils import *


class Observation:
    def __init__(self, player: int, env_config: dict):
        self.player = player
        params: dict[str, int] = env_config
        self.max_units = params["max_units"]
        self.match_count = params["match_count_per_episode"]
        self.max_steps = params["max_steps_in_match"]
        self.H = params["map_height"]
        self.W = params["map_width"]
        self.move_cost = params["unit_move_cost"]
        self.sap_cost = params["unit_sap_cost"]
        self.sap_range = params["unit_sap_range"]
        self.sensor_range = params["unit_sensor_range"]
        self.step: int
        self.pts: tuple[int, int] = (0, 0)
        self.pt_diff: tuple[int, int]
        self.player: int
        self.units: dict[int, tuple[tuple[int, int], int]] = {}
        self.enemy_units: dict[int, tuple[tuple[int, int], int]] = {}
        self.energy: ma.MaskedArray = ma.array(
            np.zeros((self.W, self.H)), mask=np.ones((self.W, self.H)).astype(bool)
        )
        self.vision: np.ndarray = np.zeros((self.W, self.H))
        self.exploration = np.full((self.W, self.H), -1)
        # Relic nodes are the tiles on the map, whilst relic tiles are the map tiles that give points
        self.relic_tiles: set[tuple[int, int]]
        self.relic_nodes: set[tuple[int, int]]
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
        self.relic_nodes = {
            pair
            for n, m in zip(obs["relic_nodes"], obs["relic_nodes_mask"])
            if m
            for pair in [(n[0], n[1]), (n[1], n[0])]
        }

        pts = obs["team_points"]
        self.pt_diff = pts - self.pts
        self.pts = pts

    def update_vis(self, new_tiles) -> None:
        if ~self.drift_steps and (self.step == 21 or self.step == 41):
            i = match(self.vision, self.exploration != 0, new_tiles)
            if i:
                self.drift_steps = self.step - 1
                self.drift_dir = i
        elif self.drift_steps:
            if self.step % self.drift_steps == 1:
                self.vision = np.roll(
                    self.vision.data, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                # if squares shifted, change new row and column to unexplored
                if self.drift_dir == -1:
                    self.exploration[-1, :] = -1
                    self.exploration[:, 0] = -1
                elif self.drift_dir == 1:
                    self.exploration[0, :] = -1
                    self.exploration[:, -1] = -1
        mask = new_tiles.mask
        self.vision = np.where(mask, self.vision, new_tiles.data)

    def calc_units(self, pos, energy, mask) -> dict[int, tuple[tuple[int, int], int]]:
        return {i: (p, e) for i, (m, p, e) in enumerate(zip(mask, pos, energy)) if m}

    def update_exploration(self, vision_mask) -> None:
        self.exploration = self.exploration + (self.exploration != -1)
        self.exploration[vision_mask] = 0

    def found_all_relics(self) -> bool:
        if len(self.units)==self.max_units:
            return True
        for i in range(self.W):
            for j in range(i,self.H):
                if not(self.exploration[i][j] or self.exploration[j][i]):
                    return False
        return True


