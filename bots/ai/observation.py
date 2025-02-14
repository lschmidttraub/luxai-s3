"""
The Observation class is made to create a coherent inner representation of what the agent knows of its environment
"""

import numpy as np
from utils import *


class Observation:
    def __init__(self, player: int, env_config: dict):
        self.player = player
        # Save the parameters of the entire map just in case, might delete later
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
        # Difference in points between successive turns
        # shows many relic tiles were visited during this turn
        self.pt_diff: tuple[int, int]
        self.player: int
        self.units: dict[int, tuple[tuple[int, int], int]] = {}
        self.enemy_units: dict[int, tuple[tuple[int, int], int]] = {}
        self.energy: np.ndarray = np.zeros((self.W, self.H), dtype=int)
        # This is the vision of the playing field(empty tiles, asteroids, nebulae etc.)
        self.vision: np.ndarray = np.full((self.W, self.H), UNKNOWN, dtype=int)
        # This shows how many turns it has been since the tiles have been seen (the idea is that tiles
        # that aren't currently being seen but have been visited at previous turns still serve a purpose)
        self.exploration: np.ndarray = np.full((self.W, self.H), UNKNOWN, dtype=int)
        # Relic tiles are the tiles on the map, whilst relic nodes are the map tiles that give points
        self.relic_tiles: set[tuple[int, int]] = set({})
        self.relic_nodes: set[tuple[int, int]] = set({})
        # Relic tiles aren't visible, so we play an elimination game with all tiles in the map
        self.relic_tile_mask: np.ndarray = np.zeros((self.W, self.H)).astype(bool)

        # Mobile tiles move at regular intervals, drift_steps = number of steps between motion
        self.drift_steps = 0
        # Either 0(no motion), 1(towards top-right), -1(towards bottom-left)
        self.drift_dir = 0

    def update_observation(self, step: int, obs: dict) -> None:
        """
        Updates the attributes of the observation class
        """
        self.step = step
        units = obs["units"]
        unit_mask = obs["units_mask"]
        self.units = self.calc_units(
            units["position"][self.player],
            units["energy"][self.player],
            unit_mask[self.player],
        )
        opp = ~self.player
        # The idea is to apply the same policy used on friendly units to the
        # opponent's units
        self.enemy_units = self.calc_units(
            units["position"][opp], units["energy"][opp], unit_mask[opp]
        )
        # boolean mask of tiles that are currently visible
        vision_mask = np.array(obs["sensor_mask"], dtype=bool)
        # dictionary with energy and tile types
        map_features = obs["map_features"]
        self.energy = map_features["energy"]
        new_tiles = np.array(map_features["tile_type"], dtype=int)
        self.update_vis(new_tiles, vision_mask)
        self.update_exploration(vision_mask, new_tiles)
        # only the relic nodes that are unmasked are visible, the other ones should be ignored
        self.relic_nodes = {
            pair
            for n, m in zip(obs["relic_nodes"], obs["relic_nodes_mask"])
            if m
            for pair in [(n[0], n[1]), (n[1], n[0])]
        }
        # update points are point difference
        pts = obs["team_points"]
        self.pt_diff = pts - self.pts
        self.pts = pts

    def update_vis(self, new_tiles, new_mask) -> None:
        """
        Updates the vision parameter (representation of map)
        """
        # These are the two possible non-zero times at which the tiles first move
        # speed: -0.05, -0.025, 0, 0.025, 0.05
        if ~self.drift_steps and (self.step == 21 or self.step == 41):
            # match the new tiles to a tiles motion of the previous board: -1, 0 or 1
            i = Utils.match(self.vision, self.exploration == 0, new_tiles, new_mask)
            if i:
                # if i is non-zero, then we know dift_steps, as this is the first time the board moves
                # if i is always 0, then we know the board doesn't move
                self.drift_steps = self.step - 1
                self.drift_dir = i
        if self.drift_steps:
            # There is an annoying off-by-one error for the motion of the board, as turns start at 1, not 0
            # since modulo by 0 leads to an error we need to check that the drift_steps are not 0
            if self.step % self.drift_steps == 1:
                # Rotate vision and exploration mask in appropriate direction every k=drift_steps steps
                self.vision = np.roll(
                    self.vision, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                self.exploration = np.roll(
                    self.exploration, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                # if squares shifted, change new row and column to unexplored
                # We use the UNKNOWN constant in two contexts: for exploration and for vision
                # In both cases, the idea is the same (-1 for tiles we haven't seen)
                if self.drift_dir == 1:
                    self.exploration[-1, :] = UNKNOWN
                    self.exploration[:, 0] = UNKNOWN
                elif self.drift_dir == -1:
                    self.exploration[0, :] = UNKNOWN
                    self.exploration[:, -1] = UNKNOWN
                # This exception still arises every now and then, I don't know why
                # NEED TO FIX
                joined_mask = np.logical_and(self.exploration == 0, new_mask)
                if not np.all(self.vision[joined_mask] == new_tiles[joined_mask]):
                    if DEBUG:
                        Utils.tofile("debug/vision.txt", self.vision)
                        Utils.tofile("debug/new.txt", new_tiles)
                        Utils.tofile("debug/mask.txt", joined_mask)
                        Utils.tofile("debug/sensor.txt", new_mask)
                        Utils.tofile("debug/exploration.txt", self.exploration)

                    raise Exception(
                        "Discrepancy between predicted and observed shift",
                        self.drift_steps,
                        self.drift_dir,
                        self.step,
                    )
        # replace all tiles in vision that are currently seen with what is currently seen
        self.vision = np.where(new_mask, new_tiles, self.vision)

    def calc_units(self, pos, energy, mask) -> dict[int, tuple[tuple[int, int], int]]:
        """
        Calculates the units of a given team
        """
        # Units have an id (position in the action array), a position and an energy level
        # The conversion of p from np.ndarray to tuple is important
        return {
            i: (tuple(p), e) for i, (m, p, e) in enumerate(zip(mask, pos, energy)) if m
        }

    def update_exploration(self, vision_mask, vision) -> None:
        """
        Updates the exploration attribute of the Observation class
        Has to be called after the vision is updated
        """
        # Increment the time since exploration all tiles that have already been explored by one
        self.exploration = self.exploration + (self.exploration != -1)
        # Set the time since exploration of all tiles currently seen to 0
        # It is important to remember that nebula tiles aren't counted as explored:
        # we don't know what's underneath them
        self.exploration[np.logical_and(vision_mask, vision != NEBULA_TILE)] = 0

    def undiscovered_count(self) -> int:
        """
        Returns the count of unexplored tiles
        """
        return np.sum(self.exploration == UNKNOWN)

    def found_all_relics(self):
        """
        INCOMPLETE
        This function is supposed to indicate whether we have discovered all relic tiles.
        For now, we just return false
        """
        return False
