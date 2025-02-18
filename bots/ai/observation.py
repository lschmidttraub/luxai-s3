"""
The Observation class is made to create a coherent inner representation of what the agent knows of its environment
"""

import numpy as np
from utils import *
import seaborn as sns


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
        self.shape = (self.W, self.H)
        self.move_cost: int = params["unit_move_cost"]
        self.sap_cost: int = params["unit_sap_cost"]
        self.sap_range: int = params["unit_sap_range"]
        self.sensor_range: int = params["unit_sensor_range"]

        self.player: int = player

        self.step: int = 0
        self.pts: np.ndarray = np.zeros(2, dtype=int)
        # Difference in points between successive turns
        # shows many relic tiles were visited during this turn
        self.pt_diff: np.ndarray | None = None

        self.units: dict[int, tuple[tuple[int, int], int]] = {}
        self.enemy_units: dict[int, tuple[tuple[int, int], int]] = {}
        self.energy: np.ndarray = np.zeros(self.shape, dtype=int)
        # This is the vision of the playing field(empty tiles, asteroids, nebulae etc.)
        self.vision: np.ndarray = np.full(self.shape, UNKNOWN, dtype=int)
        # This shows how many turns it has been since the tiles have been seen (the idea is that tiles
        # that aren't currently being seen but have been visited at previous turns still serve a purpose)
        self.exploration: np.ndarray = np.full(self.shape, UNKNOWN, dtype=int)
        # mask of nebula tiles
        self.nebula_mask = np.zeros(self.shape, dtype=bool)

        # Relic tiles are the tiles on the map, whilst relic nodes are the map tiles that give points
        self.relic_nodes: set[tuple[int, int]] = set()

        # Mobile tiles move at regular intervals, drift_steps = number of steps between motion
        self.drift_steps = 0
        # Either 0(no motion), 1(towards top-right), -1(towards bottom-left)
        self.drift_dir = 0
        self.prev_mask: np.ndarray

    def update_observation(self, step: int, obs: dict) -> None:
        """
        Updates the attributes of the observation class
        """
        units = obs["units"]
        unit_mask = obs["units_mask"]
        self.units = self.calc_units(
            units["position"][self.player],
            units["energy"][self.player],
            unit_mask[self.player],
        )
        opp = 1 - self.player
        # The idea is to apply the same policy used on friendly units to the
        # opponent's units
        self.enemy_units = self.calc_units(
            units["position"][opp], units["energy"][opp], unit_mask[opp]
        )

        # boolean mask of tiles that are currently visible
        vision_mask = np.array(obs["sensor_mask"], dtype=bool)
        self.prev_mask = vision_mask

        # dictionary with energy and tile types
        map_features = obs["map_features"]
        # we don't currently use energy information but probably should
        new_energy = map_features["energy"]
        # new observed tiles
        new_tiles = np.array(map_features["tile_type"], dtype=int)

        self.shift(step, new_tiles, vision_mask)

        # Update exploration before vision
        self.update_exploration(new_tiles, vision_mask)

        self.update_vis(new_tiles, vision_mask)

        self.update_nebulae(new_tiles, vision_mask)

        self.update_energy(new_energy, vision_mask)

        # only the relic nodes that are unmasked are visible, the other ones should be ignored
        # we add all elements (witht the | operator) instead of replacing the set outright since the observed relic nodes reset after each match
        self.relic_nodes |= {
            pair
            for coords in map(tuple, obs["relic_nodes"][obs["relic_nodes_mask"]])
            # since the map is symmetric w.r.t. the diagonal we can exploit this
            for pair in [coords, Utils.symmetric(coords)]
        }

        self.update_pts(obs["team_points"])

    def update_pts(self, new_pts) -> None:
        """
        Updates pts and pt_diff attributes
        """
        pts = new_pts
        if self.pts is not None:
            self.pt_diff = pts - self.pts
        # check if not not None to avoid IDE error
        if self.pt_diff is not None and (self.pt_diff < 0).any():
            # if pt_diff is negative, a new match has started, so we set pt_diff to None
            self.pt_diff = None
        self.pts = pts

    def shift(self, step: int, new_tiles, vision_mask) -> None:
        """
        In the first part, this function determines the drift speed and direction of asteroids and nebula
        Once this has been determined, it shifts all relevant arrays in accordance with the drift speed and
        direction.
        """
        self.step = step

        # These are the two possible non-zero times at which the tiles first move
        # speed: -0.05, -0.025, 0, 0.025, 0.05
        if not self.drift_steps and (self.step == 21 or self.step == 41):
            # match the new tiles to a tiles motion of the previous board: -1, 0 or 1
            d = Utils.match_shift(
                self.vision, self.exploration == 0, new_tiles, vision_mask
            )
            if d:
                # if i is non-zero, then we know dift_steps, as this is the first time the board moves
                # if i is always 0, then we know the board doesn't move
                self.drift_steps = self.step - 1
                self.drift_dir = d

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
                self.nebula_mask = np.roll(
                    self.nebula_mask, (self.drift_dir, -self.drift_dir), axis=(1, 0)
                )
                # if squares shifted, change new row and column to unexplored
                # We use the UNKNOWN constant in two contexts: for exploration and for vision
                # In both cases, the idea is the same (-1 for tiles we haven't seen)
                if self.drift_dir == 1:
                    self.exploration[-1, :] = UNKNOWN
                    self.exploration[:, 0] = UNKNOWN
                    self.nebula_mask[-1, :] = False
                    self.nebula_mask[:, 0] = False

                elif self.drift_dir == -1:
                    self.exploration[0, :] = UNKNOWN
                    self.exploration[:, -1] = UNKNOWN
                    self.nebula_mask[0, :] = False
                    self.nebula_mask[:, -1] = False

                # This exception still arises every now and then, I don't know why
                # NEED TO FIX
                joined_mask = np.logical_and(self.exploration == 0, vision_mask)
                if not np.all(self.vision[joined_mask] == new_tiles[joined_mask]):
                    if DEBUG:
                        Utils.heatmap("debug/vision.jpg", self.vision)
                        Utils.heatmap("debug/new.jpg", new_tiles)
                        Utils.heatmap("debug/mask.jpg", joined_mask)
                        Utils.heatmap("debug/sensor.jpg", vision_mask)
                        Utils.heatmap("debug/exploration.jpg", self.exploration)

                    raise Exception(
                        "Discrepancy between predicted and observed shift",
                        self.drift_steps,
                        self.drift_dir,
                        self.step,
                    )

    def update_vis(self, new_tiles, vision_mask) -> None:
        """
        Updates the vision parameter (representation of map)
        """
        # replace all tiles in vision that are currently seen with what is currently seen
        # except for nebula tiles
        self.vision = np.where(
            np.logical_and(vision_mask, new_tiles != NEBULA_TILE),
            new_tiles,
            self.vision,
        )
        self.vision[self.exploration == UNKNOWN] = UNKNOWN

    def calc_units(self, pos, energy, mask) -> dict[int, tuple[tuple[int, int], int]]:
        """
        Calculates the units of a given team
        """
        # Units have an id (position in the action array), a position and an energy level
        # The conversion of p from np.ndarray to tuple is important
        return {
            i: (tuple(p), e) for i, (m, p, e) in enumerate(zip(mask, pos, energy)) if m
        }

    def update_exploration(self, new_tiles, vision_mask) -> None:
        """
        Updates the exploration attribute of the Observation class as well as the new_explored_tiles attribute
        """
        # Increment the time since exploration all tiles that have already been explored by one
        self.exploration = self.exploration + (self.exploration != UNKNOWN)
        # Set the time since exploration of all tiles currently seen to 0
        # It is important to remember that nebula tiles aren't counted as explored:
        # we don't know what's underneath them
        self.exploration[np.logical_and(vision_mask, new_tiles != NEBULA_TILE)] = 0
        """
        if self.step < 150 and not self.step % 10:
            Utils.heatmap(
                f"debug/plots/exploration{self.step}", self.exploration == UNKNOWN
            )
        """

    def update_nebulae(self, new_tiles, vision_mask) -> None:
        """
        adds newly observed nebula tiles to nebula array
        """
        self.nebula_mask[np.logical_and(new_tiles == NEBULA_TILE, vision_mask)] = True

    def update_energy(self, new_energy, vision_mask) -> None:
        self.energy = np.where(vision_mask, new_energy, 0)

    def undiscovered_count(self) -> int:
        """
        Returns the count of unexplored tiles
        """
        return np.sum(self.exploration == UNKNOWN)

    def discovered_all_tiles_except_nebula(self):
        """
        return True iff all tiles, except for nebula tiles, have been discovered
        """
        return np.logical_or(self.exploration != UNKNOWN, self.nebula_mask).all()
