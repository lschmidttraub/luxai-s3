import numpy as np
from typing import Tuple
from .base.constants import Global, SPACE_SIZE

class Maps:
    def __init__(self) -> None:
        self.tile_map: np.ndarray = None  # Stores tile types.
        self.relic_map: np.ndarray = None  # Stores relic node existence.
        self.energy_map: np.ndarray = None  # Stores energy values.

    def init_maps(self) -> None:
        """Initialize the tile, relic, and energy maps."""
        num_tiles = SPACE_SIZE * SPACE_SIZE

        # Initialize tile_map with random 10% nebula and 10% asteroid tiles.
        self.tile_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_nebula = int(num_tiles * 0.1) 
        num_asteroid = int(num_tiles * 0.1)
        indices = np.random.choice(num_tiles, num_nebula + num_asteroid, replace=False)
        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nebula]] = 1
        flat_tiles[indices[num_nebula:]] = 2
        self.tile_map = flat_tiles.reshape((SPACE_SIZE, SPACE_SIZE))

        # Initialize relic_map with 3 random relic nodes.
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        relic_indices = np.random.choice(num_tiles, 3, replace=False)
        flat_relic = self.relic_map.flatten()
        flat_relic[relic_indices] = 1
        self.relic_map = flat_relic.reshape((SPACE_SIZE, SPACE_SIZE))

        # Initialize energy_map with 2 random energy nodes.
        self.energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_energy_nodes = 2
        indices_energy = np.random.choice(num_tiles, num_energy_nodes, replace=False)
        flat_energy = self.energy_map.flatten()
        flat_energy[indices_energy] = Global.MAX_ENERGY_PER_TILE
        self.energy_map = flat_energy.reshape((SPACE_SIZE, SPACE_SIZE))

    def roll_maps(self) -> None:
        """Roll (shift) the maps to simulate movement."""
        self.tile_map = np.roll(self.tile_map, shift=1, axis=1)
        self.relic_map = np.roll(self.relic_map, shift=1, axis=1)
        self.energy_map = np.roll(self.energy_map, shift=1, axis=1)
