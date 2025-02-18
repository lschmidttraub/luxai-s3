import numpy as np
from typing import Dict, Any, List, Tuple
from .base.constants import Global, SPACE_SIZE, MAX_RELIC_NODES

class Obs:
    def __init__(self, tile_map: np.ndarray, energy_map: np.ndarray, relic_map: np.ndarray) -> None:
        self.tile_map = tile_map
        self.energy_map = energy_map
        self.relic_map = relic_map
    
    def compute_unit_vision(self, unit: Dict[str, int]) -> np.ndarray:
        """
        Compute the sensor mask for a unit.
        Returns a boolean array of shape (SPACE_SIZE, SPACE_SIZE).
        """
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2
        vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        x, y = unit["x"], unit["y"]
        for dy in range(-sensor_range, sensor_range + 1):
            for dx in range(-sensor_range, sensor_range + 1):
                new_x = x + dx
                new_y = y + dy
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    continue
                contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                if self.tile_map[new_y, new_x] == 1:
                    contrib -= nebula_reduction
                vision[new_y, new_x] += contrib
        return vision > 0
    
    def get_global_sensor_mask(self, team_units: List[Dict[str, int]]) -> np.ndarray:
        """
        Return the combined sensor mask from all friendly units.
        """
        mask = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        for unit in team_units:
            mask |= self.compute_unit_vision(unit)
        return mask
    
    def get_unit_obs(self, unit: Dict[str, int], team_units: List[Dict[str, int]], enemy_units: List[Dict[str, int]]) -> Dict[str, Any]:
        """
        Create a local observation for a given unit.
        Only tiles in the unit's vision are visible.
        """
        sensor_mask = self.compute_unit_vision(unit)
        map_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_energy = np.where(sensor_mask, self.energy_map, -1)
        map_features = {"tile_type": map_tile_type, "energy": map_energy}
        sensor_mask_int = sensor_mask.astype(np.int8)
        
        # Process units info.
        NUM_TEAMS = 2
        MAX_UNITS = Global.MAX_UNITS
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)
        for i, u in enumerate(team_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = [ux, uy]
                units_energy[0, i] = u["energy"]
                units_mask[0, i] = 1
        for i, u in enumerate(enemy_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = [ux, uy]
                units_energy[1, i] = u["energy"]
                units_mask[1, i] = 1
                
        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros(MAX_RELIC_NODES, dtype=np.int8)
        idx = 0
        for (ry, rx) in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = [rx, ry]
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = [-1, -1]
                relic_nodes_mask[idx] = 0
            idx += 1

        team_points = np.array([0, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = 0
        match_steps = 0

        obs = {
            "units": {"position": units_position, "energy": units_energy},
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features": map_features,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps
        }
        return {"obs": obs, "remainingOverageTime": 60, "player": "player_0", "info": {}}