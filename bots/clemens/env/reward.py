import numpy as np
from typing import List, Dict, Tuple
from .base.constants import Global, SPACE_SIZE
from .base.enums import ActionType

class Reward: 
    def __init__(self, relic_configurations: List[Tuple[int, int, np.ndarray]]) -> None:
        self.relic_configurations = relic_configurations
        self.potential_visited: np.ndarray = None
        self.team_points_space: np.ndarray = None
        
    def init_reward_maps(self) -> None:
        self.potential_visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        self.team_points_space = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        
    def calculate_reward_for_unit(self, unit: Dict[str, int], action: ActionType, obs: Dict[str, any], tile_map: np.ndarray) -> float:
        """
        Calculate reward for a unit based on its action and observation.
        TODO: improve this
        """
        unit_reward: float = 0.0
        if action == ActionType.sap:
            if np.any(obs["obs"]["relic_nodes_mask"] == 1):
                enemy_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = unit["x"] + dx
                        ny = unit["y"] + dy
                        if not (0 <= nx < SPACE_SIZE and 0 <= ny < SPACE_SIZE):
                            continue
                        # Placeholder: count enemy units (to be implemented properly)
                        enemy_count += 0  
                if enemy_count >= 2:
                    unit_reward += 1.0 * enemy_count
                else:
                    unit_reward -= 4.0
            else:
                unit_reward -= 4.0
        else:
            # Additional reward computations for movement, energy nodes, etc.
            pass
        return unit_reward
    