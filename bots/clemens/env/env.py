import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Tuple
from .base.constants import Global, SPACE_SIZE, MAX_RELIC_NODES
from .base.enums import ActionType
from .maps import Maps
from .units import Units
from .obs import Obs
from .reward import Reward

NUM_TEAMS: int = 2
MAX_UNITS: int = Global.MAX_UNITS

class Env(gym.Env):
    """
    A gym environment for PPO.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self) -> None:
        super(Env, self).__init__()
        self.action_space = spaces.MultiDiscrete([len(ActionType)] * MAX_UNITS) # One action per unit
        self.observation_space = spaces.Dict({
            "units_position": spaces.Box(low=0, high=SPACE_SIZE - 1, shape=(NUM_TEAMS, MAX_UNITS, 2), dtype=np.int32),
            "units_energy": spaces.Box(low=0, high=400, shape=(NUM_TEAMS, MAX_UNITS, 1), dtype=np.int32),
            "units_mask": spaces.Box(low=0, high=1, shape=(NUM_TEAMS, MAX_UNITS), dtype=np.int8),
            "sensor_mask": spaces.Box(low=0, high=1, shape=(SPACE_SIZE, SPACE_SIZE), dtype=np.int8),
            "map_features_tile_type": spaces.Box(low=-1, high=2, shape=(SPACE_SIZE, SPACE_SIZE), dtype=np.int8),
            "map_features_energy": spaces.Box(low=-1, high=Global.MAX_ENERGY_PER_TILE, shape=(SPACE_SIZE, SPACE_SIZE), dtype=np.int8),
            "relic_nodes_mask": spaces.Box(low=0, high=1, shape=(MAX_RELIC_NODES,), dtype=np.int8),
            "relic_nodes": spaces.Box(low=-1, high=SPACE_SIZE - 1, shape=(MAX_RELIC_NODES, 2), dtype=np.int32),
            "team_points": spaces.Box(low=0, high=1000, shape=(NUM_TEAMS,), dtype=np.int32),
            "team_wins": spaces.Box(low=0, high=1000, shape=(NUM_TEAMS,), dtype=np.int32),
            "steps": spaces.Box(low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32),
            "match_steps": spaces.Box(low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32),
            "remainingOverageTime": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            "env_cfg_map_width": spaces.Box(low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32),
            "env_cfg_map_height": spaces.Box(low=0, high=SPACE_SIZE, shape=(1,), dtype=np.int32),
            "env_cfg_max_steps_in_match": spaces.Box(low=0, high=Global.MAX_STEPS_IN_MATCH, shape=(1,), dtype=np.int32),
            "env_cfg_unit_move_cost": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "env_cfg_unit_sap_cost": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "env_cfg_unit_sap_range": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
        })
        
        self.max_steps: int = Global.MAX_STEPS_IN_MATCH
        self.current_step: int = 0
        self.score: int = 0
        
        self.maps = Maps()
        self.maps.init_maps()
        self.units = Units(team_spawn=(0,0), enemy_spawn=(SPACE_SIZE-1,SPACE_SIZE-1))
        self.units.init_units()
        self.obs = Obs(self.maps.tile_map, self.maps.energy_map, self.maps.relic_map)
        
        self.relic_configurations = []
        relic_coords = np.argwhere(self.maps.relic_map == 1)
        for (y, x) in relic_coords:
            mask = np.zeros((5, 5), dtype=bool)
            indices_mask = np.random.choice(25, 5, replace=False)
            mask_flat = mask.flatten()
            mask_flat[indices_mask] = True
            mask = mask_flat.reshape((5, 5))
            self.relic_configurations.append((x, y, mask))
            
        self.reward = Reward(self.relic_configurations)
        self.reward.init_reward_maps()
        
        self.visited = self.obs.get_global_sensor_mask(self.units.team_units)
        
        self.env_cfg: Dict[str, Any] = {
            "map_width": SPACE_SIZE,
            "map_height": SPACE_SIZE,
            "max_steps_in_match": Global.MAX_STEPS_IN_MATCH,
            "unit_move_cost": Global.UNIT_MOVE_COST,
            "unit_sap_cost": Global.UNIT_SAP_COST,
            "unit_sap_range": Global.UNIT_SAP_RANGE,
            "max_units": MAX_UNITS
        }
        
    def get_obs(self) -> Dict[str, Any]:
        """
        Return a flattened global observation.
        """
        sensor_mask = self.obs.get_global_sensor_mask(self.units.team_units)
        sensor_mask_int = sensor_mask.astype(np.int8)
        map_features_tile_type = np.where(sensor_mask, self.maps.tile_map, -1)
        map_features_energy = np.where(sensor_mask, self.maps.energy_map, -1)
        
        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)
        for i, unit in enumerate(self.units.team_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = [ux, uy]
                units_energy[0, i] = unit["energy"]
                units_mask[0, i] = 1
        for i, unit in enumerate(self.units.enemy_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = [ux, uy]
                units_energy[1, i] = unit["energy"]
                units_mask[1, i] = 1
                
        relic_coords = np.argwhere(self.maps.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros((MAX_RELIC_NODES,), dtype=np.int8)
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

        team_points = np.array([self.score, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = np.array([self.current_step], dtype=np.int32)
        match_steps = np.array([self.current_step], dtype=np.int32)
        remainingOverageTime = np.array([60], dtype=np.int32)

        flat_obs = {
            "units_position": units_position,
            "units_energy": units_energy,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features_tile_type": map_features_tile_type,
            "map_features_energy": map_features_energy,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps,            
            "remainingOverageTime": remainingOverageTime,
            "env_cfg_map_width": np.array([self.env_cfg["map_width"]], dtype=np.int32),
            "env_cfg_map_height": np.array([self.env_cfg["map_height"]], dtype=np.int32),
            "env_cfg_max_steps_in_match": np.array([self.env_cfg["max_steps_in_match"]], dtype=np.int32),
            "env_cfg_unit_move_cost": np.array([self.env_cfg["unit_move_cost"]], dtype=np.int32),
            "env_cfg_unit_sap_cost": np.array([self.env_cfg["unit_sap_cost"]], dtype=np.int32),
            "env_cfg_unit_sap_range": np.array([self.env_cfg["unit_sap_range"]], dtype=np.int32)
        }
        return flat_obs
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.
        """
        self.current_step = 0
        self.score = 0
        self.maps.init_maps()
        self.units.init_units()
        self.obs = Obs(self.maps.tile_map, self.maps.energy_map, self.maps.relic_map)
        self.visited = self.obs.get_global_sensor_mask(self.units.team_units)
        return self.get_obs()
    
    def step(self, actions: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply the actions
        Update the environment
        Return (observation, reward, done, info)
        """
        self.current_step += 1
        total_reward: float = 0.0
        
        # Friendly units
        for idx, unit in enumerate(self.units.team_units):
            unit_reward: float = 0.0
            act = actions[idx]
            action_enum: ActionType = ActionType(act)
            unit_obs = self.obs.get_unit_obs(unit, self.units.team_units, self.units.enemy_units)
            
            if action_enum == ActionType.SAP:
                # Sap
                if np.any(unit_obs["obs"]["relic_nodes_mask"] == 1):
                    enemy_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx = unit["x"] + dx
                            ny = unit["y"] + dy
                            if not (0 <= nx < SPACE_SIZE and 0 <= ny < SPACE_SIZE):
                                continue
                            enemy_count += 0  # TODO: Placeholder for enemy count.
                    if enemy_count >= 2:
                        unit_reward += 1.0 * enemy_count # Reward correct sap
                    else:
                        unit_reward -= 4.0 # Penalize sap into crowd of less than 2 enemies
                else:
                    unit_reward -= 4.0 # Penalize random sap for now TODO
            else:
                # Movement
                if action_enum in [ActionType.UP, ActionType.RIGHT, ActionType.DOWN, ActionType.LEFT]:
                    dx, dy = action_enum.to_direction()
                else:
                    dx, dy = (0, 0)
                new_x = unit["x"] + dx
                new_y = unit["y"] + dy
                if not (0 <= new_x < SPACE_SIZE and 0 <= new_y < SPACE_SIZE):
                    unit_reward -= 2.0 # Penalize move outside
                    new_x, new_y = unit["x"], unit["y"]
                elif self.maps.tile_map[new_y, new_x] == 2:
                    unit_reward -= 2.0 # Penalize move into obstacle
                    new_x, new_y = unit["x"], unit["y"]
                else:
                    unit["x"], unit["y"] = new_x, new_y
                
                # Additional relic collection
                unit_obs = self.obs.get_unit_obs(unit, self.units.team_units, self.units.enemy_units)
                for (rx, ry, mask) in self.relic_configurations:
                    if rx - 2 <= unit["x"] <= rx + 2 and ry - 2 <= unit["y"] <= ry + 2:
                        ix = unit["x"] - rx + 2
                        iy = unit["y"] - ry + 2
                        if 0 <= ix < 5 and 0 <= iy < 5 and mask[iy, ix]:
                            if not self.reward.potential_visited[unit["y"], unit["x"]]:
                                unit_reward += 2.0 # New tile near relic
                                self.reward.potential_visited[unit["y"], unit["x"]] = True
                            if not self.reward.team_points_space[unit["y"], unit["x"]]:
                                self.score += 1
                                unit_reward += 5.0 # Capture relic first time
                                self.reward.team_points_space[unit["y"], unit["x"]] = True
                            else:
                                unit_reward += 5.0 # Capture relic again
                if unit_obs["obs"]["map_features"]["energy"][unit["y"], unit["x"]] == Global.MAX_ENERGY_PER_TILE:
                    unit_reward += 0.2 # Reward max energy
                if unit_obs["obs"]["map_features"]["tile_type"][unit["y"], unit["x"]] == 1:
                    unit_reward -= 1.0 # Penalize nebula
                for enemy in self.units.enemy_units:
                    if enemy["x"] == unit["x"] and enemy["y"] == unit["y"]:
                        if enemy["energy"] < unit["energy"]:
                            unit_reward += 1.0 # Shares tile with enemy & more energy
            total_reward += unit_reward
            
        union_mask = self.obs.get_global_sensor_mask(self.units.team_units)
        new_tiles = union_mask & (~self.visited)
        num_new = np.sum(new_tiles)
        if num_new > 0:
            total_reward += 0.05 * num_new # Reward new tiles
        self.visited[new_tiles] = True

        if self.current_step % 3 == 0:
            if len(self.units.team_units) < MAX_UNITS:
                self.units.spawn_unit(team=0)
            if len(self.units.enemy_units) < MAX_UNITS:
                self.units.spawn_unit(team=1)

        if self.current_step % 20 == 0:
            self.maps.roll_maps()
            self.units.update_enemy_units()

        done = self.current_step >= 1000
        info = {"score": self.score, "step": self.current_step}
        return self.get_obs(), total_reward, done, info
    
    def render(self, mode: str = "human") -> None:
        display = self.maps.tile_map.astype(str).copy()
        for unit in self.units.team_units:
            display[unit["y"], unit["x"]] = "A"
        print("Step: ", self.current_step)
        print(display)