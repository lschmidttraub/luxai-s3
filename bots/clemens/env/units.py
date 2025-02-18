from typing import List, Dict, Type, Tuple
from .base.constants import Global, SPACE_SIZE

class Units:
    def __init__(self, team_spawn: Tuple[int, int], enemy_spawn: Tuple[int, int]) -> None:
        self.team_spawn: Tuple[int, int] = team_spawn
        self.enemy_spawn: Tuple[int, int] = enemy_spawn
        self.team_units: List[Dict[str, int]] = [] # x, y, energy
        self.enemy_units: List[Dict[str, int]] = [] # x, y, energy 

    def init_units(self) -> None:
        """Initialize friendly and enemy units."""
        self.team_units = [{"x": self.team_spawn[0], "y": self.team_spawn[1], "energy": 100}]
        self.enemy_units = [{"x": self.enemy_spawn[0], "y": self.enemy_spawn[1], "energy": 100}]

    def spawn_unit(self, team: int) -> None:
        """Spawn a new unit for the given team."""
        if team == 0:
            self.team_units.append({"x": self.team_spawn[0], "y": self.team_spawn[1], "energy": 100})
        elif team == 1:
            self.enemy_units.append({"x": self.enemy_spawn[0], "y": self.enemy_spawn[1], "energy": 100})

    def update_enemy_units(self) -> None:
        """Update enemy units by moving each enemy unit to the right (with boundary check)."""
        for enemy in self.enemy_units:
            enemy["x"] = min(enemy["x"] + 1, SPACE_SIZE - 1)