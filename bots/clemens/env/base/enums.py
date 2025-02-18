from enum import IntEnum

class NodeType(IntEnum):
    UNKNOWN = -1
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.name.lower()


class ActionType(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_coordinates(cls, current_position: tuple[int, int], next_position: tuple[int, int]) -> "ActionType":
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]

        if dx < 0:
            return cls.LEFT
        elif dx > 0:
            return cls.RIGHT
        elif dy < 0:
            return cls.UP
        elif dy > 0:
            return cls.DOWN
        else:
            return cls.CENTER

    def to_direction(self) -> tuple[int, int]:
        return _DIRECTIONS[self]


_DIRECTIONS: list[tuple[int, int]] = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  # down
    (-1, 0),  # left
    (0, 0),  # sap
]
