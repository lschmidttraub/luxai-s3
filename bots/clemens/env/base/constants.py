class Global:
    # Constants
    SPACE_SIZE: int = 24
    MAX_UNITS: int = 16
    RELIC_REWARD_RANGE: int = 2
    MAX_STEPS_IN_MATCH: int = 100
    MAX_ENERGY_PER_TILE: int = 20
    MAX_RELIC_NODES: int = 6
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR: int = 50
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR: int = 2

    # Dynamic constants
    UNIT_MOVE_COST: int = 1  # OPTIONS: list(range(1, 6))
    UNIT_SAP_COST: int = 30  # OPTIONS: list(range(30, 51))
    UNIT_SAP_RANGE: int = 3  # OPTIONS: list(range(3, 8))
    UNIT_SENSOR_RANGE: int = 2  # OPTIONS: [1, 2, 3, 4]
    OBSTACLE_MOVEMENT_PERIOD: int = 20  # OPTIONS: 6.67, 10, 20, 40
    OBSTACLE_MOVEMENT_DIRECTION: tuple[int, int] = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]

    # Static constants
    NEBULA_ENERGY_REDUCTION: int = 5  # OPTIONS: [0, 1, 2, 3, 5, 25]

SPACE_SIZE: int = Global.SPACE_SIZE
MAX_RELIC_NODES: int = Global.MAX_RELIC_NODES