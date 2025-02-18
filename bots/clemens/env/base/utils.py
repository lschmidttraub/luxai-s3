from constants import Global, SPACE_SIZE

def get_match_step(step: int) -> int:
    """Returns the match step (resets after MAX_STEPS_IN_MATCH)."""
    return step % (Global.MAX_STEPS_IN_MATCH + 1)


def get_match_number(step: int) -> int:
    """Returns the match number based on step count."""
    return step // (Global.MAX_STEPS_IN_MATCH + 1)


def get_opposite(x: int, y: int) -> tuple[int, int]:
    """Returns the mirrored point across the diagonal."""
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_upper_sector(x: int, y: int) -> bool:
    """Determines if a coordinate is in the upper sector."""
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x: int, y: int) -> bool:
    """Determines if a coordinate is in the lower sector."""
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id: int, x: int, y: int) -> bool:
    """Checks if a coordinate belongs to a team's sector."""
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)
