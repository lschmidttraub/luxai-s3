"""
Agent class groups Strategy and Observation together
"""

import numpy as np
from strategy import Strategy
from observation import Observation


class Agent:
    def __init__(self, player: str, env_cfg):
        self.player = 0 if player == "player_0" else 1
        self.opp = ~self.player
        self.observation = Observation(self.player, env_cfg)
        self.strategy = Strategy(self.observation)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.observation.update_observation(step, obs)
        return self.strategy.choose_action()
