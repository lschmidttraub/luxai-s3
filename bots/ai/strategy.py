"""
The Strategy class is made to handle decision-making, namely choosing actions for each unit.
"""

import numpy as np
from observation import Observation
from utils import *
from roles.scout import Scout, Scouts
from roles.attacker import Attacker, Attackers
from roles.miner import Miner, Miners
from roles.unit import Unit


class Strategy:
    def __init__(self, observation: Observation):
        self.obs = observation
        H, W = self.obs.H, self.obs.W
        self.relic_tile_mask: np.ndarray = np.zeros((W, H)).astype(bool)
        self.relic_tile_probs: np.ndarray = np.zeros((W, H))
        # create arrays of different roles
        self.unit_roles: dict = {
            "scout": Scouts(self.obs),
            "attacker": Attackers(self.obs),
            "miner": Miners(self.obs),
        }
        self.all_relics_discovered = False

    def choose_action(self) -> np.ndarray:
        """
        Choose the actions for each different class of unit (see roles)
        """
        self.update_roles()
        actions = np.zeros((self.obs.max_units, 3), dtype=int)
        self.unit_roles["attacker"].choose_actions(actions)
        self.unit_roles["scout"].choose_actions(actions)
        self.unit_roles["miner"].choose_actions(actions)

        if DEBUG:
            Utils.tofile("debug/actions.txt", actions)
        return actions

    def update_potential_relic_tiles(self):
        """
        Used to track which tiles could potentially be relic tiles (i.e. which tiles give points)
        INCOMPLETE
        """
        pt_diff = self.obs.pt_diff
        pos1 = [pos for _, (pos, e) in self.obs.units.items()]
        pos2 = [pos for _, (pos, e) in self.obs.enemy_units.items()]
        if ~pt_diff[0]:
            self.relic_tile_mask[pos1] = True
        if ~pt_diff[1]:
            self.relic_tile_mask[pos2] = True

    def update_roles(self) -> None:
        """
        First calculates the proportions scout, miner and attacker units we want,
        then updates the units list of each role
        """
        self.all_relics_discovered = self.obs.found_all_relics()

        n_units = len(self.obs.units)
        # Doing this avoids divide by 0 errors
        if not n_units:
            self.unit_roles["scout"].update_units([])
            self.unit_roles["attacker"].update_units([])
            self.unit_roles["miner"].update_units([])
            return

        # these formulas are probably super shitty, NEED TO IMPROVE
        scout_prop = self.obs.undiscovered_count() / (self.obs.H * self.obs.W)
        attacker_prop = len(self.obs.enemy_units) / n_units
        miner_prop = (
            len(self.obs.relic_nodes) + len(self.obs.relic_tiles)
        ) / self.obs.max_units
        total = scout_prop + attacker_prop + miner_prop
        scout_prop /= total
        attacker_prop /= total
        miner_prop /= total

        # rounding products to the nearest integer (instead of rounding down) helps us
        # avoid the edge case where n_miners = 1 but no relic nodes have been discovered
        n_scouts = int(round(n_units * scout_prop))
        n_attackers = int(round(n_units * attacker_prop))
        n_miners = n_units - n_scouts - n_attackers

        scouts = self.unit_roles["scout"].units
        attackers = self.unit_roles["attacker"].units
        miners = self.unit_roles["miner"].units

        # The role of this array is to show which unit ids are already being used
        taken = [False for _ in range(self.obs.max_units)]

        def remove_excess_units(l: list[Unit], expected_length) -> None:
            for unit in l:
                # There might be an edge case in which a unit with a certain id is killed and a new
                # unit with the same id is created
                if (
                    not unit.id in self.obs.units
                    # Check if the position of the unit is indeed the predicted position of the unit
                    or unit.pos != self.obs.units[unit.id][0]
                ):
                    l.remove(unit)
                else:
                    taken[unit.id] = True
            # remove extra units
            while len(l) > expected_length:
                l.pop()

        remove_excess_units(scouts, n_scouts)
        remove_excess_units(attackers, n_attackers)
        remove_excess_units(miners, n_miners)

        def add_new_units(l, expected_length, unit_type):
            for u_id in self.obs.units:
                if len(l) == expected_length:
                    return
                if not taken[u_id]:
                    pos, energy = self.obs.units[u_id]
                    l.append(unit_type(u_id, pos, energy))
                    taken[u_id] = True

        add_new_units(scouts, n_scouts, Scout)
        add_new_units(attackers, n_attackers, Attacker)
        add_new_units(miners, n_miners, Miner)

        self.unit_roles["scout"].update_units(scouts)
        self.unit_roles["attacker"].update_units(attackers)
        self.unit_roles["miner"].update_units(miners)

        if DEBUG:
            with open("debug/log.txt", "w") as log:
                log.write(f"{scouts}, {attackers}, {miners}")
