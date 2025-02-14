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
        # Maintain an mask array that shows all potential relic tiles
        # I think the best way of doing this to maintain an array of probabilities
        self.base = 0.025
        self.relic_tile_probs: np.ndarray = np.full((W, H), self.base)
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
        # We assume that on average 5 relic tiles spawn for each relic node, and that the spawn
        # radius is 2 in maximum (Chebyshev) distance
        # This means that each tile in the vicinity of a relic node initialy has a 1/5 chance of being a relic tile
        # We can also make a rough estimate of the area covered by relic nodes:
        # Assuming, on average, 4 relic nodes (k ranges from 1 to 3), we can estimate the average covered area at 1/8 (100/576 + potential overlap),
        # which explains the starting probability of 0.025
        for tile in self.obs.new_explored_tiles:
            if tile in self.obs.relic_nodes:
                x, y = tile
                count = 25
                mask = [
                    (i, j) for i in range(x - 2, x + 3) for j in range(y - 2, y + 3)
                ]
                for pos in mask:
                    if not Utils.in_bounds(pos) or not self.relic_tile_probs[pos]:
                        count -= 1
                # we must increase the probability of each neihbour by a certain factor
                # we add a little over 4 to the divisor so that in the base case where only 1 tile has a non-zero probability, we still don't get a probability over 1
                prob = 5 / (count + 4)
                for pos in mask:
                    if Utils.in_bounds(pos) and self.relic_tile_probs[pos]:
                        self.update_prob(
                            pos, Utils.prob_mult(self.relic_tile_probs[pos], prob)
                        )

        pt_diff = self.obs.pt_diff
        # since the map is symmetric, we always add a position as well as its corresponding image

        pos1 = [p for _, (p, _) in self.obs.units.items()]
        pos2 = [p for _, (p, _) in self.obs.enemy_units.items()]
        self.update_visited_probs(pos1, pt_diff[0])
        self.update_visited_probs(pos2, pt_diff[1])

    def update_prob(self, pos: tuple[int, int], new_prob: float) -> None:
        """
        Updates the probability of the corresponding tile, as well as its symmetric image
        """
        self.relic_tile_probs[pos] = new_prob
        self.relic_tile_probs[Utils.symmetric(pos)] = new_prob

    def update_visited_probs(self, positions: list[tuple[int, int]], k: int) -> None:
        """
        updates the probabilities of the squares visited by units using Bayes formula and a variation of
        Poisson's formula
        """
        P = self.relic_tile_probs[positions]
        b_given_a = Utils.poisson(P, k)
        probs = Utils.bayes(P, 1, b_given_a)
        self.relic_tile_probs[positions] = probs
        for pos, prob in zip(positions, probs):
            self.update_prob(pos, prob)

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
