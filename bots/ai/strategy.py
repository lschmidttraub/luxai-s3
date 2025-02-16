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
import seaborn as sns
import matplotlib.pyplot as plt


class Strategy:
    def __init__(self, observation: Observation):
        self.obs = observation
        shape = self.obs.vision.shape
        # Maintain an mask array that shows all potential relic tiles
        # I think the best way of doing this to maintain an array of probabilities
        self.base = 0.025
        self.around_relic_prob = 0.2
        self.relic_tile_probs: np.ndarray = np.full(shape, self.base)
        Utils.is_symmetric(self.relic_tile_probs)
        # used to rule out tiles that cannot be relic tiles (either already visited or too far)
        self.possible_relic_tiles: np.ndarray = np.ones(shape, dtype=bool)
        # tracks if we still have to check the surroundings of tile to determine if it could best
        # a relic tile
        self.scanned_surroundings: np.ndarray = np.zeros(shape, dtype=bool)
        # set of relic nodes alreadz incorporated into the probability map
        self.processed_relic_nodes = set()
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
        self.update_potential_relic_tiles()
        self.update_roles()
        actions = np.zeros((self.obs.max_units, 3), dtype=int)
        self.unit_roles["attacker"].choose_actions(actions)
        self.unit_roles["scout"].choose_actions(actions)
        self.unit_roles["miner"].update_relic_probs(self.relic_tile_probs)
        self.unit_roles["miner"].choose_actions(actions)

        # if DEBUG:
        # Utils.tofile("debug/actions.txt", actions)
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

        # Check for new relic nodes
        for tile in self.obs.relic_nodes:
            if not tile in self.processed_relic_nodes:
                self.update_probs_around_node(tile)
                self.processed_relic_nodes.add(tile)
                self.processed_relic_nodes.add(Utils.symmetric(tile))

        # Check if some squares are too far from relic nodes to be relic tiles
        for tile in map(
            tuple, np.dstack(np.where(self.obs.exploration != UNKNOWN)).squeeze()
        ):
            # Check if scanning is even necessary
            if self.possible_relic_tiles[tile] and not self.scanned_surroundings[tile]:
                self.check_surroundings(tile)

        # Update probabilities of the tiles units are on with poisson binomial calculation
        pt_diff = self.obs.pt_diff
        # if pt_diff is None, then a new match has started
        if pt_diff is not None:
            # We store positions in numpy arrays of length 2 to make indexing the relic_tile_probs array
            # easier (just take the transpose of the position matrix)
            unit_pos = np.array(
                [
                    [x, y]
                    for _, ((x, y), _) in self.obs.units.items()
                    if self.possible_relic_tiles[x, y]
                ],
                dtype=int,
            )
            self.update_unit_probs(unit_pos, pt_diff[self.obs.player])

            enemy_unit_pos = np.array(
                [
                    [x, y]
                    for _, ((x, y), _) in self.obs.enemy_units.items()
                    if self.possible_relic_tiles[x, y]
                ],
                dtype=int,
            )
            self.update_unit_probs(enemy_unit_pos, pt_diff[1 - self.obs.player])

            if DEBUG:
                interval = 50
                if not self.obs.step % interval:
                    plot = sns.heatmap(
                        # take the transpose so that the heatmap looks like the visualizer
                        self.relic_tile_probs.T,
                        linewidth=0.5,
                        cmap="coolwarm",
                        vmin=-1,
                        vmax=1,
                    )
                    plt.savefig(f"debug/plots/probs_{self.obs.step}.jpg")
                    plt.clf()
                    Utils.tofile(
                        "debug/possible.txt", self.possible_relic_tiles.astype(int)
                    )

    def update_prob(self, pos: tuple[int, int], new_prob: float) -> None:
        """
        Updates the probability of the corresponding tile, as well as its symmetric image
        """
        self.relic_tile_probs[pos] = new_prob
        # since the map is symmetric, we always add a position as well as its corresponding image
        self.relic_tile_probs[Utils.symmetric(pos)] = new_prob

    def update_probs_around_node(self, node: tuple[int, int]):
        """
        Updates the probabilities of all tiles in the vicinity of a relic node.
        We assume that each tile around a relic node has a self.around_relic_prob chance of being a relic tile
        """
        x, y = node
        mask = [
            (i, j)
            for i in range(x - 2, x + 3)
            for j in range(y - 2, y + 3)
            if Utils.in_bounds((i, j)) and self.possible_relic_tiles[i, j]
        ]
        # we must increase the probability of each neihbour by a certain factor
        # we add a little over 4 to the divisor so that in the base case where only 1 tile has a non-zero probability, we still don't get a probability over 1
        for pos in mask:
            self.update_prob(
                pos,
                Utils.prob_mult(self.relic_tile_probs[pos], self.around_relic_prob),
            )
        if DEBUG:
            plot = sns.heatmap(
                self.relic_tile_probs.T, linewidth=0.5, cmap="Spectral", vmin=-1, vmax=1
            )
            plt.savefig(f"debug/plots/probs_{self.obs.step}.jpg")
            plt.clf()

    def check_surroundings(self, pos: tuple[int, int]) -> None:
        """
        Checks if all tiles in a perimeter of 2 around the position have been explored,
        and if none of these tiles are relic nodes. If this is the case, we can safely rule out this
        position as a relic tile. If some surrounding tiles remain unknown, we can't say, and if There
        is a relic_node within distance, we no longer need to call this function on the position
        """
        # if we run this function on an impossible tile without checking, the function might set it back to possible
        if not self.possible_relic_tiles[pos]:
            return
        for p in Utils.position_mask(pos, 2):
            if self.obs.exploration[p] == UNKNOWN:
                return
        # The entire surroundings have been scanned
        self.scanned_surroundings[pos] = True
        self.possible_relic_tiles[pos] = False
        for p in self.obs.relic_nodes:
            # if a relic_node is in range, then it is possible
            if Utils.max_dist(p, pos) <= 2:
                self.possible_relic_tiles[pos] = True

    def update_unit_probs(self, positions: np.ndarray, k: int) -> None:
        """
        updates the probabilities of the squares visited by units using Bayes formula and a variation of
        Poisson's formula
        """
        with open("debug/diff.txt", "a") as file:
            file.write(f"{self.obs.step} : {k} : {positions.tolist()}\n")
        if not positions.size:
            return
        if positions.shape[1] != 2:
            raise Exception("positions has wrong shape:", positions.shape)
        if not k:
            for p in map(tuple, positions):
                self.possible_relic_tiles[p] = False
                self.possible_relic_tiles[Utils.symmetric(p)] = False
                self.update_prob(p, -1)
            Utils.is_symmetric(self.relic_tile_probs)
            return
        P = self.relic_tile_probs[tuple(positions.T)]
        b, b_given_a = Utils.poisson_binomial(P, k)
        probs = Utils.bayes(P, b, b_given_a)

        for pos, prob in zip(map(tuple, positions), probs):
            self.update_prob(pos, prob)
        Utils.is_symmetric(self.relic_tile_probs)

    def update_roles(self) -> None:
        """
        First calculates the proportions scout, miner and attacker units we want,
        then updates the units list of each role
        """
        n_units = len(self.obs.units)
        # Doing this avoids divide by 0 errors
        if not n_units:
            self.unit_roles["scout"].update_units([])
            self.unit_roles["attacker"].update_units([])
            self.unit_roles["miner"].update_units([])
            return
        """
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
        """
        n_scouts, n_miners, n_attackers = 0, 0, 0
        if self.obs.relic_nodes:
            n_miners = math.ceil(n_units * 2 / 3)
        if self.obs.enemy_units and n_units - n_miners > 1:
            n_attackers = 1
        n_scouts = n_units - n_miners - n_attackers

        if DEBUG:
            with open("debug/props.txt", "a") as file:
                file.write(
                    f"step:{self.obs.step}, n:{n_units}, scout:{n_scouts}, miners:{n_miners}, attackers:{n_attackers}\n"
                )

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
