"""
tab AI Agents - GA-based and Random agents.

All agents work directly with both players - NO reverse_board.
The fitness function evaluates from a specified player's perspective.

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
"""

import random
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

from tab_game import (
    Board, TabRules, StickDice, TurnAllocation, SoldierMove,
    PLAYER_1, PLAYER_2, DEFAULT_COLUMNS
)


# =============================================================================
# Agent Interface
# =============================================================================

class TabAgent(ABC):
    """Abstract base class for tab AI agents."""
    
    @abstractmethod
    def choose_allocation(self, board: Board, player: int,
                          throw_values: List[int],
                          movable_soldiers: List[int],
                          rules: TabRules) -> TurnAllocation:
        """Choose how to allocate throw values to soldiers."""
        pass


# =============================================================================
# Fitness Evaluator
# =============================================================================

class FitnessEvaluator:
    """
    Evaluates board position from a specified player's perspective.
    Higher score = better for the evaluated player.
    """
    
    def __init__(self, columns: int = DEFAULT_COLUMNS):
        self.columns = columns
        self.board_size = 4 * columns
        self.rules = TabRules(columns)
    
    def evaluate(self, cells: List[int], player: int = PLAYER_1) -> float:
        """
        Evaluate board from 'player' perspective.
        Uses the original paper's 7-rule heuristic.
        """
        C = self.columns
        fit = 0.0
        
        my_total = 0
        opp_total = 0
        
        for i in range(self.board_size):
            row = i // C
            val = cells[i]
            
            if val * player > 0:
                # My soldier
                count = abs(val)
                my_total += count
                
                if self.rules.is_own_home(i, player):
                    fit += count * 3    # safe in home
                elif self.rules.is_enemy_home(i, player):
                    fit += count * 8    # safe in enemy home (frozen but alive)
                else:
                    fit += count * 2    # on battlefield
                
                # Grouping penalty
                if count > 1:
                    fit -= count * 3
                
                # Threat detection: find enemies who can reach me
                # Must check on the OPPONENT's track, not my own!
                # Because opponents approach from opposite entry point.
                opp = -player
                opp_track = self.rules.get_track(opp)
                # Find my position on opponent's track
                try:
                    my_idx_on_opp = opp_track.index(i)
                except ValueError:
                    my_idx_on_opp = -1
                
                if my_idx_on_opp >= 0:
                    # Check 1-6 steps behind me on opponent's track
                    # Probability-weighted penalties (Ahmad's insight):
                    #   dist 1 (value 1): 25.0%
                    #   dist 2 (value 2): 37.5% - MOST DANGEROUS
                    #   dist 3 (value 3): 25.0% - VERY DANGEROUS
                    #   dist 4 (value 4): 6.25%
                    #   dist 5 (combo):   ~4%
                    #   dist 6 (value 6): 6.25%
                    threat_weights = {1: 5.0, 2: 7.5, 3: 5.0, 4: 1.25, 5: 0.8, 6: 1.25}
                    for d in range(1, 7):
                        behind_idx = my_idx_on_opp - d
                        if behind_idx >= 0:
                            behind_pos = opp_track[behind_idx]
                            if cells[behind_pos] * player < 0:
                                enemy_count = abs(cells[behind_pos])
                                fit -= threat_weights.get(d, 1.0) * enemy_count
            
            elif val * player < 0:
                # Opponent soldier
                opp_total += abs(val)
        
        # Material advantage
        fit += (my_total - opp_total) * 5
        
        return fit
    
    def evaluate_advanced(self, cells: List[int], player: int = PLAYER_1,
                           original_cells: Optional[List[int]] = None) -> float:
        """
        Advanced evaluation with capture bonus and phase detection.
        """
        fit = self.evaluate(cells, player)
        
        # Capture bonus
        if original_cells is not None:
            orig_opp = sum(abs(c) for c in original_cells if c * player < 0)
            curr_opp = sum(abs(c) for c in cells if c * player < 0)
            captures = orig_opp - curr_opp
            if captures > 0:
                if curr_opp <= 3:  # endgame
                    fit += captures * 15
                else:
                    fit += captures * 8
        
        return fit


# =============================================================================
# Expert Fitness Evaluator - Ahmad's 7 Strategic Rules
# =============================================================================

class ExpertFitnessEvaluator:
    """
    Expert fitness function based on Ahmad Hassanat's strategic analysis
    of 4 expert games (280 decisions). Implements 7 rules derived from
    human expert play patterns.

    Key differences from original:
    - Positional: behind enemy = good, 2-3 ahead = very bad
    - Tactical: ambush positions at gateways (B7/B8, C1/B1)
    - Strategic: hide when outnumbered, sacrifice when ahead
    - Capture: context-dependent (good when ahead, risky when behind)
    """

    # Probability of opponent throwing exactly value v:
    #   1: 25%, 2: 37.5%, 3: 25%, 4: 6.25%, 6: 6.25%
    THREAT_BY_DISTANCE = {
        1: 0.250,   # dangerous
        2: 0.375,   # MOST dangerous
        3: 0.250,   # dangerous
        4: 0.0625,  # unlikely
        5: 0.04,    # multi-throw only
        6: 0.0625,  # unlikely
    }

    def __init__(self, columns: int = DEFAULT_COLUMNS):
        self.columns = columns
        self.board_size = 4 * columns
        self.rules = TabRules(columns)
        # Ambush positions near gateways
        # P1: B7(14),B8(15) watch D8->C8; C1(16),B1(8) watch crossing
        # P2: C2(17),C1(16) watch A1->B1; B8(15),C8(23) watch crossing
        self.p1_ambush = {14, 15, 16, 8}
        self.p2_ambush = {17, 16, 15, 23}

    def evaluate(self, cells: List[int], player: int = PLAYER_1) -> float:
        """
        Expert fitness: identical to original base + 5 small targeted additions.
        Strategy: DON'T break what works. Only add expert signals.
        """
        C = self.columns
        fit = 0.0
        my_total = 0
        opp_total = 0
        my_bf = []     # (pos, count) on battlefield
        opp_bf = []    # (pos, count) on battlefield
        my_enemy_home = 0

        for i in range(self.board_size):
            val = cells[i]
            if val * player > 0:
                count = abs(val)
                my_total += count
                if self.rules.is_own_home(i, player):
                    fit += count * 3
                elif self.rules.is_enemy_home(i, player):
                    fit += count * 8
                    my_enemy_home += count
                else:
                    fit += count * 2
                    my_bf.append((i, count))
                # Grouping penalty (same as original)
                if count > 1:
                    fit -= count * 3
                # Threat detection (same as original)
                opp_track = self.rules.get_track(-player)
                try:
                    my_idx_on_opp = opp_track.index(i)
                except ValueError:
                    my_idx_on_opp = -1
                if my_idx_on_opp >= 0:
                    threat_weights = {1: 5.0, 2: 7.5, 3: 5.0, 4: 1.25, 5: 0.8, 6: 1.25}
                    for d in range(1, 7):
                        behind_idx = my_idx_on_opp - d
                        if behind_idx >= 0:
                            behind_pos = opp_track[behind_idx]
                            if cells[behind_pos] * player < 0:
                                enemy_count = abs(cells[behind_pos])
                                fit -= threat_weights.get(d, 1.0) * enemy_count
            elif val * player < 0:
                count = abs(val)
                opp_total += count
                if not self.rules.is_own_home(i, -player) and \
                   not self.rules.is_enemy_home(i, -player):
                    opp_bf.append((i, count))

        # Material advantage (same as original)
        fit += (my_total - opp_total) * 5

        # ========= EXPERT ADDITIONS (small targeted bonuses) =========

        outnumbered = my_total < opp_total
        opp_track = self.rules.get_track(-player)
        ambush = self.p1_ambush if player == PLAYER_1 else self.p2_ambush

        # RULE 3: Extra hiding bonus when outnumbered (+3 per hidden soldier)
        if outnumbered and my_enemy_home > 0:
            fit += my_enemy_home * 3

        # RULE 5: Reduce threat penalty when ahead (willing to trade)
        # Re-scan threats and give back partial penalty
        if my_total > opp_total:
            for pos, count in my_bf:
                try:
                    my_idx = opp_track.index(pos)
                except ValueError:
                    continue
                for d in range(1, 4):  # distances 1-3 only
                    behind_idx = my_idx - d
                    if behind_idx >= 0:
                        behind_pos = opp_track[behind_idx]
                        if cells[behind_pos] * player < 0:
                            enemy_count = abs(cells[behind_pos])
                            tw = {1: 5.0, 2: 7.5, 3: 5.0}
                            # Give back 40% of penalty (net: 60% of original)
                            fit += tw.get(d, 1.0) * enemy_count * 0.4

        # RULE 1: Small bonus for being behind enemy (can follow/capture)
        for pos, count in my_bf:
            try:
                my_idx = opp_track.index(pos)
            except ValueError:
                continue
            for opp_pos, opp_count in opp_bf:
                try:
                    opp_idx = opp_track.index(opp_pos)
                except ValueError:
                    continue
                if my_idx < opp_idx and (opp_idx - my_idx) <= 6:
                    fit += 1.0  # small bonus per trailing pair

        # RULES 6 & 7: Small ambush position bonus
        for pos, count in my_bf:
            if pos in ambush:
                fit += 1.5  # modest gateway control bonus

        return fit

    def evaluate_advanced(self, cells: List[int], player: int = PLAYER_1,
                           original_cells: Optional[List[int]] = None) -> float:
        """Expert evaluation with capture context."""
        fit = self.evaluate(cells, player)

        if original_cells is not None:
            orig_opp = sum(abs(c) for c in original_cells if c * player < 0)
            curr_opp = sum(abs(c) for c in cells if c * player < 0)
            captures = orig_opp - curr_opp
            if captures > 0:
                my_total = sum(abs(c) for c in cells if c * player > 0)
                # RULE 5: Capturing when ahead - close the game
                if my_total > curr_opp:
                    fit += captures * 15
                # RULE 4: Raiding enemy home (few enemies left)
                elif curr_opp <= 3:
                    fit += captures * 12
                else:
                    fit += captures * 8

            # Game-winning move - absolute priority
            if curr_opp == 0:
                fit += 10000

        return fit


# =============================================================================
# Fuzzy Win-Win Fitness Evaluator
# Based on: Altarawneh, Hassanat, Tarawneh, Carfi, Almuhaimeed (2022)
# "Fuzzy Win-Win: A Novel Approach to Quantify Win-Win Using Fuzzy Logic"
# Mathematics 2022, 10, 884. https://doi.org/10.3390/math10060884
#
# Uses Bellman-Zadeh product T-norm aggregation:
#   f = u_S(S)^w1 x u_C(C)^w2,  where w1 + w2 = 1
#
# S = Score (capture opportunity), C = Cost (danger/exposure)
# Built ON TOP OF the original fitness (proven base).
# =============================================================================

class FuzzyWinWinFitness:
    """
    Fuzzy Win-Win fitness using product T-norm with exponential weights.

    For each battlefield soldier:
      u_S = membership in "good score" (capture opportunity nearby)
      u_C = membership in "low cost" (safety from capture)
      soldier_fuzzy = u_S^w1 x u_C^w2

    w1 + w2 = 1, shifted by material balance:
      ahead -> w1 > w2 (prioritize aggression)
      behind -> w2 > w1 (prioritize safety)

    The original fitness provides the BASE. The fuzzy win-win provides
    a BONUS/MODIFIER per battlefield soldier on top of that base.

    Threat probabilities use SUBSET-SUM analysis (2M Monte Carlo turns).
    """

    CAPTURE_PROB = {
        1:  0.286,   2:  0.632,   3:  0.572,   4:  0.227,
        5:  0.080,   6:  0.149,   7:  0.099,   8:  0.089,
        9:  0.073,  10:  0.043,  11:  0.022,  12:  0.023,
    }

    def __init__(self, columns: int = DEFAULT_COLUMNS, use_trades: bool = True):
        self.columns = columns
        self.board_size = 4 * columns
        self.rules = TabRules(columns)
        self.use_trades = use_trades
        self.base_eval = FitnessEvaluator(columns)  # original as base

    def _mu_score(self, pos, player, cells):
        """
        u_S: Score membership - capture opportunity.
        How many enemies can I reach, weighted by probability?
        Returns [0, 1]: 0 = no opportunity, 1 = maximum opportunity.
        """
        my_track = self.rules.get_track(player)
        try:
            my_idx = my_track.index(pos)
        except ValueError:
            return 0.0

        score = 0.0
        for d in range(1, 13):
            ahead_idx = my_idx + d
            if ahead_idx < len(my_track):
                ahead_pos = my_track[ahead_idx]
                if cells[ahead_pos] * player < 0:
                    score += self.CAPTURE_PROB.get(d, 0.01) * abs(cells[ahead_pos])

        # Normalize: S_max ~ 1.0 (one enemy at distance 2 = 0.632)
        # Multiple enemies or closer = higher. Clamp to [0, 1].
        return max(0.0, min(1.0, score))

    def _mu_cost(self, pos, player, cells):
        """
        u_C: Cost membership - safety (INVERTED: high = safe).
        How much threat am I under from enemies behind me?
        Returns [0, 1]: 0 = maximum danger, 1 = fully safe.
        u_C = (C_max - C) / (C_max - C_min) = 1 - threat
        """
        opp_track = self.rules.get_track(-player)
        try:
            my_idx = opp_track.index(pos)
        except ValueError:
            return 1.0  # not on opponent's track = safe

        threat = 0.0
        for d in range(1, 13):
            behind_idx = my_idx - d
            if behind_idx >= 0:
                behind_pos = opp_track[behind_idx]
                if cells[behind_pos] * player < 0:
                    threat += self.CAPTURE_PROB.get(d, 0.01) * abs(cells[behind_pos])

        # u_C = 1 - threat, clamped to [0, 1]
        return max(0.0, min(1.0, 1.0 - threat))

    def evaluate(self, cells: List[int], player: int = PLAYER_1) -> float:
        """
        Fuzzy Win-Win fitness = Original base + product T-norm modifier.

        f = base_fitness + Sum (u_S^w1 x u_C^w2) x scale x count
        """
        # Start with the ORIGINAL proven fitness as base
        fit = self.base_eval.evaluate(cells, player)

        # Collect material counts
        my_total = 0
        opp_total = 0
        my_battlefield = []

        for i in range(self.board_size):
            val = cells[i]
            if val * player > 0:
                my_total += abs(val)
                if not self.rules.is_own_home(i, player) and \
                   not self.rules.is_enemy_home(i, player):
                    my_battlefield.append((i, abs(val)))
            elif val * player < 0:
                opp_total += abs(val)

        # --- WEIGHTS: w1 + w2 = 1 ---
        # Material ratio determines priority
        total = my_total + opp_total
        if total > 0:
            ratio = my_total / total  # 0.5 when even
        else:
            ratio = 0.5

        # Game phase adjustment
        if total <= 4:  # endgame
            if my_total > opp_total:
                w1 = 0.8   # aggressive endgame - go for kill
            elif my_total < opp_total:
                w1 = 0.15  # defensive endgame - survive
            else:
                w1 = 0.3   # even endgame - safety-first
        else:
            # w1 follows material ratio, clamped to [0.25, 0.75]
            w1 = max(0.25, min(0.75, ratio))

        w2 = 1.0 - w1

        # --- PRODUCT T-NORM per battlefield soldier ---
        EPSILON = 0.01  # avoid u=0 making product = 0
        for pos, count in my_battlefield:
            mu_s = self._mu_score(pos, player, cells) + EPSILON
            mu_c = self._mu_cost(pos, player, cells) + EPSILON

            # f = u_S^w1 x u_C^w2 (Bellman-Zadeh product aggregation)
            fuzzy_ww = (mu_s ** w1) * (mu_c ** w2)

            # Scale and add as bonus on top of original
            fit += fuzzy_ww * 8.0 * count

        # --- TRADE EVALUATION (optional) ---
        if self.use_trades and my_battlefield:
            opp_track = self.rules.get_track(-player)
            my_track = self.rules.get_track(player)
            for pos, count in my_battlefield:
                my_capture_ev = 0.0
                try:
                    my_idx = my_track.index(pos)
                except ValueError:
                    my_idx = -1
                if my_idx >= 0:
                    for d in range(1, 7):
                        fwd_idx = my_idx + d
                        if fwd_idx < len(my_track):
                            fwd_pos = my_track[fwd_idx]
                            if cells[fwd_pos] * player < 0:
                                my_capture_ev += self.CAPTURE_PROB.get(d, 0.01)

                my_risk = 0.0
                try:
                    opp_idx = opp_track.index(pos)
                except ValueError:
                    opp_idx = -1
                if opp_idx >= 0:
                    for d in range(1, 7):
                        behind_idx = opp_idx - d
                        if behind_idx >= 0:
                            behind_pos = opp_track[behind_idx]
                            if cells[behind_pos] * player < 0:
                                my_risk += self.CAPTURE_PROB.get(d, 0.01)

                trade_ev = my_capture_ev - my_risk * count
                if my_total > opp_total:
                    fit += max(0, trade_ev) * 1.5
                elif my_total < opp_total:
                    fit += min(0, trade_ev) * 2.0
                else:
                    fit += trade_ev * 0.5

        # Hiding bonus when outnumbered
        if my_total < opp_total:
            enemy_home_count = sum(abs(cells[i]) for i in range(self.board_size)
                                   if cells[i] * player > 0 and
                                   self.rules.is_enemy_home(i, player))
            fit += enemy_home_count * 3

        # Last soldier protection
        if my_total == 1 and opp_total >= 1:
            for pos, count in my_battlefield:
                mu_c = self._mu_cost(pos, player, cells)
                fit += mu_c * 15.0

        return fit

    def evaluate_advanced(self, cells: List[int], player: int = PLAYER_1,
                           original_cells: Optional[List[int]] = None) -> float:
        """Fuzzy Win-Win with trade-aware capture bonus."""
        fit = self.evaluate(cells, player)

        if original_cells is not None:
            orig_opp = sum(abs(c) for c in original_cells if c * player < 0)
            curr_opp = sum(abs(c) for c in cells if c * player < 0)
            my_total = sum(abs(c) for c in cells if c * player > 0)
            captures = orig_opp - curr_opp

            if captures > 0:
                orig_my = sum(abs(c) for c in original_cells if c * player > 0)
                my_losses = orig_my - my_total

                if my_losses == 0:
                    fit += captures * 15  # pure capture
                else:
                    net = captures - my_losses
                    if my_total > curr_opp:
                        fit += captures * 12 + net * 5  # good trade
                    elif my_total == curr_opp:
                        fit += captures * 8  # even trade
                    else:
                        fit += captures * 5  # bad trade

                if curr_opp <= 2:
                    fit += captures * 10  # endgame urgency

            if curr_opp == 0:
                fit += 10000  # winning move

        return fit


# =============================================================================
# Random Agent
# =============================================================================

class RandomAgent(TabAgent):
    """Plays randomly - assigns each value to a random soldier."""
    
    def choose_allocation(self, board: Board, player: int,
                          throw_values: List[int],
                          movable_soldiers: List[int],
                          rules: TabRules) -> TurnAllocation:
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        
        n = len(movable_soldiers)
        alloc = TurnAllocation()
        assignments = [[] for _ in range(n)]
        
        for val in throw_values:
            assignments[random.randint(0, n - 1)].append(val)
        
        for i, values in enumerate(assignments):
            if values:
                alloc.soldier_moves.append(SoldierMove(
                    from_pos=movable_soldiers[i],
                    to_pos=movable_soldiers[i],
                    values_used=values))
        
        return alloc


# =============================================================================
# GA Engine
# =============================================================================

class GAEngine:
    """
    Genetic Algorithm for finding optimal dice allocation.
    Works directly with both players - no board reversal needed.
    """
    
    def __init__(self, population_size: int = 200,
                 generations: int = 100,
                 crossover_rate: float = 0.75,
                 mutation_rate: float = 0.10,
                 columns: int = DEFAULT_COLUMNS,
                 fitness_type: str = 'original'):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.columns = columns
        self.rules = TabRules(columns)
        if fitness_type == 'expert':
            self.fitness_eval = ExpertFitnessEvaluator(columns)
        elif fitness_type == 'fuzzy':
            self.fitness_eval = FuzzyWinWinFitness(columns, use_trades=True)
        elif fitness_type == 'fuzzy_notrade':
            self.fitness_eval = FuzzyWinWinFitness(columns, use_trades=False)
        else:
            self.fitness_eval = FitnessEvaluator(columns)
    
    def _random_alloc(self, values: List[int], n: int) -> List[List[int]]:
        alloc = [[] for _ in range(n)]
        for v in values:
            alloc[random.randint(0, n - 1)].append(v)
        return alloc
    
    def _histogram(self, values):
        h = {}
        for v in values:
            h[v] = h.get(v, 0) + 1
        return h
    
    def _repair(self, alloc, values):
        expected = self._histogram(values)
        actual = {}
        for i, vals in enumerate(alloc):
            cleaned = []
            for v in vals:
                cur = actual.get(v, 0)
                if cur < expected.get(v, 0):
                    cleaned.append(v)
                    actual[v] = cur + 1
            alloc[i] = cleaned
        for v, exp in expected.items():
            act = actual.get(v, 0)
            for _ in range(exp - act):
                alloc[random.randint(0, len(alloc) - 1)].append(v)
        return alloc
    
    def _crossover(self, p1, p2, values):
        n = len(p1)
        if n <= 1:
            return [list(v) for v in p1], [list(v) for v in p2]
        point = random.randint(1, n - 1)
        c1 = [list(v) for v in p1[:point]] + [list(v) for v in p2[point:]]
        c2 = [list(v) for v in p2[:point]] + [list(v) for v in p1[point:]]
        return self._repair(c1, values), self._repair(c2, values)
    
    def _mutate(self, alloc):
        n = len(alloc)
        if n < 2:
            return
        mt = random.randint(1, 3)
        if mt == 1 and n >= 2:
            i, j = random.sample(range(n), 2)
            alloc[i], alloc[j] = alloc[j], alloc[i]
        elif mt == 2:
            idx = random.randint(0, n - 1)
            if len(alloc[idx]) >= 2:
                i, j = random.sample(range(len(alloc[idx])), 2)
                alloc[idx][i], alloc[idx][j] = alloc[idx][j], alloc[idx][i]
        elif mt == 3:
            srcs = [(i, v) for i, v in enumerate(alloc) if v]
            if srcs:
                si, sv = random.choice(srcs)
                vi = random.randint(0, len(sv) - 1)
                val = sv.pop(vi)
                alloc[random.randint(0, n - 1)].append(val)
    
    def _simulate_and_evaluate(self, board, alloc, soldiers, player):
        """Simulate allocation and evaluate from player's perspective."""
        result_board = self.rules.simulate_allocation(
            board, alloc, soldiers, player)
        return self.fitness_eval.evaluate_advanced(
            result_board.cells, player, board.cells)
    
    def evolve(self, board: Board, player: int,
               throw_values: List[int],
               movable_soldiers: List[int]) -> Tuple[List[List[int]], float]:
        """Run GA to find best allocation. Returns (allocation, fitness)."""
        n = len(movable_soldiers)
        if n == 0 or not throw_values:
            return ([], 0.0)
        if n == 1:
            alloc = [list(throw_values)]
            f = self._simulate_and_evaluate(board, alloc, movable_soldiers, player)
            return (alloc, f)
        
        # Initialize population
        pop = []
        best_alloc = None
        best_fitness = float('-inf')
        
        for _ in range(self.population_size):
            alloc = self._random_alloc(throw_values, n)
            f = self._simulate_and_evaluate(board, alloc, movable_soldiers, player)
            pop.append((alloc, f))
            if f > best_fitness:
                best_fitness = f
                best_alloc = alloc
        
        # Evolution
        for gen in range(self.generations):
            new_pop = []
            for _ in range(self.population_size // 2):
                # Tournament selection
                i1, i2 = random.sample(range(len(pop)), 2)
                p1 = pop[i1] if pop[i1][1] > pop[i2][1] else pop[i2]
                i3, i4 = random.sample(range(len(pop)), 2)
                p2 = pop[i3] if pop[i3][1] > pop[i4][1] else pop[i4]
                
                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1[0], p2[0], throw_values)
                else:
                    c1 = [list(v) for v in p1[0]]
                    c2 = [list(v) for v in p2[0]]
                
                if random.random() < self.mutation_rate:
                    self._mutate(c1)
                if random.random() < self.mutation_rate:
                    self._mutate(c2)
                
                for alloc in [c1, c2]:
                    f = self._simulate_and_evaluate(
                        board, alloc, movable_soldiers, player)
                    new_pop.append((alloc, f))
                    if f > best_fitness:
                        best_fitness = f
                        best_alloc = alloc
            
            combined = pop + new_pop
            combined.sort(key=lambda x: x[1], reverse=True)
            pop = combined[:self.population_size]
        
        return (best_alloc, best_fitness)


# =============================================================================
# GA Agent
# =============================================================================

class GAAgent(TabAgent):
    """
    GA-based agent. Works directly for both P1 and P2.
    
    Levels:
        'beginner': small GA (100 pop, 25 gen)
        'advanced': larger GA (200 pop, 100 gen)
    """
    
    def __init__(self, level: str = 'advanced',
                 population_size: int = None,
                 generations: int = None,
                 columns: int = DEFAULT_COLUMNS,
                 fitness_type: str = 'original'):
        self.level = level
        self.columns = columns
        self.fitness_type = fitness_type
        
        if level == 'beginner':
            ps = population_size or 100
            gs = generations or 25
        else:
            ps = population_size or 200
            gs = generations or 100
        
        self.ga = GAEngine(
            population_size=ps,
            generations=gs,
            columns=columns,
            fitness_type=fitness_type)
    
    def choose_allocation(self, board: Board, player: int,
                          throw_values: List[int],
                          movable_soldiers: List[int],
                          rules: TabRules) -> TurnAllocation:
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        
        n = len(movable_soldiers)
        if n == 1:
            alloc = TurnAllocation()
            alloc.soldier_moves.append(SoldierMove(
                from_pos=movable_soldiers[0],
                to_pos=movable_soldiers[0],
                values_used=list(throw_values)))
            return alloc
        
        best_alloc, fitness = self.ga.evolve(
            board, player, throw_values, movable_soldiers)
        
        allocation = TurnAllocation(fitness=fitness)
        for i, values in enumerate(best_alloc):
            if values and i < len(movable_soldiers):
                allocation.soldier_moves.append(SoldierMove(
                    from_pos=movable_soldiers[i],
                    to_pos=movable_soldiers[i],
                    values_used=values))
        
        return allocation
