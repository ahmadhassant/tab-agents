"""
tab_look.py - LOOK Agent (one-ply lookahead with explicit tactical rules)
===========================================================================

Design goals (decided with Ahmad, May 2026):
  1. Inherit PSA's strengths: explicit fork rules, contextual home-hiding,
     biased allocation seeding, rescue-aware stacking.
  2. Add what PSA lacks: one-ply lookahead with dice-marginalized opponent.
  3. Continuous phase in [0, 1] instead of sharp thresholds.
  4. Multi-enemy home-hiding oracle (per-soldier escape probability).
  5. Dice-prior-corrected danger weights using THREAT_BY_DISTANCE.

This file does NOT modify any existing engine code. It adds a new agent
that fits the TabAgent interface. All evaluation uses board.cells +
board.frozen_at + board.frozen_queue - no global state.

Author: Claude + Ahmad Hassanat - Mutah University, Jordan
Project: tab agent comparison study (IEEE TOG)
"""

from __future__ import annotations
import random
from typing import List, Tuple, Optional, Callable

from tab_game import (Board, TabRules, TurnAllocation, SoldierMove,
                      StickDice, PLAYER_1, PLAYER_2, DEFAULT_COLUMNS,
                      STICK_VALUES, THROW_AGAIN)
from tab_ai import TabAgent, FitnessEvaluator


# ============================================================================
# Dice distribution (precomputed)
# ============================================================================
# P(single throw = v) for v in {1, 2, 3, 4, 6}:
#   0/4 flat: 1/16 -> value 6
#   1/4 flat: 4/16 -> value 1
#   2/4 flat: 6/16 -> value 2
#   3/4 flat: 4/16 -> value 3
#   4/4 flat: 1/16 -> value 4
SINGLE_THROW_PROB = {1: 4/16, 2: 6/16, 3: 4/16, 4: 1/16, 6: 1/16}

# THREAT_BY_DISTANCE = single-roll probability that opponent rolls value d
# (used to weight danger at distance d). Distance 5 needs combos (multi-throw).
THREAT_BY_DISTANCE = {
    1: SINGLE_THROW_PROB[1],   # 0.25
    2: SINGLE_THROW_PROB[2],   # 0.375
    3: SINGLE_THROW_PROB[3],   # 0.25
    4: SINGLE_THROW_PROB[4],   # 0.0625
    5: 0.04,                    # 1+4, 2+3, etc. - rough estimate
    6: SINGLE_THROW_PROB[6],   # 0.0625
}

# Most likely opponent dice sequences for lookahead, with their probabilities.
# Sequence ends on 2 or 3 (the only non-re-throw values). Captures top 10
# sequences by probability mass - covers >75% of all turns.
LIKELY_OPPONENT_SEQUENCES = [
    ([2], 6/16),                       # 0.375
    ([3], 4/16),                       # 0.250
    ([1, 2], (4/16) * (6/16)),         # 0.094
    ([1, 3], (4/16) * (4/16)),         # 0.0625
    ([4, 2], (1/16) * (6/16)),         # 0.0234
    ([4, 3], (1/16) * (4/16)),         # 0.0156
    ([6, 2], (1/16) * (6/16)),         # 0.0234
    ([6, 3], (1/16) * (4/16)),         # 0.0156
    ([1, 1, 2], (4/16)**2 * (6/16)),   # 0.0234
    ([1, 1, 3], (4/16)**2 * (4/16)),   # 0.0156
]


# ============================================================================
# Phase, geometry, helpers
# ============================================================================

def continuous_phase(board: Board) -> float:
    """Phase in [0, 1]. 0 = opening (16 soldiers alive), 1 = endgame (few left)."""
    alive = board.count_alive(PLAYER_1) + board.count_alive(PLAYER_2)
    # 16 alive -> phase 0. 2 alive -> phase ~0.875. 0 alive -> phase 1.0.
    return max(0.0, min(1.0, (16 - alive) / 16))


def is_enemy_home_pos(pos: int, player: int, C: int) -> bool:
    if player == PLAYER_1:
        return 3 * C <= pos < 4 * C
    return 0 <= pos < C


def is_own_home_pos(pos: int, player: int, C: int) -> bool:
    if player == PLAYER_1:
        return 0 <= pos < C
    return 3 * C <= pos < 4 * C


def is_battlefield(pos: int, C: int) -> bool:
    return C <= pos < 3 * C


# ============================================================================
# Danger model - dice-prior corrected
# ============================================================================

def compute_danger(pos: int, owner: int, board: Board,
                   rules: TabRules) -> float:
    """
    Probability opponent can capture this soldier next turn.
    Uses SINGLE_THROW_PROB (not unconditional weights) for accuracy.
    """
    if is_own_home_pos(pos, owner, rules.columns):
        return 0.0
    if is_enemy_home_pos(pos, owner, rules.columns):
        return 0.0  # enemy home soldiers are frozen anyway

    opp = -owner
    opp_track = rules.get_track(opp)
    try:
        my_idx_on_opp = opp_track.index(pos)
    except ValueError:
        return 0.0

    total_threat = 0.0
    for d in range(1, 7):
        if d not in THREAT_BY_DISTANCE:
            continue
        behind_idx = my_idx_on_opp - d
        if behind_idx < 0:
            continue
        behind_pos = opp_track[behind_idx]
        if board.cells[behind_pos] * opp > 0:
            # Enemy at distance d. Probability they capture = P(roll = d).
            total_threat += THREAT_BY_DISTANCE[d]

    return min(1.0, total_threat)


# ============================================================================
# Home-hiding oracle - multi-enemy escape probability
# ============================================================================

def home_hiding_escape_prob(home_pos: int, player: int, board: Board,
                             rules: TabRules) -> float:
    """
    Probability the soldier at home_pos will be captured on exit.
    Returns escape probability in [0, 1]. Higher = safer.

    Strategy:
      For each freed enemy soldier in enemy home, compute the probability
      that my soldier reaches the battlefield without being captured.
      A freed enemy soldier at column c, where the exit is past my column,
      threatens my soldier when they finally exit and turn the corner.
    """
    C = rules.columns
    if not is_enemy_home_pos(home_pos, player, C):
        return 0.0  # not even in enemy home
    opp = -player
    opp_home_start = 3 * C if opp == PLAYER_1 else 0
    opp_home_end = (3 + 1) * C if opp == PLAYER_1 else C
    # Wait - enemy home for opponent. opp home is where opp's soldiers START.
    # opp=PLAYER_1 -> opp home rows 0..C. opp=PLAYER_2 -> opp home rows 3C..4C.
    if opp == PLAYER_1:
        opp_home_start, opp_home_end = 0, C
    else:
        opp_home_start, opp_home_end = 3 * C, 4 * C

    my_col = home_pos % C
    pi_opp = board.player_index(opp)

    # Count freed enemies in their home; track which ones are "ahead" of me
    # along the exit direction.
    # P1 exits home at col 0 (lowest). So "ahead" of me = lower col than me.
    # P2 exits home at col 7 (highest). So "ahead" of me = higher col than me.
    threats_ahead = 0
    for i in range(opp_home_start, opp_home_end):
        if board.cells[i] * opp > 0:
            frozen = board.frozen_at[pi_opp].get(i, 0)
            total = abs(board.cells[i])
            freed = total - frozen
            if freed > 0:
                col = i % C
                # Am I behind (safe) or in front (death trap)?
                if opp == PLAYER_1:
                    # P1 exits at col 0 -> P1 freed soldiers move toward col -1.
                    # "Ahead of me" (i.e., already past me, can come back if
                    # they loop) = lower col. I'm in front of them if my col
                    # is lower than theirs.
                    if my_col < col:
                        threats_ahead += freed
                else:
                    # P2 exits at col 7 -> freed move toward higher col.
                    # I'm in front of them if my col is higher than theirs.
                    if my_col > col:
                        threats_ahead += freed

    # Heuristic: each threatened-position freed enemy has ~50% chance of
    # capturing me on exit (averaged over future dice). No threats = safe.
    if threats_ahead == 0:
        return 1.0  # fully safe
    return max(0.0, 1.0 - 0.5 * threats_ahead)


# ============================================================================
# LOOK Evaluator
# ============================================================================

class LookEvaluator:
    """
    Phase-aware, dice-prior-corrected board evaluator.
    Returns scalar score from `player`'s perspective.

    Args:
        use_continuous_phase: if True, phase in [0,1]; if False, discrete
            {opening, midgame, endgame} like PSA.
        binary_hide: if True, enemy-home hiding is {-15, +8} binary;
            if False, smooth interpolation in that range.
    """

    def __init__(self, columns: int = DEFAULT_COLUMNS,
                 use_continuous_phase: bool = True,
                 binary_hide: bool = True,
                 danger_scale: float = 10.0,
                 stacking_mode: str = 'quadratic',
                 material_scale: float = 15.0,
                 frozen_distinction: bool = True,
                 use_home_hiding: bool = True,
                 use_central_bonus: bool = True,
                 use_entry_pressure: bool = True):
        """
        Args:
            danger_scale: multiplier on dng^2 * cnt term. Default 10.
            stacking_mode: 'quadratic' (cnt^2 * 8), 'linear' (cnt * 4),
                'simple' (cnt * 3 -- like FitnessEvaluator's grouping penalty,
                applied to ALL stacks not just new ones), or 'off'.
            material_scale: multiplier on (my_total - op_total). Default 15;
                FitnessEvaluator uses 5.
            frozen_distinction: if True, separate frozen (-2) and freed (+3)
                in own home. If False, +3 per soldier regardless (FitnessEval).
        """
        self.columns = columns
        self.use_home_hiding = use_home_hiding
        self.use_central_bonus = use_central_bonus
        self.use_entry_pressure = use_entry_pressure
        self.rules = TabRules(columns)
        self.use_continuous_phase = use_continuous_phase
        self.binary_hide = binary_hide
        self.danger_scale = danger_scale
        self.stacking_mode = stacking_mode
        self.material_scale = material_scale
        self.frozen_distinction = frozen_distinction
        self.use_home_hiding = use_home_hiding
        self.use_central_bonus = use_central_bonus
        self.use_entry_pressure = use_entry_pressure

    def evaluate(self, board: Board, player: int,
                 original_board: Optional[Board] = None) -> float:
        C = self.columns
        rules = self.rules
        cells = board.cells

        my_total = board.count_alive(player)
        op_total = board.count_alive(-player)

        # Terminal
        if op_total == 0:
            return 100000.0
        if my_total == 0:
            return -100000.0

        # Phase
        if self.use_continuous_phase:
            phase = continuous_phase(board)
            phase_is_endgame = phase > 0.6
            phase_is_opening = phase < 0.15
        else:
            # Discrete (PSA-style)
            bf_tot = (board.count_on_battlefield(PLAYER_1)
                      + board.count_on_battlefield(PLAYER_2))
            if my_total <= 3 or op_total <= 3:
                phase = 1.0
                phase_is_endgame = True
                phase_is_opening = False
            elif my_total >= 7 and op_total >= 7 and bf_tot <= 4:
                phase = 0.0
                phase_is_endgame = False
                phase_is_opening = True
            else:
                phase = 0.5
                phase_is_endgame = False
                phase_is_opening = False

        score = 0.0

        # --- Material -------------------------------------------------
        score += (my_total - op_total) * self.material_scale

        # --- Captures (this turn) -------------------------------------
        if original_board is not None:
            orig_op = original_board.count_alive(-player)
            captures = orig_op - op_total
            if captures > 0:
                if phase_is_endgame:
                    score += captures * captures * 50.0
                elif phase_is_opening:
                    score += captures * 20.0
                else:
                    score += captures * 30.0

        # --- Per-soldier positional terms -----------------------------
        pi = board.player_index(player)
        op_bf = board.count_on_battlefield(-player)
        for i in range(rules.board_size):
            v = cells[i]
            if v * player <= 0:
                continue
            cnt = abs(v)

            if is_own_home_pos(i, player, C):
                if self.frozen_distinction:
                    frozen = board.frozen_at[pi].get(i, 0)
                    freed = cnt - frozen
                    if frozen > 0:
                        score -= frozen * 2.0
                    if freed > 0:
                        score += freed * 3.0
                else:
                    # FitnessEvaluator style: +3 per soldier, no distinction
                    score += cnt * 3.0

            elif is_enemy_home_pos(i, player, C):
                escape = home_hiding_escape_prob(i, player, board, rules)
                if self.binary_hide:
                    # PSA-style binary
                    if self.use_home_hiding:
                        if escape > 0.5:
                            score += cnt * 8.0
                        else:
                            score -= cnt * 15.0
                else:
                    # Smooth: range [-15, +8]
                    if self.use_home_hiding:
                        hide_value = -15.0 + 23.0 * escape
                        score += cnt * hide_value
                if my_total == 1 and escape > 0.5:
                    score += 25.0

            else:  # battlefield
                col = i % C
                if self.use_central_bonus and 3 <= col <= 4:
                    score += cnt * 2.0
                dng = compute_danger(i, player, board, rules)
                score -= dng * dng * cnt * self.danger_scale
                op_track = rules.get_track(player)
                try:
                    my_idx = op_track.index(i)
                    entry_dist = rules.track_len - 1 - my_idx
                except ValueError:
                    entry_dist = 99
                if self.use_entry_pressure and op_bf > 0 and entry_dist <= 3:
                    score += cnt * 4.0

        # --- Stacking penalty -------------------------------------------
        # 'simple' mode: penalize ALL stacks > 1 (FitnessEvaluator style)
        if self.stacking_mode == 'simple':
            for i in range(rules.board_size):
                v = cells[i]
                if v * player <= 0:
                    continue
                cnt = abs(v)
                if cnt > 1:
                    score -= cnt * 3.0
        # 'quadratic'/'linear': rescue-aware, only NEW stacks
        elif original_board is not None and self.stacking_mode != 'off':
            for i in range(rules.board_size):
                v = cells[i]
                if v * player <= 0:
                    continue
                cnt = abs(v)
                if cnt <= 1:
                    continue
                orig_v = original_board.cells[i]
                orig_cnt = abs(orig_v) if orig_v * player > 0 else 0
                if cnt > orig_cnt:
                    is_rescue = False
                    for j in range(rules.board_size):
                        orig_vj = original_board.cells[j]
                        if orig_vj * player <= 0 or j == i:
                            continue
                        new_v = cells[j]
                        new_cnt = abs(new_v) if new_v * player > 0 else 0
                        orig_cnt_j = abs(orig_vj)
                        if new_cnt < orig_cnt_j:
                            src_dng = compute_danger(j, player, original_board, rules)
                            dst_dng = compute_danger(i, player, board, rules)
                            if src_dng > 0.3 and dst_dng < src_dng:
                                is_rescue = True
                                break
                    if not is_rescue:
                        if self.stacking_mode == 'linear':
                            score -= cnt * 4.0
                        else:  # 'quadratic'
                            if phase_is_endgame:
                                score -= cnt * 2.0
                            else:
                                score -= cnt * cnt * 8.0

        # --- Mobility -------------------------------------------------
        my_frozen = len(board.frozen_queue[pi])
        if my_frozen == 0 and not phase_is_endgame:
            score += 10.0

        return score


# ============================================================================
# Allocation generation - enumerate or sample
# ============================================================================

def histogram(values):
    h = {}
    for v in values:
        h[v] = h.get(v, 0) + 1
    return h


def enumerate_allocations(values: List[int], n_soldiers: int,
                           max_total: int = 10000) -> Optional[List[List[List[int]]]]:
    """
    Enumerate all distinct allocations of values across n_soldiers.
    Returns None if total > max_total (caller should sample instead).
    Uses histogram of values to avoid duplicates from permutations of same value.
    """
    if not values or n_soldiers == 0:
        return []
    hist = histogram(values)
    unique_vals = sorted(hist.keys())
    counts = [hist[v] for v in unique_vals]
    # Estimate: for each unique value with count c, partition c indistinguishable
    # items into n bins = C(c + n - 1, n - 1). Product across values.
    from math import comb
    estimate = 1
    for c in counts:
        estimate *= comb(c + n_soldiers - 1, n_soldiers - 1)
    if estimate > max_total:
        return None

    # Recursive enumeration
    results = []
    def go(val_idx, current_alloc):
        if val_idx == len(unique_vals):
            results.append([list(s) for s in current_alloc])
            return
        v = unique_vals[val_idx]
        c = counts[val_idx]
        # Distribute c copies of v across n_soldiers bins
        def distribute(remaining, bin_idx):
            if bin_idx == n_soldiers - 1:
                for _ in range(remaining):
                    current_alloc[bin_idx].append(v)
                go(val_idx + 1, current_alloc)
                for _ in range(remaining):
                    current_alloc[bin_idx].pop()
                return
            for k in range(remaining + 1):
                for _ in range(k):
                    current_alloc[bin_idx].append(v)
                distribute(remaining - k, bin_idx + 1)
                for _ in range(k):
                    current_alloc[bin_idx].pop()
        distribute(c, 0)

    go(0, [[] for _ in range(n_soldiers)])
    return results


def biased_sample_allocations(values: List[int], soldiers: List[int],
                                player: int, board: Board, rules: TabRules,
                                n_samples: int = 200) -> List[List[List[int]]]:
    """PSA-style biased sampling: seed with patterns, fill with weighted random."""
    n = len(soldiers)
    allocs = []

    # (a) All-to-one for each soldier
    for i in range(n):
        a = [[] for _ in range(n)]
        for v in values:
            a[i].append(v)
        allocs.append(a)

    # (b) Spread (one value per soldier, if feasible)
    if len(values) <= n:
        a = [[] for _ in range(n)]
        for vi, v in enumerate(values):
            a[vi].append(v)
        allocs.append(a)

    # (c) Free-first: 1s to early soldiers
    if 1 in values and n > 1:
        a = [[] for _ in range(n)]
        fi = 0
        for v in values:
            if v == 1 and fi < n:
                a[fi].append(v)
                fi += 1
            else:
                a[min(fi, n - 1)].append(v)
        allocs.append(a)

    # (d) Biased random fill
    weights = []
    for p in soldiers:
        dng = compute_danger(p, player, board, rules)
        has_enemy_near = 0
        # Cheap proximity check: any opponent within +/-4 along my track
        my_track = rules.get_track(player)
        try:
            my_idx = my_track.index(p)
            for d in range(1, 5):
                for sign in (1, -1):
                    ni = my_idx + sign * d
                    if 0 <= ni < rules.track_len:
                        if board.cells[my_track[ni]] * player < 0:
                            has_enemy_near = 1
                            break
                if has_enemy_near:
                    break
        except ValueError:
            pass
        weights.append(1.0 + dng * 2.0 + has_enemy_near * 3.0)

    while len(allocs) < n_samples:
        a = [[] for _ in range(n)]
        for v in values:
            total_w = sum(weights)
            if total_w <= 0:
                idx = random.randint(0, n - 1)
            else:
                r = random.random() * total_w
                cum = 0
                idx = n - 1
                for i, w in enumerate(weights):
                    cum += w
                    if r < cum:
                        idx = i
                        break
            a[idx].append(v)
        allocs.append(a)

    return allocs


def get_candidate_allocations(values: List[int], soldiers: List[int],
                                player: int, board: Board, rules: TabRules,
                                enum_threshold: int = 10000,
                                sample_n: int = 200) -> List[List[List[int]]]:
    """Enumerate if feasible, otherwise sample."""
    enumerated = enumerate_allocations(values, len(soldiers),
                                         max_total=enum_threshold)
    if enumerated is not None:
        return enumerated
    return biased_sample_allocations(values, soldiers, player, board, rules,
                                       n_samples=sample_n)


# ============================================================================
# Smart fork picker (replaces simulate_allocation's primitive scorer)
# ============================================================================

def smart_fork_picker(board: Board, pos: int, val: int, player: int,
                       dests: List[int], rules: TabRules,
                       evaluator: LookEvaluator) -> int:
    """
    Pick the better fork using full board evaluation.
    Returns index into dests (0 or 1).
    """
    if len(dests) <= 1:
        return 0
    best_idx, best_score = 0, float('-inf')
    sign = 1 if player == PLAYER_1 else -1
    for ci, d in enumerate(dests):
        sim = board.clone()
        total = abs(sim.cells[pos])
        pi = sim.player_index(player)
        frozen = sim.frozen_at[pi].get(pos, 0)
        m = total - frozen
        if val == 1 and m > 1:
            m = 1
        if m <= 0:
            continue
        sim.cells[pos] -= m * sign
        if sim.cells[d] * player < 0:
            sim.cells[d] = 0
        sim.cells[d] += m * sign
        score = evaluator.evaluate(sim, player, original_board=None)
        if score > best_score:
            best_score = score
            best_idx = ci
    return best_idx


def simulate_with_smart_forks(board: Board, allocation: List[List[int]],
                                soldiers: List[int], player: int,
                                rules: TabRules,
                                evaluator: LookEvaluator) -> Board:
    """
    Apply allocation to a cloned board, using smart_fork_picker at every fork.
    This is the lookahead-aware version of rules.simulate_allocation.
    """
    sim = board.clone()
    sign = 1 if player == PLAYER_1 else -1
    pi = sim.player_index(player)
    for i, values in enumerate(allocation):
        if i >= len(soldiers) or not values:
            continue
        pos = soldiers[i]
        for val in values:
            # Freeing check
            if val == 1 and rules.is_own_home(pos, player) and \
                    sim.frozen_at[pi].get(pos, 0) > 0 and \
                    rules.can_free(pos, player, sim):
                rules.free_specific_soldier(sim, player, pos)
                continue
            total = abs(sim.cells[pos])
            frozen = sim.frozen_at[pi].get(pos, 0)
            m = total - frozen
            if m <= 0:
                continue
            dests = rules.compute_destinations(pos, val, player, sim)
            if not dests:
                continue
            choice = smart_fork_picker(sim, pos, val, player, dests, rules,
                                         evaluator)
            new_pos, cap, _ = rules.apply_single_move(sim, pos, val, player,
                                                       choice=choice)
            pos = new_pos
    return sim


# ============================================================================
# One-ply lookahead with dice-marginalized opponent
# ============================================================================

def opponent_best_response_score(board: Board, opponent: int, rules: TabRules,
                                   evaluator: LookEvaluator,
                                   my_player: int) -> float:
    """
    For a given opponent dice sequence (already rolled), find their best
    allocation and return the resulting score FROM MY PERSPECTIVE.

    Opponent uses LOOK's own evaluator at depth 0 (self-play assumption).
    """
    # Note: this function is called from `lookahead_score` AFTER dice
    # have been determined. But for marginalization we sample multiple
    # dice outcomes. So this is really just one outcome's response.
    pass  # implemented inline in lookahead_score for efficiency


def lookahead_score(board_after_my_move: Board, my_player: int,
                     rules: TabRules, evaluator: LookEvaluator,
                     n_sequences: int = 5,
                     opp_n_candidates: int = 30,
                     simple_fork_picker: bool = True) -> float:
    """
    Expected score after opponent's best response, marginalized over the
    top-n most likely dice sequences.

    Opponent uses the SAME candidate-generation pipeline as LOOK itself
    (biased sampling, capped at opp_n_candidates for compute). This is
    true self-play - not the crippled toy model that the first version had.

    Args:
        simple_fork_picker: if True, use cheap fork heuristic during sim
            (avoids feedback loop with evaluator). Default True.
    """
    opp = -my_player
    # Terminal - if opponent has no soldiers left, we already won
    if board_after_my_move.count_alive(opp) == 0:
        return 100000.0
    if board_after_my_move.count_alive(my_player) == 0:
        return -100000.0

    # Use top-n sequences from precomputed list
    sequences = LIKELY_OPPONENT_SEQUENCES[:n_sequences]
    total_prob = sum(p for _, p in sequences)
    if total_prob == 0:
        return evaluator.evaluate(board_after_my_move, my_player)

    expected_score = 0.0
    for seq, prob in sequences:
        # Opponent's response to this sequence
        opp_board = board_after_my_move.clone()
        remaining = list(seq)
        # Auto-free as many soldiers as possible
        free_count = remaining.count(1)
        pi_opp = opp_board.player_index(opp)
        for _ in range(free_count):
            if opp_board.frozen_queue[pi_opp]:
                next_pos = opp_board.frozen_queue[pi_opp][0]
                if rules.can_free(next_pos, opp, opp_board):
                    rules.free_specific_soldier(opp_board, opp, next_pos)
                    remaining.remove(1)

        # Get opponent's movable soldiers
        opp_movable = rules.get_movable_soldiers(opp_board, opp, remaining)
        if not opp_movable or not remaining:
            score = evaluator.evaluate(opp_board, my_player,
                                        original_board=board_after_my_move)
        else:
            # REAL opponent model: same candidate generation as self
            candidates = get_candidate_allocations(
                remaining, opp_movable, opp, opp_board, rules,
                enum_threshold=2000,  # smaller for lookahead speed
                sample_n=opp_n_candidates)
            best_opp_alloc = None
            best_opp_score = float('-inf')
            for alloc in candidates:
                if simple_fork_picker:
                    sim = _simulate_simple_forks(opp_board, alloc, opp_movable,
                                                   opp, rules)
                else:
                    sim = simulate_with_smart_forks(opp_board, alloc,
                                                     opp_movable, opp, rules,
                                                     evaluator)
                opp_score = evaluator.evaluate(sim, opp,
                                                original_board=opp_board)
                if opp_score > best_opp_score:
                    best_opp_score = opp_score
                    best_opp_alloc = alloc
            if best_opp_alloc is not None:
                if simple_fork_picker:
                    final_board = _simulate_simple_forks(opp_board,
                                                            best_opp_alloc,
                                                            opp_movable, opp,
                                                            rules)
                else:
                    final_board = simulate_with_smart_forks(opp_board,
                                                              best_opp_alloc,
                                                              opp_movable,
                                                              opp, rules,
                                                              evaluator)
            else:
                final_board = opp_board
            # Score from MY perspective after opponent's best play
            score = evaluator.evaluate(final_board, my_player,
                                        original_board=board_after_my_move)

        expected_score += prob * score

    return expected_score / total_prob


def _simulate_simple_forks(board: Board, allocation: List[List[int]],
                             soldiers: List[int], player: int,
                             rules: TabRules) -> Board:
    """
    Apply allocation with cheap fork-picking heuristic (capx10 + enemy_homex5).
    Used during lookahead to break feedback loops with the evaluator.
    Mirrors the heuristic in rules.simulate_allocation.
    """
    sim = board.clone()
    pi = sim.player_index(player)
    for i, values in enumerate(allocation):
        if i >= len(soldiers) or not values:
            continue
        pos = soldiers[i]
        for val in values:
            if val == 1 and rules.is_own_home(pos, player) and \
                    sim.frozen_at[pi].get(pos, 0) > 0 and \
                    rules.can_free(pos, player, sim):
                rules.free_specific_soldier(sim, player, pos)
                continue
            total = abs(sim.cells[pos])
            frozen = sim.frozen_at[pi].get(pos, 0)
            m = total - frozen
            if m <= 0:
                continue
            dests = rules.compute_destinations(pos, val, player, sim)
            if not dests:
                continue
            choice = 0
            if len(dests) > 1:
                best_score = float('-inf')
                for ci, d in enumerate(dests):
                    s = 0.0
                    if sim.cells[d] * player < 0:
                        s += abs(sim.cells[d]) * 10.0
                    if rules.is_enemy_home(d, player):
                        s += 5.0
                    if s > best_score:
                        best_score = s
                        choice = ci
            new_pos, cap, _ = rules.apply_single_move(sim, pos, val, player,
                                                       choice=choice)
            pos = new_pos
    return sim


# ============================================================================
# LOOK Agent
# ============================================================================

class LookAgent(TabAgent):
    """
    LOOK = One-ply lookahead with tactical rules.

    Pipeline per turn:
      1. Generate candidate allocations (enumerate <=10k, else 200 samples).
      2. For each, simulate with SIMPLE fork picking (avoids feedback loop).
      3. Score = lookahead expected value (opponent's best response,
         marginalized over top-5 dice sequences).
      4. Pick best.

    Tuning knobs:
      - enum_threshold: max allocations to enumerate (default 10k)
      - sample_n: fallback sample size (default 200)
      - lookahead_seqs: number of opponent dice sequences (default 5)
      - opp_n_candidates: opponent allocation candidates in lookahead (default 30)
      - use_lookahead: if False, score by static eval only (for ablation)
      - use_rules: if False, use original FitnessEvaluator (for ablation)
      - use_continuous_phase: if False, use discrete PSA-style phases (ablation)
      - smart_fork_picker_self: if True, use evaluator for own fork picks
        (off by default - causes feedback loop, slower)
    """

    def __init__(self, columns: int = DEFAULT_COLUMNS,
                 enum_threshold: int = 10000,
                 sample_n: int = 200,
                 lookahead_seqs: int = 5,
                 opp_n_candidates: int = 30,
                 use_lookahead: bool = True,
                 use_rules: bool = True,
                 use_continuous_phase: bool = True,
                 smart_fork_picker_self: bool = False,
                 binary_hide: bool = True,
                 danger_scale: float = 10.0,
                 stacking_mode: str = 'quadratic',
                 material_scale: float = 15.0,
                 frozen_distinction: bool = True,
                 use_home_hiding: bool = True,
                 use_central_bonus: bool = True,
                 use_entry_pressure: bool = True):
        self.columns = columns
        self.enum_threshold = enum_threshold
        self.sample_n = sample_n
        self.lookahead_seqs = lookahead_seqs
        self.opp_n_candidates = opp_n_candidates
        self.use_lookahead = use_lookahead
        self.use_rules = use_rules
        self.use_continuous_phase = use_continuous_phase
        self.smart_fork_picker_self = smart_fork_picker_self
        self.binary_hide = binary_hide
        self.danger_scale = danger_scale
        self.stacking_mode = stacking_mode
        self.material_scale = material_scale
        self.frozen_distinction = frozen_distinction
        self.rules = TabRules(columns)
        if use_rules:
            self.evaluator = LookEvaluator(
                columns,
                use_continuous_phase=use_continuous_phase,
                binary_hide=binary_hide,
                danger_scale=danger_scale,
                stacking_mode=stacking_mode,
                material_scale=material_scale,
                frozen_distinction=frozen_distinction,
                use_home_hiding=use_home_hiding,
                use_central_bonus=use_central_bonus,
                use_entry_pressure=use_entry_pressure)
        else:
            self.evaluator = _LegacyEvaluatorAdapter(FitnessEvaluator(columns))

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

        candidates = get_candidate_allocations(
            throw_values, movable_soldiers, player, board, self.rules,
            enum_threshold=self.enum_threshold, sample_n=self.sample_n)

        if not candidates:
            return TurnAllocation()

        best_alloc, best_score = None, float('-inf')
        for alloc in candidates:
            # Use SIMPLE fork picker during simulation (avoids feedback loop)
            if self.smart_fork_picker_self:
                sim = simulate_with_smart_forks(board, alloc, movable_soldiers,
                                                  player, self.rules,
                                                  self.evaluator)
            else:
                sim = _simulate_simple_forks(board, alloc, movable_soldiers,
                                                player, self.rules)
            if self.use_lookahead:
                score = lookahead_score(sim, player, self.rules,
                                          self.evaluator,
                                          n_sequences=self.lookahead_seqs,
                                          opp_n_candidates=self.opp_n_candidates,
                                          simple_fork_picker=True)
            else:
                score = self.evaluator.evaluate(sim, player,
                                                  original_board=board)
            if score > best_score:
                best_score = score
                best_alloc = alloc

        if best_alloc is None:
            return TurnAllocation()

        allocation = TurnAllocation(fitness=best_score)
        for i, values in enumerate(best_alloc):
            if values and i < len(movable_soldiers):
                allocation.soldier_moves.append(SoldierMove(
                    from_pos=movable_soldiers[i],
                    to_pos=movable_soldiers[i],
                    values_used=values))
        return allocation


class _LegacyEvaluatorAdapter:
    """Adapt FitnessEvaluator to LookEvaluator's signature."""

    def __init__(self, fe: FitnessEvaluator):
        self.fe = fe

    def evaluate(self, board: Board, player: int,
                 original_board: Optional[Board] = None) -> float:
        orig_cells = original_board.cells if original_board is not None else None
        return self.fe.evaluate_advanced(board.cells, player,
                                          original_cells=orig_cells)


# ============================================================================
# Convenience constructors for ablation
# ============================================================================

def make_look_full(columns: int = DEFAULT_COLUMNS) -> LookAgent:
    """Full LOOK with all features."""
    return LookAgent(columns=columns, use_lookahead=True, use_rules=True,
                       use_continuous_phase=True)


def make_look_no_lookahead(columns: int = DEFAULT_COLUMNS) -> LookAgent:
    """LOOK without one-ply lookahead (pure heuristic)."""
    return LookAgent(columns=columns, use_lookahead=False, use_rules=True)


def make_look_no_rules(columns: int = DEFAULT_COLUMNS) -> LookAgent:
    """LOOK with original FitnessEvaluator (tests if rules matter)."""
    return LookAgent(columns=columns, use_lookahead=True, use_rules=False)


def make_look_no_lookahead_no_rules(columns: int = DEFAULT_COLUMNS) -> LookAgent:
    """Pure baseline: no lookahead, no expert rules."""
    return LookAgent(columns=columns, use_lookahead=False, use_rules=False)


if __name__ == "__main__":
    # Quick sanity test
    from tab_game import Board
    rules = TabRules()
    board = Board()
    agent = LookAgent()
    print(f"LOOK agent created. Phase at start: {continuous_phase(board):.3f}")
    print(f"Danger at A1 for P1: {compute_danger(0, PLAYER_1, board, rules):.3f}")
    print(f"Escape prob A1 for P2 (hiding in P1 home): "
          f"{home_hiding_escape_prob(0, PLAYER_2, board, rules):.3f}")

    # Run a move
    throws = [2, 4]
    movable = rules.get_movable_soldiers(board, PLAYER_1, throws)
    print(f"Movable soldiers: {[board.position_label(p) for p in movable]}")
    if movable:
        alloc = agent.choose_allocation(board, PLAYER_1, throws, movable, rules)
        print(f"Chose allocation with fitness {alloc.fitness:.2f}")
        for sm in alloc.soldier_moves:
            print(f"  {board.position_label(sm.from_pos)}: {sm.values_used}")
