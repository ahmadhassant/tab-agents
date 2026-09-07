"""
tab_psa.py - PSA, the Phase-Sampling Agent.
===========================================

A greedy baseline designed independently of the LOOK/TLA family: biased
sampling of candidate allocations (no enumeration, no lookahead), scored by
PhaseEvaluator - discrete game phases, binary safe/unsafe home hiding, an
ambush bonus for soldiers hidden in the enemy starting row, and a
rescue-aware quadratic stacking penalty applied against the pre-move board.

Included in the tournament as a check on whether the ranking among the
LOOK/TLA agents is an artifact of their shared design lineage.
"""

from __future__ import annotations
import random
from typing import List, Optional

from tab_game import (Board, TabRules, TurnAllocation, SoldierMove,
                      PLAYER_1, PLAYER_2, DEFAULT_COLUMNS)
from tab_ai import TabAgent
from tab_look import (compute_danger, home_hiding_escape_prob,
                       is_enemy_home_pos, is_own_home_pos,
                       biased_sample_allocations, simulate_with_smart_forks,
                       _LegacyEvaluatorAdapter)


def psa_phase(board: Board) -> str:
    """Discrete phase: opening / development / midgame / endgame."""
    my_tot = board.count_alive(PLAYER_1)
    op_tot = board.count_alive(PLAYER_2)
    bf_tot = (board.count_on_battlefield(PLAYER_1)
              + board.count_on_battlefield(PLAYER_2))
    if my_tot <= 3 or op_tot <= 3:
        return 'endgame'
    if my_tot >= 7 and op_tot >= 7 and bf_tot <= 4:
        return 'opening'
    if my_tot >= 5 and op_tot >= 5 and bf_tot <= 8:
        return 'development'
    return 'midgame'


class PhaseEvaluator:
    """PSA's evaluator - discrete phase, binary safe/death hiding.

    stacking_mode:
      'quadratic' (default, as designed) - rescue-aware penalty on NEWLY
          created stacks: -8*cnt^2 outside the endgame, -2*cnt inside it,
          waived when the stack is a rescue (source danger > 0.3 and the
          destination is safer).
      'simple' - flat -3*cnt on ALL stacks, no rescue clause and no
          reference to the pre-move board. This is the same substitution
          that turns TLE into TLE-X in tab_look.py.
    """

    def __init__(self, columns: int = DEFAULT_COLUMNS,
                 stacking_mode: str = 'quadratic'):
        self.columns = columns
        self.stacking_mode = stacking_mode
        self.rules = TabRules(columns)

    def evaluate(self, board: Board, player: int,
                 original_board: Optional[Board] = None) -> float:
        C = self.columns
        rules = self.rules
        cells = board.cells

        my_tot = board.count_alive(player)
        op_tot = board.count_alive(-player)

        if op_tot == 0:
            return 100000.0
        if my_tot == 0:
            return -100000.0

        phase = psa_phase(board)
        mat = my_tot - op_tot
        score = mat * 15.0

        # Captures
        if original_board is not None:
            orig_op = original_board.count_alive(-player)
            captures = orig_op - op_tot
            if captures > 0:
                if phase == 'endgame':
                    score += captures * captures * 50.0
                elif phase == 'opening':
                    score += captures * 20.0
                else:
                    score += captures * 30.0

        # Per-soldier terms
        pi = board.player_index(player)
        op_bf = sum(1 for i in range(C, 3 * C) if cells[i] * -player > 0)
        for i in range(rules.board_size):
            v = cells[i]
            if v * player <= 0:
                continue
            cnt = abs(v)
            if is_own_home_pos(i, player, C):
                frozen = board.frozen_at[pi].get(i, 0)
                freed = cnt - frozen
                if frozen > 0:
                    score -= frozen * 2.0
                if freed > 0:
                    score += freed * 3.0
            elif is_enemy_home_pos(i, player, C):
                # Binary safe/death (this is PSA's design - not the smooth one)
                escape = home_hiding_escape_prob(i, player, board, rules)
                if escape > 0.5:  # PSA: binary
                    score += cnt * 8.0
                else:
                    score -= cnt * 15.0
                if my_tot == 1:
                    score += 25.0
            else:
                col = i % C
                if 3 <= col <= 4:
                    score += cnt * 2.0
                dng = compute_danger(i, player, board, rules)
                score -= dng * dng * cnt * 10.0
                # Approach to enemy home entry
                my_track = rules.get_track(player)
                try:
                    my_idx = my_track.index(i)
                    entry_dist = rules.track_len - 1 - my_idx
                except ValueError:
                    entry_dist = 99
                if op_bf > 0 and entry_dist <= 3:
                    score += cnt * 4.0

        # Stacking
        if self.stacking_mode == 'simple':
            # flat -3*cnt on every stack (FitnessEvaluator style, as in TLE-X)
            for i in range(rules.board_size):
                v = cells[i]
                if v * player <= 0:
                    continue
                cnt = abs(v)
                if cnt > 1:
                    score -= cnt * 3.0
        elif self.stacking_mode != 'off' and original_board is not None:
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
                        if new_cnt < abs(orig_vj):
                            src_dng = compute_danger(j, player, original_board,
                                                       rules)
                            dst_dng = compute_danger(i, player, board, rules)
                            if src_dng > 0.3 and dst_dng < src_dng:
                                is_rescue = True
                                break
                    if not is_rescue:
                        if phase == 'endgame':
                            score -= cnt * 2.0
                        else:
                            score -= cnt * cnt * 8.0

        # Mobility
        my_frozen = len(board.frozen_queue[pi])
        if my_frozen == 0 and phase != 'endgame':
            score += 10.0

        # Ambush threats (soldiers safely hidden in enemy home)
        opp_home_start = 3 * C if player == PLAYER_1 else 0
        opp_home_end = 4 * C if player == PLAYER_1 else C
        ambush_count = 0
        for i in range(opp_home_start, opp_home_end):
            if cells[i] * player > 0:
                if home_hiding_escape_prob(i, player, board, rules) > 0.5:
                    ambush_count += 1
        score += ambush_count * 12.0

        return score


class PhaseSamplingAgent(TabAgent):
    """
    Python port of PSA. Uses biased random allocation generation
    (no enumeration), discrete phases, binary safe/death hiding.
    """

    def __init__(self, columns: int = DEFAULT_COLUMNS,
                 n_samples: int = 200,
                 stacking_mode: str = 'quadratic'):
        self.columns = columns
        self.n_samples = n_samples
        self.stacking_mode = stacking_mode
        self.rules = TabRules(columns)
        self.evaluator = PhaseEvaluator(columns, stacking_mode=stacking_mode)

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

        # PSA uses biased sampling only - no enumeration
        candidates = biased_sample_allocations(
            throw_values, movable_soldiers, player, board, self.rules,
            n_samples=self.n_samples)

        best_alloc, best_score = None, float('-inf')
        for alloc in candidates:
            sim = simulate_with_smart_forks(board, alloc, movable_soldiers,
                                              player, self.rules,
                                              self.evaluator)
            score = self.evaluator.evaluate(sim, player, original_board=board)
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
