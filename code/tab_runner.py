"""
tab_runner.py - Headless game runner for tournament use.

Plays one complete game between two agents. Used by tournament_grid.py
for parallel evaluation on multi-core machines.

Fork-picking heuristic is fixed and uniform across all agents:
  cap x 10  +  enemy_home x 5
This keeps the comparison fair - any agent that wants smarter fork picking
should do it inside its own choose_allocation().

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
(AI assistance disclosed in the manuscript acknowledgments)
"""

import time
import random
from tab_game import Board, TabRules, StickDice, PLAYER_1, PLAYER_2


def play_one_game(agent1, agent2, max_turns: int = 300,
                    columns: int = 8, seed: int | None = None) -> dict:
    """
    Play one game between agent1 (P1) and agent2 (P2).

    Returns:
        dict with keys: winner (1, -1, or 0), turns, time_sec,
        p1_alive, p2_alive (final counts), p1_captures, p2_captures.
    """
    if seed is not None:
        random.seed(seed)

    board = Board(columns)
    rules = TabRules(columns)
    t0 = time.time()

    # Track captures (initial soldier counts minus final)
    initial_alive = {PLAYER_1: 8, PLAYER_2: 8}

    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else PLAYER_2
        agent = agent1 if player == PLAYER_1 else agent2

        # Throw dice
        seq = StickDice.throw_turn()

        # Process freeing for 1s in queue order
        remaining = list(seq)
        pi = board.player_index(player)
        ones_used = 0
        while 1 in remaining and ones_used < seq.count(1):
            if not board.frozen_queue[pi]:
                break
            next_pos = board.frozen_queue[pi][0]
            if rules.can_free(next_pos, player, board):
                rules.free_specific_soldier(board, player, next_pos)
                remaining.remove(1)
                ones_used += 1
            else:
                break

        # Get movable soldiers
        movable = rules.get_movable_soldiers(board, player, remaining)
        if not movable or not remaining:
            turn += 1
            winner = board.is_game_over()
            if winner is not None:
                break
            continue

        # Agent chooses allocation
        alloc = agent.choose_allocation(board, player, remaining, movable, rules)

        # Apply allocation
        for sm in alloc.soldier_moves:
            pos = sm.from_pos
            for val in sm.values_used:
                # Freeing check
                if val == 1 and rules.is_own_home(pos, player) and \
                        board.frozen_at[pi].get(pos, 0) > 0 and \
                        rules.can_free(pos, player, board):
                    rules.free_specific_soldier(board, player, pos)
                    continue
                total = abs(board.cells[pos])
                frozen = board.frozen_at[pi].get(pos, 0)
                m = total - frozen
                if m <= 0:
                    continue
                dests = rules.compute_destinations(pos, val, player, board)
                if not dests:
                    continue
                # Uniform fork-picking: cap x 10 + enemy_home x 5
                choice = 0
                if len(dests) > 1:
                    best_score = float('-inf')
                    for ci, d in enumerate(dests):
                        s = 0.0
                        if board.cells[d] * player < 0:
                            s += abs(board.cells[d]) * 10.0
                        if rules.is_enemy_home(d, player):
                            s += 5.0
                        if s > best_score:
                            best_score = s
                            choice = ci
                new_pos, cap, _ = rules.apply_single_move(
                    board, pos, val, player, choice=choice)
                pos = new_pos

        winner = board.is_game_over()
        if winner is not None:
            break
        turn += 1

    elapsed = time.time() - t0
    final_winner = board.is_game_over()
    if final_winner is None:
        # Game hit max_turns - declare winner by remaining soldier count
        p1 = board.count_alive(PLAYER_1)
        p2 = board.count_alive(PLAYER_2)
        if p1 > p2:
            final_winner = PLAYER_1
        elif p2 > p1:
            final_winner = PLAYER_2
        else:
            final_winner = 0  # draw

    p1_alive = board.count_alive(PLAYER_1)
    p2_alive = board.count_alive(PLAYER_2)
    return {
        'winner': final_winner,
        'turns': turn + 1,
        'time_sec': elapsed,
        'p1_alive': p1_alive,
        'p2_alive': p2_alive,
        'p1_captures': initial_alive[PLAYER_2] - p2_alive,
        'p2_captures': initial_alive[PLAYER_1] - p1_alive,
    }


if __name__ == '__main__':
    # Smoke test
    from tab_ai import GAAgent
    from tab_look import LookAgent
    look = LookAgent(lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
                       use_rules=False)
    ga = GAAgent('beginner', fitness_type='expert')
    r = play_one_game(look, ga, seed=42)
    print(f"Test game: winner={r['winner']}, turns={r['turns']}, "
          f"time={r['time_sec']:.2f}s, captures P1={r['p1_captures']} "
          f"P2={r['p2_captures']}")
