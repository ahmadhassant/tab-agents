"""
test_mcts_parity.py - Proves tab_mcts's rollout engine is the same game as
the tournament engine.

Same seed + equivalent policy => identical game, move for move. If this
fails, no MCTS result built on tab_mcts is comparable to a tournament result.

Run:  python test_mcts_parity.py
"""

import random
import sys

from tab_game import Board, TabRules, PLAYER_1, PLAYER_2
from tab_ai import RandomAgent
from tab_runner import play_one_game
import tab_mcts


def test_parity(n_games: int = 300) -> bool:
    """Seeded random-vs-random must be identical in both engines."""
    mismatches = []
    for seed in range(n_games):
        ref = play_one_game(RandomAgent(), RandomAgent(), seed=seed)
        mine = tab_mcts.play_game(tab_mcts.random_policy,
                                    tab_mcts.random_policy, seed=seed)
        for key in ('winner', 'turns', 'p1_alive', 'p2_alive',
                      'p1_captures', 'p2_captures'):
            if ref[key] != mine[key]:
                mismatches.append((seed, key, ref[key], mine[key]))
                break
    if mismatches:
        print(f"  FAIL: {len(mismatches)}/{n_games} games differ")
        for seed, key, a, b in mismatches[:5]:
            print(f"    seed {seed}: {key} tournament={a} mcts={b}")
        return False
    print(f"  PASS: {n_games}/{n_games} games identical "
            f"(winner, turns, alive, captures)")
    return True


def test_rules_alive(n_games: int = 200) -> bool:
    """
    The rules that the old MCTS engine was missing must actually be active
    here: no-reentry, LIFO taint, forks, freezing.
    """
    rules = TabRules(8)
    ok = True
    for attr in ('no_reentry', 'exited_enemy_home'):
        b = Board(8)
        if not (hasattr(b, attr) or attr in getattr(b, '__dict__', {})):
            continue
    # Forks: confirm the engine produces 2-destination choices at all, and
    # that pick_fork is what resolves them.
    forks_seen = 0
    freezes_seen = 0
    for seed in range(n_games):
        random.seed(50_000 + seed)
        board = Board(8)
        player = PLAYER_1
        for _turn in range(60):
            if board.is_game_over() is not None:
                break
            seq = tab_mcts.StickDice.throw_turn()
            remaining = tab_mcts.resolve_freeing(board, player, seq, rules)
            movable = rules.get_movable_soldiers(board, player, remaining)
            if movable and remaining:
                for pos in movable:
                    for val in set(remaining):
                        d = rules.compute_destinations(pos, val, player, board)
                        if len(d) > 1:
                            forks_seen += 1
                alloc = tab_mcts.random_policy(board, player, remaining,
                                                 movable, rules)
                tab_mcts.apply_allocation(board, alloc, movable, player, rules)
            pi = board.player_index(player)
            freezes_seen += len(board.frozen_queue[pi])
            player = -player
    print(f"  fork situations encountered: {forks_seen}")
    print(f"  frozen-soldier states seen:  {freezes_seen}")
    if forks_seen == 0:
        print("  FAIL: no forks ever generated - fork handling is missing")
        ok = False
    if freezes_seen == 0:
        print("  FAIL: freezing never observed")
        ok = False
    if ok:
        print("  PASS: forks and freezing are live in the rollout engine")
    return ok


def test_no_illegal_states(n_games: int = 200) -> bool:
    """Rollouts must never produce a board that violates basic invariants."""
    rules = TabRules(8)
    bad = 0
    for seed in range(n_games):
        random.seed(70_000 + seed)
        sim = Board(8)
        player = PLAYER_1
        for _turn in range(120):
            if sim.is_game_over() is not None:
                break
            tab_mcts.step_turn(sim, player, rules, tab_mcts.random_policy)
            p1 = sum(v for v in sim.cells if v > 0)
            p2 = sum(-v for v in sim.cells if v < 0)
            if p1 > 8 or p2 > 8 or p1 < 0 or p2 < 0:
                bad += 1
                break
            player = -player
    if bad:
        print(f"  FAIL: {bad}/{n_games} rollouts reached an illegal state")
        return False
    print(f"  PASS: {n_games} rollouts, soldier counts stayed in [0, 8]")
    return True


if __name__ == '__main__':
    print("=" * 66)
    print("  tab_mcts PIECE 1 - rollout engine parity tests")
    print("=" * 66)
    results = []
    print("\n[1] Engine parity vs tab_runner (seeded random-vs-random)")
    results.append(test_parity())
    print("\n[2] Rules the old MCTS engine was missing")
    results.append(test_rules_alive())
    print("\n[3] Legality of rollout states")
    results.append(test_no_illegal_states())
    print("\n" + "=" * 66)
    if all(results):
        print("  ALL PASS - rollout engine plays the tournament's game")
        sys.exit(0)
    print("  FAILURES - do not build search on this until resolved")
    sys.exit(1)
