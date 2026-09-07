"""
measure_forks_psa.py - fork simulate/execute agreement for PSA specifically.

measure_forks.py measures the TLA family, whose agents run with
smart_fork_picker_self=False and therefore cannot diverge. PSA is different:
PhaseSamplingAgent.choose_allocation scores its candidates on
simulate_with_smart_forks(..., self.evaluator), which resolves every fork with
PSA's own evaluator, while the runner executes the allocation with the fixed
engine-level picker. This script measures how often those two choices differ.

Usage:  python measure_forks_psa.py --games 400 --workers 12
"""
import argparse
import random
from collections import Counter

from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_mcts import resolve_freeing
from measure_forks import apply_and_count


def one_game(seed, columns=8, max_turns=300):
    random.seed(seed)
    from tab_ai import GAAgent
    from tab_psa import PhaseSamplingAgent
    board, rules = Board(columns), TabRules(columns)
    a1 = PhaseSamplingAgent(columns=columns, n_samples=80)   # tournament config
    a2 = GAAgent('beginner', fitness_type='original')
    c = Counter()
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        rem = resolve_freeing(board, player, seq, rules)
        mov = rules.get_movable_soldiers(board, player, rem)
        if mov and rem:
            agent = a1 if player == PLAYER_1 else a2
            alloc = agent.choose_allocation(board, player, rem, mov, rules)
            # only attribute fork comparisons to PSA's own moves
            ev = a1.evaluator if player == PLAYER_1 else None
            apply_and_count(board, alloc, mov, player, rules, c, evaluator=ev)
        if board.is_game_over() is not None:
            break
        turn += 1
    c['games'] = 1
    c['turns'] = turn + 1
    return c


def _w(seed):
    try:
        return one_game(seed)
    except Exception:
        return Counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--start-seed', type=int, default=5000)
    args = ap.parse_args()
    import multiprocessing as mp
    t = Counter()
    with mp.Pool(args.workers) as pool:
        for c in pool.imap_unordered(_w, range(args.start_seed, args.start_seed + args.games), chunksize=4):
            t.update(c)
    g, mv, fk = t['games'], t['moves'], t['fork_moves']
    d = t['engine_vs_evaluator_differ']
    print()
    print('=' * 68)
    print('  PSA: INTERNAL FORK CHOICE vs EXECUTED FORK CHOICE')
    print('=' * 68)
    print('  games                          : %d' % g)
    print('  soldier moves counted          : %d' % mv)
    print('  moves offering a fork          : %d  (%.2f%% of moves)'
          % (fk, 100 * fk / max(mv, 1)))
    print('  PSA internal choice != executed: %d  (%.2f%% of fork moves)'
          % (d, 100 * d / max(fk, 1)))
    print('  as a share of all moves        : %.3f%%' % (100 * d / max(mv, 1)))
    print('RAW games=%d moves=%d forks=%d differ=%d' % (g, mv, fk, d))
    print()


if __name__ == '__main__':
    main()
