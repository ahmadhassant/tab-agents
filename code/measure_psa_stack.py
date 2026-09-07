"""
measure_psa_stack.py - is PSA's rescue-aware quadratic stacking penalty doing
anything at all?

The mech_probe3 finding was that TLE's penalty never fires on TLE's own candidate
moves, because its lookahead hands the evaluator the post-move board. PSA passes
the true pre-move board, so the precondition CAN be satisfied. This measures, at
the call site:

  decisiveness - share of PSA's decisions where deleting the term (PSA-N), or
                 replacing it with the flat -3c (PSA-X), changes the move the
                 agent actually plays.

Usage:  python measure_psa_stack.py --games 60
"""
import argparse, random
from collections import Counter
from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_psa import PhaseSamplingAgent
from tab_mcts import resolve_freeing
from measure_forks import apply_and_count


def key(alloc):
    return sorted((m.from_pos, tuple(m.values_used)) for m in alloc.soldier_moves)


def one_game(seed, columns=8, max_turns=300):
    random.seed(seed)
    board, rules = Board(columns), TabRules(columns)
    agents = {
        'PSA':   PhaseSamplingAgent(columns=columns, n_samples=80, stacking_mode='quadratic'),
        'PSA-X': PhaseSamplingAgent(columns=columns, n_samples=80, stacking_mode='simple'),
        'PSA-N': PhaseSamplingAgent(columns=columns, n_samples=80, stacking_mode='off'),
    }
    base = agents['PSA']
    c = Counter()
    for turn in range(max_turns):
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        rem = resolve_freeing(board, player, seq, rules)
        mov = rules.get_movable_soldiers(board, player, rem)
        if mov and rem:
            a = base.choose_allocation(board, player, rem, mov, rules)
            if len(mov) > 1:
                c['decisions'] += 1
                kq = key(a)
                if key(agents['PSA-X'].choose_allocation(board, player, rem, mov, rules)) != kq:
                    c['differs_from_X'] += 1
                if key(agents['PSA-N'].choose_allocation(board, player, rem, mov, rules)) != kq:
                    c['differs_from_N'] += 1
            apply_and_count(board, a, mov, player, rules, Counter())
        if board.is_game_over() is not None:
            break
    c['games'] = 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=60)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--start-seed', type=int, default=900)
    a = ap.parse_args()
    import multiprocessing as mp
    t = Counter()
    with mp.Pool(a.workers) as pool:
        for c in pool.imap_unordered(one_game, range(a.start_seed, a.start_seed + a.games), chunksize=2):
            t.update(c)
    d = t['decisions']
    print()
    print('=' * 66)
    print('  IS PSA\'S QUADRATIC STACKING TERM DECISIVE?')
    print('=' * 66)
    print('  games                                  : %d' % t['games'])
    print('  multi-soldier decisions                : %d' % d)
    print('  PSA vs PSA-X (flat -3c)   different move: %d  (%.2f%%)'
          % (t['differs_from_X'], 100 * t['differs_from_X'] / max(d, 1)))
    print('  PSA vs PSA-N (term deleted) different   : %d  (%.2f%%)'
          % (t['differs_from_N'], 100 * t['differs_from_N'] / max(d, 1)))
    print()


if __name__ == '__main__':
    main()
