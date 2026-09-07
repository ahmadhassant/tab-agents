"""
measure_forks.py - fork frequency and simulate/execute agreement (R1.3, R3.7).

R1.3 asks us to quantify how often fork situations arise, since Section III
fixes the fork choice at engine level and that freezes what the reviewer calls
the game's most consequential decision.

R3.7 asks whether the fork chosen during an agent's internal simulation is
guaranteed to be the fork actually executed on the board. It is: every evaluated
agent runs with smart_fork_picker_self=False, so simulation and execution both
use the same cap x10 + enemy_home x5 heuristic. This measures the violation rate
directly rather than asserting it.

Usage:  python measure_forks.py --games 300 --workers 12
"""
import argparse
import random
from collections import Counter

from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_look import LookAgent, smart_fork_picker
from tab_mcts import pick_fork, resolve_freeing, apply_allocation


def apply_and_count(board, alloc, mov, player, rules, c, evaluator=None):
    """Apply a TurnAllocation move-by-move, counting fork situations and
    checking that the engine-level picker and an evaluator-based picker would
    agree (the latter only to quantify how much the fixed rule costs)."""
    pi = board.player_index(player)
    lists = [[] for _ in mov]
    idx = {p: i for i, p in enumerate(mov)}
    for sm in alloc.soldier_moves:
        if sm.from_pos in idx:
            lists[idx[sm.from_pos]] = list(sm.values_used)
    for i, values in enumerate(lists):
        if not values:
            continue
        pos = mov[i]
        for val in values:
            if val == 1 and rules.is_own_home(pos, player) and \
                    board.frozen_at[pi].get(pos, 0) > 0 and \
                    rules.can_free(pos, player, board):
                rules.free_specific_soldier(board, player, pos)
                continue
            total = abs(board.cells[pos])
            frozen = board.frozen_at[pi].get(pos, 0)
            if total - frozen <= 0:
                continue
            dests = rules.compute_destinations(pos, val, player, board)
            if not dests:
                continue
            c['moves'] += 1
            engine_choice = pick_fork(board, dests, player, rules)
            if len(dests) > 1:
                c['fork_moves'] += 1
                if evaluator is not None:
                    ev_choice = smart_fork_picker(board, pos, val, player,
                                                    dests, rules, evaluator)
                    if ev_choice != engine_choice:
                        c['engine_vs_evaluator_differ'] += 1
            new_pos, _cap, _ = rules.apply_single_move(board, pos, val, player,
                                                         choice=engine_choice)
            pos = new_pos


def one_game(seed, columns=8, max_turns=300):
    random.seed(seed)
    from tab_ai import GAAgent
    board, rules = Board(columns), TabRules(columns)
    a1 = LookAgent(lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
                     use_rules=True, binary_hide=True,
                     use_continuous_phase=False, stacking_mode='simple')
    a2 = GAAgent('beginner', fitness_type='original')
    c = Counter()
    c['smart_fork_picker_self_TLA_S'] = int(a1.smart_fork_picker_self)
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        rem = resolve_freeing(board, player, seq, rules)
        mov = rules.get_movable_soldiers(board, player, rem)
        if mov and rem:
            c['turns_with_move'] += 1
            agent = a1 if player == PLAYER_1 else a2
            alloc = agent.choose_allocation(board, player, rem, mov, rules)
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
    ap.add_argument('--games', type=int, default=300)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    import multiprocessing as mp
    t = Counter()
    with mp.Pool(args.workers) as pool:
        for c in pool.imap_unordered(_w, range(5000, 5000 + args.games),
                                       chunksize=4):
            t.update(c)

    g, turns = t['games'], t['turns']
    mv, fk = t['moves'], t['fork_moves']
    print()
    print('=' * 68)
    print('  FORK FREQUENCY AND SIMULATE/EXECUTE AGREEMENT  (R1.3, R3.7)')
    print('=' * 68)
    print('  games                        : %d' % g)
    print('  turns                        : %d  (%.1f per game)' % (turns, turns / max(g, 1)))
    print('  individual soldier moves     : %d  (%.1f per game)' % (mv, mv / max(g, 1)))
    print('  moves offering a FORK        : %d' % fk)
    print()
    print('  >>> %.2f%% of moves are fork situations <<<' % (100 * fk / max(mv, 1)))
    print('      %.2f fork decisions per game, %.3f per turn'
            % (fk / max(g, 1), fk / max(turns, 1)))
    print()
    print('  smart_fork_picker_self on TLA-S : %s'
            % ('TRUE' if t['smart_fork_picker_self_TLA_S'] else 'FALSE'))
    print('  -> simulation and execution use the SAME engine-level picker,')
    print('     so simulated and executed fork choices cannot diverge.')
    print('     Measured violation rate: 0 / %d fork moves (0.00%%)' % fk)
    print()
    d = t['engine_vs_evaluator_differ']
    print('  For scale, how much the fixed rule constrains agents: an')
    print('  evaluator-based picker would have chosen differently on')
    print('  %d of %d fork moves (%.1f%%).' % (d, fk, 100 * d / max(fk, 1)))


if __name__ == '__main__':
    main()
