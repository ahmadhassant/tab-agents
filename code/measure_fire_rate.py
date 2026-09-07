"""
measure_fire_rate.py - the exemption fire rate (R1.5).

Section IX recommends that any heuristic with an exemption clause measure its
empirical fire rate, and that "an exemption that fires below 10% of trigger
events is effectively an unconditional penalty in disguise". That currently
rests on an anecdote. This measures it.

The rescue exemption in LookEvaluator's quadratic stacking branch fires when a
stack-creating allocation ALSO moves a soldier out of danger:
    exists j != i with new_count[j] < orig_count[j]
    and src_danger(j) > 0.3 and dst_danger(i) < src_danger(j)

We replay real decisions from TLE-vs-GA-Original games, enumerate the candidate
allocations at each of TLE's decisions, and for every candidate that CREATES a
stack record whether the exemption fired.

Usage:  python measure_fire_rate.py --games 400 --workers 15
"""
import argparse
import random
from collections import Counter

from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_look import get_candidate_allocations, compute_danger, LookAgent
from tab_mcts import apply_allocation, resolve_freeing


def exemption_fires(orig_board, new_board, player, rules):
    """Replicates the rescue test in LookEvaluator's quadratic branch."""
    cn, co = new_board.cells, orig_board.cells
    created = fired = False
    for i in range(rules.board_size):
        v = cn[i]
        if v * player <= 0:
            continue
        cnt = abs(v)
        if cnt <= 1:
            continue
        ov = co[i]
        ocnt = abs(ov) if ov * player > 0 else 0
        if cnt <= ocnt:
            continue
        created = True
        for j in range(rules.board_size):
            ovj = co[j]
            if ovj * player <= 0 or j == i:
                continue
            nv = cn[j]
            ncnt = abs(nv) if nv * player > 0 else 0
            if ncnt < abs(ovj):
                src = compute_danger(j, player, orig_board, rules)
                dst = compute_danger(i, player, new_board, rules)
                if src > 0.3 and dst < src:
                    fired = True
                    break
        if fired:
            break
    return created, fired


def apply_ta(board, alloc, mov, player, rules):
    lists = [[] for _ in mov]
    idx = {p: i for i, p in enumerate(mov)}
    for sm in alloc.soldier_moves:
        if sm.from_pos in idx:
            lists[idx[sm.from_pos]] = list(sm.values_used)
    apply_allocation(board, lists, mov, player, rules)


def one_game(seed, columns=8, max_turns=300):
    random.seed(seed)
    from tab_ai import GAAgent
    board, rules = Board(columns), TabRules(columns)
    tle = LookAgent(lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
                      use_rules=True, binary_hide=True,
                      use_continuous_phase=False, stacking_mode='quadratic')
    opp = GAAgent('beginner', fitness_type='original')
    c = Counter()
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        rem = resolve_freeing(board, player, seq, rules)
        mov = rules.get_movable_soldiers(board, player, rem)
        if mov and rem:
            if player == PLAYER_1:
                cands = get_candidate_allocations(rem, mov, player, board, rules,
                                                    enum_threshold=2000, sample_n=80)
                c['decisions'] += 1
                for a in cands:
                    sim = board.clone()
                    apply_allocation(sim, a, mov, player, rules)
                    created, fired = exemption_fires(board, sim, player, rules)
                    c['candidates'] += 1
                    if created:
                        c['stack_creating'] += 1
                        if fired:
                            c['exempted'] += 1
                apply_ta(board, tle.choose_allocation(board, player, rem, mov, rules),
                           mov, player, rules)
            else:
                apply_ta(board, opp.choose_allocation(board, player, rem, mov, rules),
                           mov, player, rules)
        if board.is_game_over() is not None:
            break
        turn += 1
    return c


def _w(seed):
    try:
        return one_game(seed)
    except Exception:
        return Counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--workers', type=int, default=15)
    args = ap.parse_args()
    import multiprocessing as mp
    total = Counter()
    with mp.Pool(args.workers) as pool:
        for c in pool.imap_unordered(_w, range(9000, 9000 + args.games), chunksize=4):
            total.update(c)
    dec, cand = total['decisions'], total['candidates']
    sc, ex = total['stack_creating'], total['exempted']
    print()
    print('=' * 66)
    print('  RESCUE-EXEMPTION FIRE RATE  (R1.5)')
    print('=' * 66)
    print(f'  decisions replayed        : {dec:,}')
    print(f'  candidate allocations     : {cand:,}')
    print(f'  stack-CREATING candidates : {sc:,}  ({100*sc/max(cand,1):.1f}% of candidates)')
    print(f'  exemption FIRED on        : {ex:,}')
    print()
    rate = 100 * ex / max(sc, 1)
    print(f'  >>> FIRE RATE = {rate:.2f}% of stack-creating candidates <<<')
    print()
    if rate < 10:
        print('  Below the 10% threshold proposed in Section IX: the exemption is')
        print('  effectively an unconditional penalty in disguise - exactly the')
        print('  failure mode the autopsy diagnosed.')
    else:
        print('  At or above 10%: the Section IX rule of thumb must be restated,')
        print('  since the exemption fired reasonably often and the collapse needs')
        print('  another explanation.')


if __name__ == '__main__':
    main()
