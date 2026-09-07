"""Why does 'quadratic' stack as much as 'off'?

Hypothesis: in lookahead_score the evaluator is called with
original_board=board_after_my_move.  The quadratic/linear branch only fires
when cnt > orig_cnt.  My own soldiers do not move during the opponent's turn,
so relative to board_after_my_move my counts can only stay equal or fall.
=> the penalty can never fire against MY OWN stacking on the lookahead path.
It only fires at the site that scores the OPPONENT's move (original_board=opp_board).

PART A attributes the actually-applied penalty to exact source lines.
PART B tests the behavioural prediction that follows.
"""
import copy
import sys

import tab_ai
import tab_look as TL
from tab_look import LookAgent
from tab_runner import play_one_game


# ---------------------------------------------------------------- PART A
def part_a(n_games=2, seed0=7000):
    orig_eval = TL.LookEvaluator.evaluate
    sites = {}

    def patched(self, board, player, original_board=None):
        val = orig_eval(self, board, player, original_board)
        twin = getattr(self, '_twin', None)
        if twin is None:
            twin = copy.copy(self)
            twin.stacking_mode = 'off'
            self._twin = twin
        # identical in every term except the stacking penalty
        val_off = orig_eval(twin, board, player, original_board)
        pen = val_off - val          # > 0 means a stacking penalty was applied
        f = sys._getframe(1)
        key = (f.f_code.co_name, f.f_lineno)
        d = sites.setdefault(key, {'n': 0, 'fired': 0, 'tot': 0.0})
        d['n'] += 1
        if abs(pen) > 1e-9:
            d['fired'] += 1
            d['tot'] += pen
        return val

    TL.LookEvaluator.evaluate = patched
    try:
        for g in range(n_games):
            a = LookAgent(columns=8, stacking_mode='quadratic')
            b = tab_ai.GAAgent('beginner', fitness_type='original')
            if g % 2 == 0:
                play_one_game(a, b, seed=seed0 + g)
            else:
                play_one_game(b, a, seed=seed0 + g)
    finally:
        TL.LookEvaluator.evaluate = orig_eval

    print('PART A - where does the quadratic penalty actually fire?')
    print('  (stacking_mode="quadratic", use_lookahead=True)')
    print()
    print('  %-34s %10s %10s %10s' % ('call site', 'calls', 'fired', 'mean pen'))
    for (name, line), d in sorted(sites.items(), key=lambda kv: -kv[1]['n']):
        mean = d['tot'] / d['fired'] if d['fired'] else 0.0
        print('  %-34s %10d %9.1f%% %10.2f'
              % ('%s():%d' % (name, line), d['n'],
                 100.0 * d['fired'] / d['n'], mean))
    print()
    print('  line 703/741 = my position after opponent reply, baseline'
          ' board_after_my_move')
    print('  line 721     = opponent scoring its own move, baseline opp_board')
    print('  line 908     = greedy path, baseline board (pre-move)')
    print()


# ---------------------------------------------------------------- PART B
def creates_stack(before, after, player):
    for i in range(len(before)):
        was = abs(before[i]) if before[i] * player > 0 else 0
        now = abs(after[i]) if after[i] * player > 0 else 0
        if now > 1 and now > was:
            return True
    return False


def probe(label, n_games=8, seed0=7000, **kw):
    st = {'n': 0, 'create': 0}
    orig = LookAgent.choose_allocation

    def wrapped(self, board, player, throw_values, movable_soldiers, rules):
        before = list(board.cells)
        res = orig(self, board, player, throw_values, movable_soldiers, rules)
        st['n'] += 1
        if res is not None:
            lists = [[] for _ in movable_soldiers]
            for sm in getattr(res, 'soldier_moves', []):
                try:
                    idx = movable_soldiers.index(sm.from_pos)
                except ValueError:
                    continue
                lists[idx].extend(sm.values_used)
            sim = rules.simulate_allocation(board, lists, movable_soldiers,
                                            player)
            if creates_stack(before, list(sim.cells), player):
                st['create'] += 1
        return res

    LookAgent.choose_allocation = wrapped
    try:
        for g in range(n_games):
            a = LookAgent(columns=8, **kw)
            b = tab_ai.GAAgent('beginner', fitness_type='original')
            if g % 2 == 0:
                play_one_game(a, b, seed=seed0 + g)
            else:
                play_one_game(b, a, seed=seed0 + g)
    finally:
        LookAgent.choose_allocation = orig
    n = st['n'] or 1
    print('  %-40s creates %5.1f%%   (%d decisions)'
          % (label, 100.0 * st['create'] / n, st['n']))


if __name__ == '__main__':
    part_a()
    print('PART B - behavioural prediction')
    print('  if the penalty is inert on the lookahead path, then turning'
          ' lookahead OFF')
    print('  (which uses the correct pre-move baseline) must cut stacking'
          ' sharply:')
    print()
    probe('quadratic + lookahead  (TLA / TLE)',
          stacking_mode='quadratic', use_lookahead=True)
    probe('quadratic + greedy     (TLA-G, correct baseline)',
          stacking_mode='quadratic', use_lookahead=False)
    probe('simple/flat + lookahead (TLA-S)',
          stacking_mode='simple', use_lookahead=True)
    probe('off + lookahead',
          stacking_mode='off', use_lookahead=True)
