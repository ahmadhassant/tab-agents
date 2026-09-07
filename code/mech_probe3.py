"""Publication-grade run of the stacking-mechanism probe.

Part 1: per-call-site census of where the quadratic penalty actually fires.
Part 2: stack-creation rate per configuration, with 95% CIs.
"""
import copy
import math
import sys

import tab_ai
import tab_look as TL
from tab_look import LookAgent
from tab_runner import play_one_game

NG_CENSUS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NG_BEHAV = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def ci(p, n):
    return 1.96 * math.sqrt(p * (1.0 - p) / n) * 100.0 if n else 0.0


def census(n_games):
    orig_eval = TL.LookEvaluator.evaluate
    sites = {}

    def patched(self, board, player, original_board=None):
        val = orig_eval(self, board, player, original_board)
        twin = getattr(self, '_twin', None)
        if twin is None:
            twin = copy.copy(self)
            twin.stacking_mode = 'off'
            self._twin = twin
        pen = orig_eval(twin, board, player, original_board) - val
        cap = 0
        if original_board is not None:
            cap = original_board.count_alive(-player) - board.count_alive(-player)
        f = sys._getframe(1)
        d = sites.setdefault((f.f_code.co_name, f.f_lineno),
                             {'n': 0, 'fired': 0, 'tot': 0.0, 'cap': 0})
        d['n'] += 1
        if abs(pen) > 1e-9:
            d['fired'] += 1
            d['tot'] += pen
        if cap > 0:
            d['cap'] += 1
        return val

    TL.LookEvaluator.evaluate = patched
    try:
        for g in range(n_games):
            a = LookAgent(columns=8, stacking_mode='quadratic')
            b = tab_ai.GAAgent('beginner', fitness_type='original')
            if g % 2 == 0:
                play_one_game(a, b, seed=7000 + g)
            else:
                play_one_game(b, a, seed=7000 + g)
    finally:
        TL.LookEvaluator.evaluate = orig_eval

    print('PART 1 - where the quadratic stacking penalty fires (%d games)'
          % n_games)
    print('  %-26s %9s %12s %10s %12s'
          % ('call site', 'evals', 'penalty', 'mean pen', 'capture bonus'))
    for (nm, ln), d in sorted(sites.items(), key=lambda kv: -kv[1]['n']):
        mean = d['tot'] / d['fired'] if d['fired'] else 0.0
        print('  %-26s %9d %11.2f%% %10.2f %11.2f%%'
              % ('%s():%d' % (nm, ln), d['n'],
                 100.0 * d['fired'] / d['n'], -mean,
                 100.0 * d['cap'] / d['n']))
    print()


def creates_stack(before, after, player):
    for i in range(len(before)):
        was = abs(before[i]) if before[i] * player > 0 else 0
        now = abs(after[i]) if after[i] * player > 0 else 0
        if now > 1 and now > was:
            return True
    return False


def probe(label, n_games, **kw):
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
                play_one_game(a, b, seed=7000 + g)
            else:
                play_one_game(b, a, seed=7000 + g)
    finally:
        LookAgent.choose_allocation = orig
    n = st['n'] or 1
    p = st['create'] / float(n)
    print('  %-44s %6.2f%% +/- %4.2f   (%d decisions)'
          % (label, 100.0 * p, ci(p, n), st['n']))
    sys.stdout.flush()


if __name__ == '__main__':
    census(NG_CENSUS)
    print('PART 2 - stack-creation rate per configuration (%d games each)'
          % NG_BEHAV)
    probe('TLE        quadratic + lookahead', NG_BEHAV,
          stacking_mode='quadratic', use_lookahead=True)
    probe('control    quadratic + greedy (correct baseline)', NG_BEHAV,
          stacking_mode='quadratic', use_lookahead=False)
    probe('TLA-S      flat -3c + lookahead', NG_BEHAV,
          stacking_mode='simple', use_lookahead=True)
    probe('TLE-N      no stacking term + lookahead', NG_BEHAV,
          stacking_mode='off', use_lookahead=True)
    print()
    print('DONE')
