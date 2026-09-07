"""PART C - the capture bonus (line 294) is gated on the same original_board.

captures = original_board.count_alive(-player) - board.count_alive(-player)

On the lookahead path original_board=board_after_my_move, so MY captures
(which happened in the transition INTO that board) fall outside the compared
interval.  Measure how often captures>0 is actually seen at each call site.
"""
import sys

import tab_ai
import tab_look as TL
from tab_look import LookAgent
from tab_runner import play_one_game

orig_eval = TL.LookEvaluator.evaluate
sites = {}


def patched(self, board, player, original_board=None):
    f = sys._getframe(1)
    key = (f.f_code.co_name, f.f_lineno)
    d = sites.setdefault(key, {'n': 0, 'cap': 0, 'neg': 0, 'nob': 0})
    d['n'] += 1
    if original_board is None:
        d['nob'] += 1
    else:
        c = original_board.count_alive(-player) - board.count_alive(-player)
        if c > 0:
            d['cap'] += 1
        elif c < 0:
            d['neg'] += 1
    return orig_eval(self, board, player, original_board)


TL.LookEvaluator.evaluate = patched
try:
    for g in range(2):
        a = LookAgent(columns=8, stacking_mode='quadratic')
        b = tab_ai.GAAgent('beginner', fitness_type='original')
        if g % 2 == 0:
            play_one_game(a, b, seed=7000 + g)
        else:
            play_one_game(b, a, seed=7000 + g)
finally:
    TL.LookEvaluator.evaluate = orig_eval

print('PART C - is the capture bonus ever triggered, per call site?')
print()
print('  %-34s %8s %10s %10s %8s'
      % ('call site', 'calls', 'captures>0', 'captures<0', 'ob=None'))
for (name, line), d in sorted(sites.items(), key=lambda kv: -kv[1]['n']):
    print('  %-34s %8d %9.1f%% %9.1f%% %8d'
          % ('%s():%d' % (name, line), d['n'],
             100.0 * d['cap'] / d['n'], 100.0 * d['neg'] / d['n'], d['nob']))
print()
print('  captures<0 means the baseline had FEWER enemies than the scored')
print('  board - i.e. the subtraction is running over the wrong interval.')
