"""
instrument.py - measure per-decision compute for every agent family.

Answers R1.1 and R3.11, which ask for per-move evaluator calls (or wall-clock)
per agent so the lookahead result can be separated from raw compute budget.

Design note: this wraps agents from the OUTSIDE and patches the engine at the
class level. No agent source is modified, so every tournament result remains
bit-for-bit reproducible with or without instrumentation.

Three units are recorded per decision:

  moves   primitive move applications (TabRules.apply_single_move). This is the
          family-neutral currency: GA, TLA and MCTS all ultimately simulate by
          applying moves, so it compares across designs in a way that
          "evaluator calls" cannot (MCTS has no evaluator at all).
  evals   board-evaluation calls, where the agent has an evaluator.
  ms      wall-clock milliseconds.

Usage:
    import instrument
    instrument.patch_engine()
    agent = instrument.wrap(LookAgent(...), 'TLA')
    ... play games ...
    print(agent.report())
"""

import time
from collections import defaultdict

import tab_game

_counters = defaultdict(int)


def reset():
    _counters.clear()


def patch_engine():
    """Count primitive move applications globally (idempotent)."""
    if getattr(tab_game.TabRules, '_instrumented', False):
        return
    orig = tab_game.TabRules.apply_single_move

    def wrapper(self, *args, **kwargs):
        _counters['moves'] += 1
        return orig(self, *args, **kwargs)

    wrapper._orig = orig
    tab_game.TabRules.apply_single_move = wrapper
    tab_game.TabRules._instrumented = True


def _patch_eval(obj):
    """Count evaluate() calls on one evaluator instance."""
    if obj is None or getattr(obj, '_instrumented', False):
        return
    for meth in ('evaluate', 'evaluate_advanced'):
        orig = getattr(obj, meth, None)
        if orig is None:
            continue

        def make(o):
            def w(*a, **k):
                _counters['evals'] += 1
                return o(*a, **k)
            return w
        setattr(obj, meth, make(orig))
    obj._instrumented = True


def _find_evaluators(agent):
    found = []
    for attr in ('evaluator', 'fitness_eval'):
        if hasattr(agent, attr):
            found.append(getattr(agent, attr))
    ga = getattr(agent, 'ga', None)
    if ga is not None and hasattr(ga, 'fitness_eval'):
        found.append(ga.fitness_eval)
    return [f for f in found if f is not None]


class Instrumented:
    """Wraps an agent, recording per-decision compute. Transparent to the
    engine: it only needs choose_allocation()."""

    def __init__(self, agent, name):
        self.agent = agent
        self.name = name
        self.ms = []
        self.moves = []
        self.evals = []
        for ev in _find_evaluators(agent):
            _patch_eval(ev)

    def choose_allocation(self, board, player, throw_values,
                            movable_soldiers, rules):
        m0, e0 = _counters['moves'], _counters['evals']
        t0 = time.perf_counter()
        out = self.agent.choose_allocation(board, player, throw_values,
                                             movable_soldiers, rules)
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        self.moves.append(_counters['moves'] - m0)
        self.evals.append(_counters['evals'] - e0)
        return out

    def summary(self):
        def stats(xs):
            if not xs:
                return (0.0, 0.0, 0.0)
            s = sorted(xs)
            mean = sum(s) / len(s)
            median = s[len(s) // 2]
            p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
            return (mean, median, p95)
        return {'agent': self.name, 'decisions': len(self.ms),
                'ms': stats(self.ms), 'moves': stats(self.moves),
                'evals': stats(self.evals)}


def wrap(agent, name):
    return Instrumented(agent, name)


HEADER = (f"{'agent':<24}{'dec':>6}{'moves/dec':>12}{'p95':>9}"
            f"{'evals/dec':>12}{'ms/dec':>10}{'p95 ms':>9}")


def format_rows(summaries):
    lines = [HEADER, '-' * len(HEADER)]
    for s in sorted(summaries, key=lambda x: -x['moves'][0]):
        lines.append(
            f"{s['agent']:<24}{s['decisions']:>6}"
            f"{s['moves'][0]:>12.0f}{s['moves'][2]:>9.0f}"
            f"{s['evals'][0]:>12.0f}"
            f"{s['ms'][0]:>10.1f}{s['ms'][2]:>9.1f}")
    return '\n'.join(lines)
