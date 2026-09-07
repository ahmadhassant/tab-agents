"""
analyze_budget.py - E2: win rate against MEASURED per-decision compute.

Answers R1.1 / R3.11: is the lookahead lift a property of the search design or
of the compute budget it happens to consume?
"""
import csv
import math
import re
import sys
from collections import defaultdict


def parse_compute(path):
    """Read measured moves/dec from the budget_compute log."""
    out = {}
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'\s*(BM-[A-Za-z0-9-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)', line)
        if m:
            out[m.group(1)] = {'dec': int(m.group(2)), 'moves': int(m.group(3)),
                                 'evals': int(m.group(5)), 'ms': float(m.group(6))}
    return out


def load_games(path):
    return list(csv.DictReader(open(path)))


def h2h(rows, A, B):
    sel = [r for r in rows if {r['agent_p1'], r['agent_p2']} == {A, B}]
    if not sel:
        return None
    w = sum(1 for r in sel
              if (r['agent_p1'] == A and r['winner'] == '1') or
                 (r['agent_p2'] == A and r['winner'] in ('-1', '2')))
    n = len(sel)
    p = w / n
    return p, 1.96 * math.sqrt(p * (1 - p) / n), n


def overall(rows, agents):
    w = defaultdict(int); n = defaultdict(int)
    for r in rows:
        a, b = r['agent_p1'], r['agent_p2']
        n[a] += 1; n[b] += 1
        if r['winner'] == '1': w[a] += 1
        elif r['winner'] in ('-1', '2'): w[b] += 1
    return {a: (w[a] / n[a], n[a]) for a in agents if n[a]}


def main():
    comp = parse_compute(sys.argv[1] if len(sys.argv) > 1
                           else '../03_Results/budget_compute.log')
    rows = load_games(sys.argv[2] if len(sys.argv) > 2
                        else '../03_Results/budget_sweep/games.csv')
    agents = sorted(set(r['agent_p1'] for r in rows) |
                      set(r['agent_p2'] for r in rows))
    ov = overall(rows, agents)

    print('E2  WIN RATE vs MEASURED COMPUTE')
    print(f"{'agent':<14}{'moves/dec':>11}{'evals/dec':>11}{'ms/dec':>9}"
            f"{'overall win%':>14}{'n':>8}")
    print('-' * 67)
    fam = lambda a: ('TLA (lookahead)' if a.startswith('BM-TLA-') else
                       'TLA-G (greedy)' if a.startswith('BM-TLAG') else 'GA')
    for a in sorted(agents, key=lambda x: comp.get(x, {}).get('moves', 0)):
        c = comp.get(a, {})
        p, n = ov.get(a, (float('nan'), 0))
        print(f"{a:<14}{c.get('moves',0):>11}{c.get('evals',0):>11}"
                f"{c.get('ms',0):>9.1f}{p*100:>13.1f}%{n:>8}")

    print()
    print('BY FAMILY (does more compute buy wins?)')
    for f in ('TLA (lookahead)', 'TLA-G (greedy)', 'GA'):
        pts = [(comp.get(a, {}).get('moves', 0), ov.get(a, (0,))[0], a)
                 for a in agents if fam(a) == f]
        pts.sort()
        s = '  '.join(f'{m}->{p*100:.1f}%' for m, p, a in pts)
        print(f'  {f:<18} {s}')

    print()
    print('MATCHED-COMPUTE HEAD-TO-HEADS (the answer to R1.1)')
    print('  Pairs chosen so that measured moves/decision are close:')
    cands = []
    tlas = [a for a in agents if a.startswith('BM-TLA-')]
    tlags = [a for a in agents if a.startswith('BM-TLAG')]
    gas = [a for a in agents if a.startswith('BM-GA-')]
    for A in tlas:
        for B in tlags + gas:
            ca, cb = comp.get(A, {}).get('moves'), comp.get(B, {}).get('moves')
            if not ca or not cb:
                continue
            ratio = max(ca, cb) / min(ca, cb)
            if ratio <= 1.6:
                r = h2h(rows, A, B)
                if r:
                    cands.append((ratio, A, B, ca, cb, r))
    cands.sort()
    if not cands:
        print('    (no pairs within 1.6x)')
    for ratio, A, B, ca, cb, (p, ci, n) in cands:
        verdict = ('LOOKAHEAD WINS' if p - ci > 0.5 else
                     'lookahead loses' if p + ci < 0.5 else 'no difference')
        print(f'    {A} ({ca}) vs {B} ({cb})  ratio {ratio:.2f}x  '
                f'-> {p*100:.1f}% (+-{ci*100:.1f}, n={n})  {verdict}')


if __name__ == '__main__':
    main()
