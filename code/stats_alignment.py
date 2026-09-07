"""
stats_alignment.py - the alignment re-analysis reviewers asked for (C-3).

R3.17  342 decisions are clustered inside 11 games; treating them as
       independent Bernoulli trials understates uncertainty.
R3.18  Every agent is evaluated on the SAME positions, so agent comparisons
       are paired and should be tested as such.
R3.19  The four metrics are nested/dependent, so agreeing on all four is not
       four independent confirmations. One primary metric must be declared.

This computes: game-level clustered bootstrap CIs, McNemar paired tests
between agents on the primary metric (soldier-set), and the correlation
structure of the four metrics.
"""
import csv
import math
from collections import defaultdict

import numpy as np

PRIMARY = 'soldier_set_match'


def load(path):
    rows = list(csv.DictReader(open(path)))
    agents = sorted(set(r['agent'] for r in rows))
    games = sorted(set(r['game_id'] for r in rows))
    return rows, agents, games


def clustered_bootstrap(rows, agents, games, metric=PRIMARY, B=10000, seed=0):
    """Resample GAMES with replacement, not decisions."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r['agent']][r['game_id']].append(float(r[metric]))
    rng = np.random.default_rng(seed)
    out = {}
    G = len(games)
    for a in agents:
        vals = [np.array(by[a][g]) for g in games if by[a][g]]
        point = np.mean(np.concatenate(vals))
        boots = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(vals), len(vals))
            boots[b] = np.mean(np.concatenate([vals[i] for i in pick]))
        out[a] = (point, np.percentile(boots, 2.5),
                    np.percentile(boots, 97.5), np.std(boots))
    return out


def naive_ci(rows, agents, metric=PRIMARY):
    out = {}
    for a in agents:
        v = np.array([float(r[metric]) for r in rows if r['agent'] == a])
        p = v.mean()
        out[a] = (p, 1.96 * math.sqrt(p * (1 - p) / len(v)))
    return out


def mcnemar(rows, A, B, metric=PRIMARY):
    """Paired test on the same decisions."""
    ka = {(r['game_id'], r['turn']): float(r[metric])
            for r in rows if r['agent'] == A}
    kb = {(r['game_id'], r['turn']): float(r[metric])
            for r in rows if r['agent'] == B}
    keys = set(ka) & set(kb)
    b = sum(1 for k in keys if ka[k] > kb[k])     # A right, B wrong
    c = sum(1 for k in keys if ka[k] < kb[k])     # B right, A wrong
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)        # continuity-corrected
    p = math.exp(-chi2 / 2)                        # 1 df survival approx
    return b, c, min(1.0, p)


def main():
    path = ('../05_Server_Runs/look_bundles/canonical_TableIV_source/'
              'results_alignment/positions.csv')
    rows, agents, games = load(path)
    n_dec = len({(r['game_id'], r['turn']) for r in rows})
    print(f'{len(rows)} rows | {len(agents)} agents | {len(games)} games '
            f'| {n_dec} decisions\n')

    print('decisions per game:')
    per = defaultdict(int)
    for r in rows:
        if r['agent'] == agents[0]:
            per[r['game_id']] += 1
    for g in games:
        print(f'  {g:<12} {per[g]:4d}')
    print()

    print('=' * 78)
    print('C-3  PRIMARY METRIC (soldier-set): naive vs GAME-CLUSTERED CIs')
    print('=' * 78)
    nv = naive_ci(rows, agents)
    cb = clustered_bootstrap(rows, agents, games, B=10000)
    print(f"  {'agent':<22}{'rate':>8}{'naive +-':>11}{'clustered 95% CI':>26}"
            f"{'widening':>10}")
    for a in sorted(agents, key=lambda x: -cb[x][0]):
        p, ci = nv[a]
        pt, lo, hi, sd = cb[a]
        widen = (hi - lo) / (2 * ci) if ci > 0 else float('nan')
        print(f'  {a:<22}{pt*100:7.1f}%{ci*100:10.1f}'
                f'   [{lo*100:5.1f}, {hi*100:5.1f}]{widen:>13.2f}x')
    print()

    print('=' * 78)
    print('  PAIRED McNEMAR TESTS (primary metric, same 342 decisions)')
    print('=' * 78)
    top = sorted(agents, key=lambda x: -cb[x][0])
    ref = top[0]
    for a in top[1:]:
        b, c, p = mcnemar(rows, ref, a)
        star = '*' if p < 0.05 else ' '
        print(f'  {ref:<22} vs {a:<22} b={b:3d} c={c:3d}  p={p:.4f} {star}')
    print()

    print('=' * 78)
    print('  METRIC DEPENDENCE (R3.19) - correlation across decisions')
    print('=' * 78)
    mets = ['soldier_set_match', 'strict_match', 'first_match', 'jaccard']
    M = {m: np.array([float(r[m]) for r in rows]) for m in mets}
    print('       ' + ''.join(f'{m[:12]:>14}' for m in mets))
    for m in mets:
        line = f'  {m[:12]:<12}'
        for n in mets:
            line += f'{np.corrcoef(M[m], M[n])[0,1]:>14.3f}'
        print(line)
    nested = np.mean((M['strict_match'] == 1) <= (M['soldier_set_match'] == 1))
    print(f'\n  strict is nested within soldier-set in '
            f'{nested*100:.1f}% of decisions (should be 100%)')


if __name__ == '__main__':
    main()
