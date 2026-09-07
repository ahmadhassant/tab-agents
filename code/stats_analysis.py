"""
stats_analysis.py - the statistical re-analyses the reviewers asked for.

C-1 (R1.4)  TOST equivalence test for enumeration vs biased sampling.
            "Statistically the same" is a non-significance claim; equivalence
            needs a stated margin and two one-sided tests.
C-2 (R3.12) Bradley-Terry strengths with bootstrap CIs, replacing the
            informal "three statistically discernible tiers", plus
            Holm-corrected pairwise tests.
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    agents = sorted(set(r['agent_p1'] for r in rows) |
                      set(r['agent_p2'] for r in rows))
    return rows, agents


def head_to_head(rows, A, B):
    sel = [r for r in rows if {r['agent_p1'], r['agent_p2']} == {A, B}]
    wa = sum(1 for r in sel
               if (r['agent_p1'] == A and r['winner'] == '1') or
                  (r['agent_p2'] == A and r['winner'] in ('-1', '2')))
    return wa, len(sel)


def tost(wins, n, margin=0.02, alpha=0.05):
    """Two one-sided tests for equivalence of p to 0.5 within +-margin."""
    p = wins / n
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = 0.5 - margin, 0.5 + margin
    z1 = (p - lo) / se          # H0: p <= lo
    z2 = (hi - p) / se          # H0: p >= hi
    ncdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p1, p2 = 1 - ncdf(z1), 1 - ncdf(z2)
    pval = max(p1, p2)
    ci = 1.96 * se
    return {'p': p, 'ci': ci, 'n': n, 'p_tost': pval,
              'equivalent': pval < alpha, 'margin': margin}


def bradley_terry(rows, agents, iters=300):
    idx = {a: i for i, a in enumerate(agents)}
    k = len(agents)
    W = np.zeros((k, k))
    for r in rows:
        i, j = idx[r['agent_p1']], idx[r['agent_p2']]
        if r['winner'] == '1':
            W[i, j] += 1
        elif r['winner'] in ('-1', '2'):
            W[j, i] += 1
        else:
            W[i, j] += 0.5
            W[j, i] += 0.5
    p = np.ones(k)
    N = W + W.T
    for _ in range(iters):
        newp = np.zeros(k)
        for i in range(k):
            num = W[i].sum()
            den = 0.0
            for j in range(k):
                if i != j and N[i, j] > 0:
                    den += N[i, j] / (p[i] + p[j])
            newp[i] = num / den if den > 0 else p[i]
        newp = np.clip(newp, 1e-12, None)
        newp /= newp.sum()
        if np.max(np.abs(newp - p)) < 1e-12:
            p = newp
            break
        p = newp
    return p


def bt_bootstrap(rows, agents, B=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rows)
    arr = np.array(rows, dtype=object)
    out = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        out.append(bradley_terry(list(arr[idx]), agents, iters=120))
    return np.array(out)


def holm(pvals, names, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    res, prev = {}, 0.0
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        sig = pvals[i] < thr
        if not sig:
            for j in order[rank:]:
                res[names[j]] = (pvals[j], False)
            break
        res[names[i]] = (pvals[i], True)
    for nm in names:
        res.setdefault(nm, (pvals[names.index(nm)], False))
    return res


def two_prop_p(w, n):
    p = w / n
    se = math.sqrt(0.25 / n)
    z = (p - 0.5) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', default='../03_Results/tournament_main_28k/games.csv')
    ap.add_argument('--boot', type=int, default=400)
    args = ap.parse_args()

    rows, agents = load(args.games)
    print(f'loaded {len(rows)} games, {len(agents)} agents\n')

    # ---- C-1: TOST -------------------------------------------------------
    print('=' * 70)
    print('C-1  EQUIVALENCE OF ENUMERATION AND BIASED SAMPLING (R1.4)')
    print('=' * 70)
    for A, B in [('LOOK', 'LOOK-no-enum')]:
        w, n = head_to_head(rows, A, B)
        for margin in (0.02, 0.03, 0.05):
            t = tost(w, n, margin=margin)
            verdict = ('EQUIVALENT' if t['equivalent']
                         else 'not established')
            print(f"  {A} vs {B}: {t['p']*100:.1f}% (+-{t['ci']*100:.1f}), "
                    f"n={n}  margin=+-{margin*100:.0f}pp  "
                    f"p_TOST={t['p_tost']:.4f}  -> {verdict}")
    print()

    # ---- C-2: Bradley-Terry ---------------------------------------------
    print('=' * 70)
    print('C-2  BRADLEY-TERRY STRENGTHS WITH BOOTSTRAP CIs (R3.12)')
    print('=' * 70)
    p = bradley_terry(rows, agents)
    boots = bt_bootstrap(rows, agents, B=args.boot)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    order = np.argsort(-p)
    print(f"  {'agent':<24}{'BT strength':>13}{'95% CI':>22}")
    for i in order:
        print(f'  {agents[i]:<24}{p[i]:>13.4f}   [{lo[i]:.4f}, {hi[i]:.4f}]')
    print()
    print('  Overlap check (do adjacent agents separate?):')
    for a, b in zip(order[:-1], order[1:]):
        sep = lo[a] > hi[b]
        print(f'    {agents[a]:<24} vs {agents[b]:<24} '
                f'{"SEPARATED" if sep else "overlapping"}')
    print()

    # ---- Holm-corrected pairwise ----------------------------------------
    print('=' * 70)
    print('  HOLM-CORRECTED PAIRWISE HEAD-TO-HEADS')
    print('=' * 70)
    names, pv, info = [], [], {}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            w, n = head_to_head(rows, agents[i], agents[j])
            if n == 0:
                continue
            nm = f'{agents[i]} vs {agents[j]}'
            names.append(nm); pv.append(two_prop_p(w, n))
            info[nm] = (w / n, n)
    res = holm(np.array(pv), names)
    nsig = sum(1 for v in res.values() if v[1])
    print(f'  {nsig}/{len(names)} pairings significant after Holm correction')
    for nm in names:
        pval, sig = res[nm]
        rate, n = info[nm]
        if sig:
            print(f'    {nm:<52} {rate*100:5.1f}%  p={pval:.2e}  *')


if __name__ == '__main__':
    main()
