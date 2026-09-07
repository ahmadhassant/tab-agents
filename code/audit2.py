import os as _os
# results live beside this folder in the review package;
# override with the TAB_RESULTS environment variable if needed.
def _results_dir():
    here = _os.path.dirname(_os.path.abspath(__file__))
    env = _os.environ.get('TAB_RESULTS')
    if env:
        return env
    for cand in ('../05_Results', '../03_Results', '03_Results'):
        p = _os.path.normpath(_os.path.join(here, cand))
        if _os.path.isdir(p):
            return p
    return _os.path.normpath(_os.path.join(here, '../05_Results'))

import csv, math, os, random
from collections import defaultdict

RES = _results_dir()
NICE = {'LOOK-A3-simple-stack': 'TLA-S', 'LOOK': 'TLA', 'LOOK-no-enum': 'TLA-B',
        'LOOK-no-LA': 'TLA-G', 'LOOK-my-eval': 'TLE'}

def load(d):
    p = os.path.join(RES, d, 'games.csv')
    out = []
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('agent_p1') and r.get('agent_p2') and r.get('winner') is not None:
                out.append(r)
    return out

def ci(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)

# ---------------- 1. AUTOPSY (tab:autopsy_full) ---------------------------
print('=' * 78)
print('AUTOPSY_FULL  -> source for the missing Table tab:autopsy_full')
print('=' * 78)
rows = load('autopsy_full')
print('games:', len(rows))
hw = defaultdict(int); hn = defaultdict(int)
for r in rows:
    a, b = r['agent_p1'], r['agent_p2']
    hn[(a, b)] += 1; hn[(b, a)] += 1
    if r['winner'] == '1':
        hw[(a, b)] += 1
    elif r['winner'] in ('-1', '2'):
        hw[(b, a)] += 1
agents = sorted({r['agent_p1'] for r in rows} | {r['agent_p2'] for r in rows})
print('agents:', agents)
print()
print('%-22s%22s%22s' % ('variant', 'vs GA-Expert', 'vs TLA(LOOK)'))
for a in agents:
    line = '%-22s' % NICE.get(a, a)
    for ref in ('GA-Expert', 'LOOK'):
        k = (a, ref)
        if a == ref or not hn.get(k):
            line += '%22s' % '-'
        else:
            p = hw[k] / hn[k]
            line += '%15.2f%% +/-%.2f' % (p * 100, ci(p, hn[k]) * 100)
    print(line)

# baseline deltas vs TLE
print()
print('DELTA FROM TLE BASELINE (pp), per reference opponent:')
base = {}
for ref in ('GA-Expert', 'LOOK'):
    k = ('LOOK-my-eval', ref)
    if hn.get(k):
        base[ref] = hw[k] / hn[k]
for a in agents:
    if a in ('GA-Expert', 'LOOK', 'LOOK-my-eval'):
        continue
    parts = []
    for ref in ('GA-Expert', 'LOOK'):
        k = (a, ref)
        if hn.get(k) and ref in base:
            parts.append('%+7.2f' % ((hw[k] / hn[k] - base[ref]) * 100))
        else:
            parts.append('%7s' % '-')
    print('   %-22s %s' % (NICE.get(a, a), '  '.join(parts)))

# ---------------- 2. BRADLEY-TERRY (tab:bt) --------------------------------
print()
print('=' * 78)
print('BRADLEY-TERRY on tournament_main_28k -> source for the missing Table tab:bt')
print('=' * 78)
main = load('tournament_main_28k')
ags = sorted({r['agent_p1'] for r in main} | {r['agent_p2'] for r in main})
idx = {a: i for i, a in enumerate(ags)}

def fit_bt(pairs, iters=3000):
    """pairs: list of (i, j, wins_i, wins_j) aggregated."""
    p = [1.0] * len(ags)
    for _ in range(iters):
        num = [0.0] * len(ags); den = [0.0] * len(ags)
        for i, j, wi, wj in pairs:
            s = p[i] + p[j]
            num[i] += wi; den[i] += (wi + wj) / s
            num[j] += wj; den[j] += (wi + wj) / s
        newp = [(num[k] / den[k] if den[k] > 0 else p[k]) for k in range(len(ags))]
        g = sum(newp) / len(newp)
        newp = [x / g for x in newp]
        if max(abs(newp[k] - p[k]) for k in range(len(ags))) < 1e-12:
            p = newp; break
        p = newp
    return p

def agg(rows_):
    d = defaultdict(lambda: [0, 0])
    for r in rows_:
        a, b = r['agent_p1'], r['agent_p2']
        i, j = idx[a], idx[b]
        key = (min(i, j), max(i, j))
        aw = 1 if r['winner'] == '1' else 0
        bw = 1 if r['winner'] in ('-1', '2') else 0
        if i <= j:
            d[key][0] += aw; d[key][1] += bw
        else:
            d[key][0] += bw; d[key][1] += aw
    return [(k[0], k[1], v[0], v[1]) for k, v in d.items()]

point = fit_bt(agg(main))
# normalise so strengths sum to 1 (the paper quotes 0.1443-style numbers)
tot = sum(point)
point_n = [x / tot for x in point]

# clustered bootstrap over games
random.seed(12345)
boots = [[] for _ in ags]
N = len(main)
for b in range(400):
    samp = [main[random.randrange(N)] for _ in range(N)]
    try:
        pb = fit_bt(agg(samp), iters=500)
        t = sum(pb)
        for k in range(len(ags)):
            boots[k].append(pb[k] / t)
    except Exception:
        pass
print('%-22s%10s%22s' % ('agent', 'strength', '95% CI (400 boot)'))
order = sorted(range(len(ags)), key=lambda k: -point_n[k])
for k in order:
    v = sorted(boots[k])
    lo = v[int(0.025 * len(v))]; hi = v[int(0.975 * len(v))]
    print('%-22s%10.4f      [%.4f, %.4f]' % (NICE.get(ags[k], ags[k]), point_n[k], lo, hi))
print()
print('adjacent-pair separation (CIs disjoint?):')
for a in range(len(order) - 1):
    k1, k2 = order[a], order[a + 1]
    v1 = sorted(boots[k1]); v2 = sorted(boots[k2])
    lo1 = v1[int(0.025 * len(v1))]; hi2 = v2[int(0.975 * len(v2))]
    print('   %-10s vs %-10s : %s' % (NICE.get(ags[k1], ags[k1]), NICE.get(ags[k2], ags[k2]),
                                      'SEPARATE' if lo1 > hi2 else 'overlap'))

# ---------------- 3. ALIGNMENT ---------------------------------------------
print()
print('=' * 78)
print('ALIGNMENT WITH TLE -> source for Table tab:alignment (must include TLE)')
print('=' * 78)
with open(os.path.join(RES, 'alignment_with_tle', 'summary.csv')) as f:
    for line in f:
        print('   ' + line.rstrip())
