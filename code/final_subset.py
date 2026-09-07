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

import csv, math
from collections import defaultdict

P = _os.path.join(_results_dir(), 'final_roster', 'games.csv')
EXCL = {'MCTS-800-h', 'Random'}   # pairings still in flight

rows = []
with open(P, newline='') as f:
    for r in csv.DictReader(f):
        if not r.get('agent_p1') or not r.get('agent_p2') or r.get('winner') is None:
            continue
        if r['agent_p1'] in EXCL or r['agent_p2'] in EXCL:
            continue
        rows.append(r)

w = defaultdict(int); n = defaultdict(int)
h2h_w = defaultdict(int); h2h_n = defaultdict(int)
for r in rows:
    a, b = r['agent_p1'], r['agent_p2']
    n[a] += 1; n[b] += 1
    h2h_n[(a, b)] += 1; h2h_n[(b, a)] += 1
    if r['winner'] == '1':
        w[a] += 1; h2h_w[(a, b)] += 1
    elif r['winner'] in ('-1', '2'):
        w[b] += 1; h2h_w[(b, a)] += 1

print('BALANCED SUB-ROUND-ROBIN (10 agents, MCTS/Random excluded; every pairing complete)')
print('games used:', len(rows))
print('%-24s%9s%9s%9s' % ('agent', 'win%', '+/-95CI', 'games'))
out = []
for ag in n:
    pr = w[ag] / n[ag]
    ci = 1.96 * math.sqrt(pr * (1 - pr) / n[ag])
    out.append((pr, ag, ci, n[ag]))
out.sort(reverse=True)
for pr, ag, ci, nn in out:
    print('%-24s%8.2f%%%9.2f%9d' % (ag, pr * 100, ci * 100, nn))

print()
for i in range(min(3, len(out) - 1)):
    p1, a1, c1, n1 = out[i]
    p2, a2, c2, n2 = out[i + 1]
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p1 - p2) / se if se else 0.0
    # two-sided p from z
    pv = math.erfc(abs(z) / math.sqrt(2))
    print('rank %d vs %d: %s %.2f%% vs %s %.2f%%  diff=%.2f pp  z=%.2f  p=%.4f  %s' % (
        i + 1, i + 2, a1, p1 * 100, a2, p2 * 100, (p1 - p2) * 100, z, pv,
        'CI-disjoint' if (p1 - c1) > (p2 + c2) else 'CI-overlap'))

print()
print('DIRECT HEAD-TO-HEAD among top 3 (row beats col, %):')
top = [a for _, a, _, _ in out[:3]]
for a in top:
    for b in top:
        if a == b:
            continue
        k = (a, b)
        hp = h2h_w[k] / h2h_n[k]
        hci = 1.96 * math.sqrt(hp * (1 - hp) / h2h_n[k])
        print('  %-22s vs %-22s %6.2f%% +/- %.2f (n=%d)' % (a, b, hp * 100, hci * 100, h2h_n[k]))
