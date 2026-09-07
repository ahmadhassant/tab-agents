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

import csv, math, os, sys
from collections import defaultdict

P = _os.path.join(_results_dir(), 'final_roster', 'games.csv')
rows = []
with open(P, newline='') as f:
    rd = csv.DictReader(f)
    for r in rd:
        if r.get('winner') is None:
            continue
        if not r.get('agent_p1') or not r.get('agent_p2'):
            continue
        rows.append(r)
print('rows parsed:', len(rows))

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

print()
print('OVERALL round-robin standings')
print('%-24s%9s%9s%9s' % ('agent', 'win%', '+/-95CI', 'games'))
out = []
for ag in n:
    pr = w[ag] / n[ag]
    ci = 1.96 * math.sqrt(pr * (1 - pr) / n[ag])
    out.append((pr, ag, ci, n[ag]))
out.sort(reverse=True)
for pr, ag, ci, nn in out:
    print('%-24s%8.2f%%%9.2f%9d' % (ag, pr * 100, ci * 100, nn))

order = [ag for _, ag, _, _ in out]
print()
print('HEAD TO HEAD  (row win%% vs column, both sides pooled)')
hdr = '%-22s' % 'row \\ col'
for c in order:
    hdr += '%9s' % c[:8]
print(hdr)
for a in order:
    line = '%-22s' % a[:22]
    for b in order:
        if a == b:
            line += '%9s' % '-'
        else:
            k = (a, b)
            if h2h_n[k]:
                line += '%8.1f%%' % (100.0 * h2h_w[k] / h2h_n[k])
            else:
                line += '%9s' % '.'
    print(line)

# top-two separation test
if len(out) >= 2:
    p1, a1, c1, n1 = out[0]
    p2, a2, c2, n2 = out[1]
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p1 - p2) / se if se > 0 else 0.0
    print()
    print('TOP-2 SEPARATION: %s %.2f%% vs %s %.2f%%   diff=%.2f pp  z=%.2f' % (
        a1, p1 * 100, a2, p2 * 100, (p1 - p2) * 100, z))
    print('  intervals: [%.2f, %.2f] vs [%.2f, %.2f]  -> %s' % (
        (p1 - c1) * 100, (p1 + c1) * 100, (p2 - c2) * 100, (p2 + c2) * 100,
        'DISJOINT' if (p1 - c1) > (p2 + c2) else 'OVERLAP'))
    k = (a1, a2)
    if h2h_n[k]:
        hp = h2h_w[k] / h2h_n[k]
        hci = 1.96 * math.sqrt(hp * (1 - hp) / h2h_n[k])
        print('  direct h2h: %s beats %s %.2f%% +/- %.2f  (n=%d)' % (
            a1, a2, hp * 100, hci * 100, h2h_n[k]))
