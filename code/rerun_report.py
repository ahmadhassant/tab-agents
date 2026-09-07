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

RES = _results_dir()

def load(d):
    p = os.path.join(RES, d, 'games.csv')
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('agent_p1') and r.get('agent_p2') and r.get('winner') is not None:
                rows.append(r)
    return rows

def report(d, title):
    rows = load(d)
    if rows is None:
        print('MISSING:', d); return
    w = defaultdict(int); n = defaultdict(int)
    h2w = defaultdict(int); h2n = defaultdict(int)
    for r in rows:
        a, b = r['agent_p1'], r['agent_p2']
        n[a] += 1; n[b] += 1
        h2n[(a, b)] += 1; h2n[(b, a)] += 1
        if r['winner'] == '1':
            w[a] += 1; h2w[(a, b)] += 1
        elif r['winner'] in ('-1', '2'):
            w[b] += 1; h2w[(b, a)] += 1
    print()
    print('=' * 70)
    print(title, ' | games:', len(rows))
    print('%-24s%9s%9s%9s' % ('agent', 'win%', '+/-95CI', 'games'))
    out = []
    for ag in n:
        pr = w[ag] / n[ag]
        ci = 1.96 * math.sqrt(pr * (1 - pr) / n[ag])
        out.append((pr, ag, ci, n[ag]))
    out.sort(reverse=True)
    for pr, ag, ci, nn in out:
        print('%-24s%8.2f%%%9.2f%9d' % (ag, pr * 100, ci * 100, nn))
    return out, h2w, h2n

def duel(h2w, h2n, a, b, label=''):
    k = (a, b)
    if not h2n.get(k):
        print('   (no games %s vs %s)' % (a, b)); return
    p = h2w[k] / h2n[k]
    ci = 1.96 * math.sqrt(p * (1 - p) / h2n[k])
    sig = 'SIG' if abs(p - 0.5) > ci else 'n.s.'
    print('   %s%-22s vs %-22s %6.2f%% +/- %.2f  n=%d  [%s]' % (label, a, b, p * 100, ci * 100, h2n[k], sig))

# ---- 1. board sweep -------------------------------------------------------
print('#' * 70)
print('# BOARD-SIZE SWEEP AT PROTOCOL (1,000 games per pairing)')
print('#' * 70)
for c in (7, 8, 9, 11):
    r = report('board_4x%d_p' % c, 'BOARD 4x%d' % c)
    if r:
        out, h2w, h2n = r
        print('  key duels:')
        duel(h2w, h2n, 'LOOK-A3-simple-stack', 'LOOK')
        duel(h2w, h2n, 'LOOK-A3-simple-stack', 'LOOK-my-eval')
        duel(h2w, h2n, 'LOOK', 'LOOK-no-LA')
        duel(h2w, h2n, 'LOOK-A3-simple-stack', 'GA-Original')

# ---- 2. MCTS depth ablation ----------------------------------------------
print()
print('#' * 70)
print('# MCTS DEPTH ABLATION AT PROTOCOL')
print('#' * 70)
r = report('mcts_tree_p', 'TREE vs FLAT, equal budget')
if r:
    out, h2w, h2n = r
    print('  key duels:')
    duel(h2w, h2n, 'MCTST-800-h', 'MCTS-800-h', 'tree vs flat:   ')
    duel(h2w, h2n, 'MCTST-800-h-d2', 'MCTST-800-h', 'd2 vs d4:       ')
    duel(h2w, h2n, 'MCTST-800-h-d6', 'MCTST-800-h', 'd6 vs d4:       ')
    duel(h2w, h2n, 'MCTST-800-h-d6', 'MCTST-800-h-d2', 'd6 vs d2:       ')
    duel(h2w, h2n, 'LOOK-A3-simple-stack', 'MCTST-800-h', 'TLA-S vs tree:  ')

# ---- 3. budget ladder -----------------------------------------------------
print()
print('#' * 70)
print('# MATCHED-COMPUTE BUDGET LADDER AT PROTOCOL')
print('#' * 70)
r = report('budget_sweep_p', 'BUDGET LADDER')
if r:
    out, h2w, h2n = r
    print('  matched-tier duels (TLA vs GA at equal budget):')
    for k in (1, 2, 3, 4):
        duel(h2w, h2n, 'BM-TLA-%d' % k, 'BM-GA-%d' % k, 'tier %d: ' % k)
    print('  matched-tier duels (TLA-greedy vs GA at equal budget):')
    for k in (1, 2, 3, 4):
        duel(h2w, h2n, 'BM-TLAG-%d' % k, 'BM-GA-%d' % k, 'tier %d: ' % k)
    print('  lookahead value at each tier (TLA vs its own greedy):')
    for k in (1, 2, 3, 4):
        duel(h2w, h2n, 'BM-TLA-%d' % k, 'BM-TLAG-%d' % k, 'tier %d: ' % k)
