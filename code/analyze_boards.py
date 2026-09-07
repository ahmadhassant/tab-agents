"""analyze_boards.py - does the ranking and the stacking result survive off 4x8? (R2.6/AE1)"""
import csv, math, os, sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else '../03_Results'
NAME = {'LOOK-A3-simple-stack': 'TLA-S', 'LOOK': 'TLA', 'LOOK-my-eval': 'TLE',
        'LOOK-no-LA': 'TLA-G', 'GA-Original': 'GA-Orig', 'GA-Expert': 'GA-Exp'}
BOARDS = [7, 8, 9, 11]

def load(c):
    p = os.path.join(ROOT, 'board_4x%d' % c, 'games.csv')
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))

def overall(rows):
    w = defaultdict(int); n = defaultdict(int)
    for r in rows:
        a, b = r['agent_p1'], r['agent_p2']
        n[a] += 1; n[b] += 1
        if r['winner'] == '1': w[a] += 1
        elif r['winner'] in ('-1', '2'): w[b] += 1
    return {a: (w[a]/n[a], n[a]) for a in n}

def h2h(rows, A, B):
    sel = [r for r in rows if {r['agent_p1'], r['agent_p2']} == {A, B}]
    if not sel: return None
    w = sum(1 for r in sel if (r['agent_p1']==A and r['winner']=='1')
            or (r['agent_p2']==A and r['winner'] in ('-1','2')))
    pr = w/len(sel)
    return pr*100, 1.96*math.sqrt(pr*(1-pr)/len(sel))*100, len(sel)

print('BOARD-SIZE SWEEP (R2.6 / AE1)')
print()
data = {c: load(c) for c in BOARDS}
avail = [c for c in BOARDS if data[c]]
print('%-10s' % 'agent', end='')
for c in avail: print('%12s' % ('4x%d' % c), end='')
print()
print('-' * (10 + 12*len(avail)))
agents = sorted({a for c in avail for a in overall(data[c])},
                key=lambda a: -overall(data[avail[0]]).get(a, (0,))[0])
for a in agents:
    print('%-10s' % NAME.get(a, a), end='')
    for c in avail:
        o = overall(data[c]).get(a)
        print('%11.1f%%' % (o[0]*100,) if o else '%12s' % '-', end='')
    print()

print()
print('DOES THE STACKING RESULT SURVIVE?  (TLA-S vs TLE head-to-head)')
print('%-10s%14s%14s' % ('board', 'TLA-S wins', 'n'))
print('-' * 38)
for c in avail:
    r = h2h(data[c], 'LOOK-A3-simple-stack', 'LOOK-my-eval')
    if r:
        print('%-10s%12.1f%%%14d' % ('4x%d' % c, r[0], r[2]))

print()
print('RANK ORDER PER BOARD')
for c in avail:
    o = overall(data[c])
    order = [NAME.get(a, a) for _, a in sorted(((v[0], k) for k, v in o.items()), reverse=True)]
    print('  4x%-3d %s' % (c, ' > '.join(order)))
