"""analyze_tlann.py - standings + key head-to-heads for a TLA-NN tournament."""
import csv, math, sys
from collections import defaultdict

P = sys.argv[1] if len(sys.argv) > 1 else '../03_Results/tlann2/games.csv'
rows = list(csv.DictReader(open(P)))
print('games:', len(rows))
w = defaultdict(int); n = defaultdict(int)
for r in rows:
    a, b = r['agent_p1'], r['agent_p2']
    n[a] += 1; n[b] += 1
    if r['winner'] == '1': w[a] += 1
    elif r['winner'] in ('-1', '2'): w[b] += 1

print()
print('OVERALL')
for pr, a, nn in sorted(((w[a]/n[a], a, n[a]) for a in n), reverse=True):
    ci = 1.96 * math.sqrt(pr*(1-pr)/nn)
    print('%-26s%7.2f%%  +/-%.2f   n=%d' % (a, pr*100, ci*100, nn))

def h2h(A, B):
    sel = [r for r in rows if {r['agent_p1'], r['agent_p2']} == {A, B}]
    if not sel: return None
    wa = sum(1 for r in sel
             if (r['agent_p1']==A and r['winner']=='1')
             or (r['agent_p2']==A and r['winner'] in ('-1','2')))
    pr = wa/len(sel)
    return pr*100, 1.96*math.sqrt(pr*(1-pr)/len(sel))*100, len(sel)

print()
print('KEY HEAD-TO-HEADS (row wins %)')
for A, B in [('TLA-NN','LOOK-A3-simple-stack'), ('TLA-NN','LOOK'),
             ('TLA-NN','TD-Greedy'), ('TLA-NN','GA-Original'),
             ('TLA-NN','GA-Expert'), ('TLA-NN','PSA'),
             ('TLA-NN','LOOK-no-LA'), ('TLA-NN','MCTS-800-h'),
             ('TLA-NN','Random')]:
    r = h2h(A, B)
    if r:
        verdict = 'WINS' if r[0]-r[1] > 50 else ('loses' if r[0]+r[1] < 50 else 'n.s.')
        print('  %-24s vs %-24s %6.1f%% +/-%.1f  (n=%d)  %s' % (A, B, r[0], r[1], r[2], verdict))
