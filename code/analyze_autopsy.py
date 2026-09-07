"""analyze_autopsy.py - the complete autopsy (R3.14), two reference opponents."""
import csv, math, sys
from collections import defaultdict

P = sys.argv[1] if len(sys.argv) > 1 else '../03_Results/autopsy_full/games.csv'
rows = list(csv.DictReader(open(P)))
print('games:', len(rows))

def h2h(A, B):
    sel = [r for r in rows if {r['agent_p1'], r['agent_p2']} == {A, B}]
    if not sel:
        return None
    w = sum(1 for r in sel
            if (r['agent_p1'] == A and r['winner'] == '1')
            or (r['agent_p2'] == A and r['winner'] in ('-1', '2')))
    n = len(sel); pr = w / n
    se = math.sqrt(0.25 / n); z = (pr - 0.5) / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return pr * 100, 1.96 * math.sqrt(pr * (1 - pr) / n) * 100, n, pv

VAR = [('LOOK-my-eval',          'TLE baseline'),
       ('LOOK-A1-material5',     'M  material weight 15->5'),
       ('LOOK-A2-no-frozendist', 'F  frozen distinction off'),
       ('LOOK-A3-simple-stack',  'X  stacking quadratic->linear'),
       ('LOOK-A4-strong-danger', 'D  danger scale 10->30'),
       ('LOOK-A5-stacking-off',  'N  stacking penalty off'),
       ('LOOK-A7-no-hiding',     'H  home hiding off      [NEW]'),
       ('LOOK-A8-no-central',    'C  central bonus off    [NEW]'),
       ('LOOK-A9-no-entry',      'E  entry pressure off   [NEW]'),
       ('LOOK-A6-all-fixed',     'A  M+F+D+X combined')]

print()
print('COMPLETE AUTOPSY - each variant vs TWO reference opponents')
print('%-34s%16s%16s' % ('variant', 'vs GA-Expert', 'vs TLA'))
print('-' * 66)
res = {}
for key, label in VAR:
    a, b = h2h(key, 'GA-Expert'), h2h(key, 'LOOK')
    res[key] = (a, b)
    sa = ('%5.1f%% +/-%.1f' % (a[0], a[1])) if a else 'n/a'
    sb = ('%5.1f%% +/-%.1f' % (b[0], b[1])) if b else 'n/a'
    print('%-34s%16s%16s' % (label, sa, sb))

ba, bb = res['LOOK-my-eval'][0][0], res['LOOK-my-eval'][1][0]
print()
print('DELTA FROM TLE BASELINE (GA-Expert %.1f%%, TLA %.1f%%)' % (ba, bb))
print('%-34s%16s%12s' % ('variant', 'd vs GA-Expert', 'd vs TLA'))
print('-' * 62)
for key, label in VAR[1:]:
    a, b = res[key]
    print('%-34s%+15.1f%+12.1f' % (label, a[0] - ba, b[0] - bb))

print()
print('HOLM-CORRECTED: each variant played DIRECTLY against the TLE baseline')
tests = []
for key, label in VAR[1:]:
    r = h2h(key, 'LOOK-my-eval')
    if r:
        tests.append((r[3], key, label, r[0], r[2]))
tests.sort()
m = len(tests)
print('%-34s%10s%12s%8s' % ('variant', 'vs TLE', 'p', 'Holm'))
print('-' * 64)
still = True
for i, (pv, key, label, pr, n) in enumerate(tests):
    thr = 0.05 / (m - i)
    sig = still and pv < thr
    if not sig:
        still = False
    print('%-34s%9.1f%%%12.2e%8s' % (label, pr, pv, 'SIG' if sig else 'ns'))
