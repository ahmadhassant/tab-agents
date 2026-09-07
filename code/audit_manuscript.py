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

"""Recompute every numeric claim the manuscript makes, straight from the CSVs."""
import csv, math, os
from collections import defaultdict

RES = _results_dir()

NICE = {
    'LOOK-A3-simple-stack': 'TLA-S',
    'LOOK': 'TLA',
    'LOOK-no-enum': 'TLA-B',
    'LOOK-no-LA': 'TLA-G',
    'LOOK-my-eval': 'TLE',
}

def load(d):
    p = os.path.join(RES, d, 'games.csv')
    if not os.path.exists(p):
        return None
    out = []
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('agent_p1') and r.get('agent_p2') and r.get('winner') is not None:
                out.append(r)
    return out

def tally(rows):
    w = defaultdict(int); n = defaultdict(int)
    hw = defaultdict(int); hn = defaultdict(int)
    for r in rows:
        a, b = r['agent_p1'], r['agent_p2']
        n[a] += 1; n[b] += 1
        hn[(a, b)] += 1; hn[(b, a)] += 1
        if r['winner'] == '1':
            w[a] += 1; hw[(a, b)] += 1
        elif r['winner'] in ('-1', '2'):
            w[b] += 1; hw[(b, a)] += 1
    return w, n, hw, hn

def ci(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)

print('=' * 78)
print('TABLE II SOURCE: tournament_main_28k')
print('=' * 78)
rows = load('tournament_main_28k')
print('total games in file:', len(rows))
w, n, hw, hn = tally(rows)
tot = sum(n.values()) // 2
print('implied pairing count:', len(set(frozenset(k) for k in hn)) // 1)
print()
print('%-24s%14s%9s%9s%9s' % ('agent', 'wins/games', 'win%', '+/-CI', 'rank'))
out = sorted(((w[a] / n[a], a) for a in n), reverse=True)
for i, (p, a) in enumerate(out, 1):
    print('%-24s%7d /%6d%8.2f%%%9.2f%9d' % (NICE.get(a, a), w[a], n[a], p * 100, ci(p, n[a]) * 100, i))

def h2h(a, b, claim=''):
    k = (a, b)
    if not hn.get(k):
        print('   MISSING pairing %s vs %s' % (a, b)); return
    p = hw[k] / hn[k]
    c = ci(p, hn[k])
    sig = 'SIG' if (p - c) > 0.5 else ('SIG-LOSS' if (p + c) < 0.5 else 'n.s.')
    print('   %-8s vs %-8s  %d/%d = %.2f%% +/- %.2f  [%s]   %s' % (
        NICE.get(a, a), NICE.get(b, b), hw[k], hn[k], p * 100, c * 100, sig, claim))

print()
print('HEAD-TO-HEADS THE TEXT CLAIMS:')
h2h('LOOK-A3-simple-stack', 'LOOK', 'text claims 54.9% (549/1000)')
h2h('LOOK', 'LOOK-no-LA', 'text claims 60.0% (600/1000)')
h2h('LOOK', 'LOOK-no-enum', 'text claims 49.3% (493/1000)')
h2h('LOOK', 'GA-Original', 'text claims 56.0%')
h2h('LOOK', 'GA-Expert', 'text claims 52.2%')
h2h('LOOK', 'GA-Fuzzy', 'text claims 59.6%')

print()
print('TLA-S AGAINST EVERY OPPONENT (text claims all > 52%, "wins half outright"):')
sig_count = 0
for _, b in out:
    if b == 'LOOK-A3-simple-stack':
        continue
    k = ('LOOK-A3-simple-stack', b)
    if hn.get(k):
        p = hw[k] / hn[k]; c = ci(p, hn[k])
        s = 'SIG' if (p - c) > 0.5 else 'n.s.'
        if s == 'SIG':
            sig_count += 1
        print('   vs %-14s %.2f%% +/- %.2f  [%s]' % (NICE.get(b, b), p * 100, c * 100, s))
print('   -> significant wins: %d of %d pairings' % (sig_count, len(out) - 1))

print()
print('DERIVED QUANTITIES THE TEXT CLAIMS:')
pS = w['LOOK-A3-simple-stack'] / n['LOOK-A3-simple-stack']
pT = w['LOOK'] / n['LOOK']
pG = w['LOOK-no-LA'] / n['LOOK-no-LA']
pB = w['LOOK-no-enum'] / n['LOOK-no-enum']
print('   TLA-S overall           = %.2f%%  (text says 56.05)' % (pS * 100))
print('   TLA-S minus TLA         = %.2f pp (text says 2.3 pp)' % ((pS - pT) * 100))
print('   TLA overall             = %.2f%%  (text says 53.7)' % (pT * 100))
print('   TLA-G overall           = %.2f%%  (text says 45.3)' % (pG * 100))
print('   TLA minus TLA-G         = %.2f pp (text says 8.4 pp)' % ((pT - pG) * 100))
print('   TLA minus TLA-B         = %.2f pp (text says 0.13 pp)' % ((pT - pB) * 100))
print('   GA-Original overall     = %.2f%%  (text says 49.3)' % (w['GA-Original'] / n['GA-Original'] * 100))
print('   GA-Expert overall       = %.2f%%  (text says 47.9)' % (w['GA-Expert'] / n['GA-Expert'] * 100))
print('   GA-Fuzzy overall        = %.2f%%  (text says 44.1)' % (w['GA-Fuzzy'] / n['GA-Fuzzy'] * 100))
if 'PSA' in n:
    print('   PSA overall            = %.2f%%  (text: "GA-Fuzzy lowest among smart agents")' % (w['PSA'] / n['PSA'] * 100))
print('   per-agent n             = %d  -> CI = +/-%.2f pp (text says +/-1.26 at n=6000)' % (
    n['LOOK'], ci(0.5, n['LOOK']) * 100))
