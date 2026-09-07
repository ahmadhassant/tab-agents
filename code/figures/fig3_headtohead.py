"""
fig3_headtohead.py -- Head-to-head win rates for TLA-S vs every other agent.

Input:  games.csv from tournament_grid.py
Output: fig3_headtohead.pdf (single-column IEEE figure)

Shows TLA-S's win rate against each opponent across all 1000 games per pairing,
with 95% CIs. Highlights the four pairings where TLA-S has a statistically
significant lead. This is the visualization that makes TLA-S's champion
status concrete.
"""

import sys
import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

RENAME = {
    'LOOK':                 'TLA',
    'LOOK-A3-simple-stack': 'TLA-S',
    'LOOK-no-LA':           'TLA-G',
    'LOOK-no-enum':         'TLA-B',
    'GA-Expert':            'GA-Expert',
    'GA-Original':          'GA-Original',
    'GA-Fuzzy':             'GA-Fuzzy',
    'PSA':                  'PSA',
}

# Top to bottom in the rendered figure (y axis is inverted below).
OPPONENTS_ORDERED = ['PSA', 'GA-Fuzzy', 'TLA-G', 'GA-Expert', 'GA-Original',
                      'TLA-B', 'TLA']
CHAMPION = 'TLA-S'


def load_h2h(csv_path):
    """Return {opponent: (wins, total)} for TLA-S vs each other agent."""
    wins = defaultdict(int)
    total = defaultdict(int)
    with open(csv_path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            a = RENAME.get(row['agent_p1'], row['agent_p1'])
            b = RENAME.get(row['agent_p2'], row['agent_p2'])
            winner = int(row['winner'])
            if a == CHAMPION and b in OPPONENTS_ORDERED:
                total[b] += 1
                if winner == 1: wins[b] += 1
            elif b == CHAMPION and a in OPPONENTS_ORDERED:
                total[a] += 1
                if winner == -1: wins[a] += 1
    return wins, total


def make_figure(wins, total, out_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Ancient palette
    ANCIENT = {
        'gold_win':    '#c89345',   # significant wins
        'gold_edge':   '#8a5a1c',
        'tan_tie':     '#b8a784',   # ties
        'blue_loss':   '#7b8aa8',   # significant losses (unused here but defined)
        'ink':         '#2a1f10',
        'rule_50':     '#7a6a4a',
    }

    win_pcts, cis = [], []
    for op in OPPONENTS_ORDERED:
        p = wins[op] / total[op] if total[op] else 0
        ci = 1.96 * math.sqrt(p * (1-p) / total[op]) if total[op] else 0
        win_pcts.append(100 * p)
        cis.append(100 * ci)

    # Significance: bar lower bound > 50% = significant win
    colors = []
    for p, c in zip(win_pcts, cis):
        if p - c > 50:
            colors.append(ANCIENT['gold_win'])
        elif p + c < 50:
            colors.append(ANCIENT['blue_loss'])
        else:
            colors.append(ANCIENT['tan_tie'])

    y = np.arange(len(OPPONENTS_ORDERED))
    ax.barh(y, win_pcts, xerr=cis, capsize=3,
             color=colors, edgecolor=ANCIENT['gold_edge'], linewidth=0.6,
             error_kw={'linewidth': 0.7, 'ecolor': ANCIENT['ink']})

    # Reference line at 50%
    ax.axvline(x=50, color=ANCIENT['rule_50'], linewidth=0.8, linestyle='--')

    # Value labels on bars
    for i, (p, c) in enumerate(zip(win_pcts, cis)):
        ax.text(p + c + 1, i, f'{p:.1f}%',
                 va='center', fontsize=7.5, color=ANCIENT['ink'])

    ax.set_yticks(y)
    ax.set_yticklabels(OPPONENTS_ORDERED, fontsize=8, color=ANCIENT['ink'])
    ax.invert_yaxis()
    ax.set_xlabel(f'{CHAMPION} win rate vs opponent (%)', fontsize=9,
                   color=ANCIENT['ink'])
    ax.set_xlim(30, 70)
    ax.set_xticks([30, 40, 50, 60, 70])
    ax.tick_params(axis='x', labelsize=8, colors=ANCIENT['ink'])
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(ANCIENT['ink'])
    ax.grid(axis='x', alpha=0.25, linewidth=0.5, color=ANCIENT['rule_50'])
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python fig3_headtohead.py games.csv [output.pdf]')
        sys.exit(1)
    mpl.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'fig3_headtohead.pdf'
    wins, total = load_h2h(csv_path)
    make_figure(wins, total, out_path)
