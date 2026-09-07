"""
fig1_winrate_matrix.py -- Win-rate heatmap from the main tournament.

Input:  games.csv from tournament_grid.py (one row per game)
Output: fig1_winrate_matrix.pdf (double-column IEEE figure)

Cell (i, j) shows agent i's win rate against agent j across all games where
they faced each other (both sides combined). Diagonal is greyed out.
Sample size is 1000 games per pairing.

Usage:
    python fig1_winrate_matrix.py results_a3_full/games.csv

Or with custom output:
    python fig1_winrate_matrix.py results_a3_full/games.csv my_fig.pdf
"""

import sys
import csv
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Display name remapping (debug labels -> paper labels) ------------------
RENAME = {
    'LOOK':                 'TLA',
    'LOOK-A3-simple-stack': 'TLA-S',
    'LOOK-no-LA':           'TLA-G',
    'LOOK-no-enum':         'TLA-B',
    'GA-Expert':            'GA-Expert',
    'GA-Original':          'GA-Orig.',  # tighter for axis labels
    'GA-Fuzzy':             'GA-Fuzzy',
    'PSA':                  'PSA',
}

# Display order — by marginal win rate in Table II (eight-agent roster)
ORDER = ['TLA-S', 'TLA-B', 'TLA', 'GA-Orig.', 'GA-Expert', 'TLA-G',
         'GA-Fuzzy', 'PSA']


def load_matrix(csv_path):
    """Build win-rate matrix and pairwise counts from games.csv."""
    wins = defaultdict(lambda: defaultdict(int))   # wins[A][B] = times A beat B
    games = defaultdict(lambda: defaultdict(int))  # symmetric games[A][B]

    with open(csv_path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            a = RENAME.get(row['agent_p1'], row['agent_p1'])
            b = RENAME.get(row['agent_p2'], row['agent_p2'])
            if a not in ORDER or b not in ORDER:
                continue
            winner = int(row['winner'])
            games[a][b] += 1
            games[b][a] += 1
            if winner == 1:
                wins[a][b] += 1
            elif winner == -1:
                wins[b][a] += 1
            # Draws (winner==0) increment games but no wins

    n = len(ORDER)
    matrix = np.full((n, n), np.nan)
    for i, ai in enumerate(ORDER):
        for j, aj in enumerate(ORDER):
            if i == j:
                continue
            g = games[ai][aj]
            if g == 0:
                continue
            matrix[i, j] = 100.0 * wins[ai][aj] / g
    return matrix


# ---- Ancient palette derived from the board image ---------------------------
# Gold/amber = P1's color (row agent wins). Blue/lavender = P2's color
# (column agent wins, i.e., row agent loses). Cream/parchment at 50%.
# These are the same hue families used in the Figure 1 board illustration.
ANCIENT = {
    'gold_dark':     '#8a5a1c',  # deep amber (strong row win)
    'gold_mid':      '#c89345',  # P1 soldier color
    'gold_light':    '#e8c885',  # pale gold
    'cream':         '#f0e6d2',  # parchment (50%, draw)
    'blue_light':    '#aab5c9',  # pale dusty blue
    'blue_mid':      '#7b8aa8',  # P2 soldier color
    'blue_dark':     '#3f4d6a',  # deep slate-blue (strong row loss)
    'diagonal_grey': '#c9bfa9',  # warm sandy grey for self-vs-self cells
    'grid_line':     '#f0e6d2',  # cream grid lines between cells
    'text_on_dark':  '#f5edd8',  # cream text on dark cells
    'text_on_light': '#2a1f10',  # dark brown text on light cells
}


def render(matrix, out_path):
    """Render heatmap as double-column IEEE figure (7.1 in wide)."""
    n = len(ORDER)
    fig, ax = plt.subplots(figsize=(7.1, 3.6))

    # Custom diverging colormap: deep blue -> pale blue -> cream -> pale gold -> deep gold
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'ancient_diverge',
        [ANCIENT['blue_dark'], ANCIENT['blue_mid'], ANCIENT['blue_light'],
         ANCIENT['cream'],
         ANCIENT['gold_light'], ANCIENT['gold_mid'], ANCIENT['gold_dark']],
        N=256)
    cmap.set_bad(color=ANCIENT['diagonal_grey'])

    # Plot
    im = ax.imshow(matrix, cmap=cmap, vmin=30, vmax=70,
                    aspect='auto', interpolation='nearest')

    # Cell labels
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '-', ha='center', va='center',
                         color='#7a6a4a', fontsize=11)
                continue
            v = matrix[i, j]
            if np.isnan(v):
                continue
            # Cream text on strong-color cells, dark-brown text on mid cells
            if v < 38 or v > 62:
                txt_color = ANCIENT['text_on_dark']
            else:
                txt_color = ANCIENT['text_on_light']
            ax.text(j, i, f'{v:.1f}',
                     ha='center', va='center', color=txt_color, fontsize=9)

    # Ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ORDER, rotation=35, ha='right', fontsize=9,
                        color=ANCIENT['text_on_light'])
    ax.set_yticklabels(ORDER, fontsize=9, color=ANCIENT['text_on_light'])
    ax.set_xlabel('Opponent (column)', fontsize=10,
                   color=ANCIENT['text_on_light'])
    ax.set_ylabel('Agent (row)', fontsize=10,
                   color=ANCIENT['text_on_light'])
    ax.tick_params(axis='both', which='both', length=0)

    # Move x ticks to top for matrix-style reading
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Row agent win rate (%) vs column agent',
                    fontsize=9, labelpad=8,
                    color=ANCIENT['text_on_light'])
    cbar.ax.tick_params(labelsize=8, colors=ANCIENT['text_on_light'])
    cbar.ax.axhline(y=50, color=ANCIENT['text_on_light'], linewidth=0.8)

    # Subtle grid between cells (cream-on-color, lighter than before)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color=ANCIENT['grid_line'], linewidth=1.0,
             alpha=0.9)
    ax.tick_params(which='minor', length=0)

    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python fig1_winrate_matrix.py games.csv [output.pdf]')
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'fig1_winrate_matrix.pdf'

    # IEEE-friendly default rcParams
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'pdf.fonttype': 42,  # embed as TrueType for IEEE pdf checker
        'ps.fonttype': 42,
    })

    matrix = load_matrix(csv_path)
    render(matrix, out_path)
