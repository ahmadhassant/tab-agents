"""
analyze_results.py - Post-tournament analysis.

Reads results/games.csv and produces:
  - A win-rate matrix (every agent vs every other)
  - Side-effect analysis (P1 advantage / P2 advantage)
  - Statistical significance flags (95% CI excludes 50%)
  - Per-agent average game length and captures

Usage:
    python analyze_results.py [--in results/games.csv] [--out results/]
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def load_games(csv_path):
    games = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            games.append({
                'game_id': int(r['game_id']),
                'agent_p1': r['agent_p1'],
                'agent_p2': r['agent_p2'],
                'winner': int(r['winner']),
                'turns': int(r['turns']),
                'time_sec': float(r['time_sec']),
                'p1_alive': int(r['p1_alive']),
                'p2_alive': int(r['p2_alive']),
                'p1_captures': int(r['p1_captures']),
                'p2_captures': int(r['p2_captures']),
            })
    return games


def win_rate_matrix(games):
    """For each agent A vs agent B, compute A's win rate (averaged over sides)."""
    matrix = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'games': 0}))
    agents = set()
    for g in games:
        a, b = g['agent_p1'], g['agent_p2']
        agents.add(a); agents.add(b)
        # Symmetric games counter (both directions of the pair)
        matrix[a][b]['games'] += 1
        matrix[b][a]['games'] += 1
        # Wins: A as P1 means winner==1; B as P2 means winner==-1
        if g['winner'] == 1:
            matrix[a][b]['wins'] += 1
        elif g['winner'] == -1:
            matrix[b][a]['wins'] += 1
        # draws: increment neither
    return matrix, sorted(agents)


def side_effect_per_agent(games):
    """Per agent: win rate when playing P1 vs P2."""
    stats = defaultdict(lambda: {
        'p1_games': 0, 'p1_wins': 0,
        'p2_games': 0, 'p2_wins': 0,
    })
    for g in games:
        a, b = g['agent_p1'], g['agent_p2']
        stats[a]['p1_games'] += 1
        stats[b]['p2_games'] += 1
        if g['winner'] == 1:
            stats[a]['p1_wins'] += 1
        elif g['winner'] == -1:
            stats[b]['p2_wins'] += 1
    return stats


def overall_per_agent(games):
    """Per agent: total games, wins, average turns, avg captures."""
    stats = defaultdict(lambda: {
        'games': 0, 'wins': 0, 'turns_sum': 0,
        'captures_for': 0, 'captures_against': 0,
    })
    for g in games:
        a, b = g['agent_p1'], g['agent_p2']
        stats[a]['games'] += 1
        stats[b]['games'] += 1
        stats[a]['turns_sum'] += g['turns']
        stats[b]['turns_sum'] += g['turns']
        stats[a]['captures_for'] += g['p1_captures']
        stats[a]['captures_against'] += g['p2_captures']
        stats[b]['captures_for'] += g['p2_captures']
        stats[b]['captures_against'] += g['p1_captures']
        if g['winner'] == 1:
            stats[a]['wins'] += 1
        elif g['winner'] == -1:
            stats[b]['wins'] += 1
    return stats


def ci_95(wins, total):
    if total == 0:
        return 0.0
    p = wins / total
    if p <= 0 or p >= 1:
        return 0.0
    return 1.96 * math.sqrt(p * (1 - p) / total)


def fmt_pct_with_ci(wins, total):
    if total == 0:
        return '  -  '
    p = wins / total
    ci = ci_95(wins, total)
    return f'{100*p:4.1f}+/-{100*ci:3.1f}'


def print_matrix(matrix, agents):
    print('\n' + '=' * 80)
    print('  WIN-RATE MATRIX (row vs column, % +/- 95% CI)')
    print('=' * 80)
    # Column headers - truncated names
    short = {a: a if len(a) <= 11 else a[:11] for a in agents}
    print(f'  {"":<14}' + ''.join(f'{short[a]:>14s}' for a in agents))
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append('     -     ')
            else:
                d = matrix[a][b]
                cells.append(fmt_pct_with_ci(d['wins'], d['games']))
        print(f'  {short[a]:<14s}' + ''.join(f'{c:>14}' for c in cells))


def print_significance(matrix, agents):
    print('\n' + '=' * 80)
    print('  STATISTICAL SIGNIFICANCE (95% CI lower bound > 50% = A wins)')
    print('=' * 80)
    findings = []
    for a in agents:
        for b in agents:
            if a >= b:
                continue
            d = matrix[a][b]
            if d['games'] < 30:
                continue
            p = d['wins'] / d['games']
            ci = ci_95(d['wins'], d['games'])
            lower = p - ci
            upper = p + ci
            if lower > 0.5:
                findings.append((p, a, b, d['wins'], d['games'], ci,
                                 f'{a} > {b}'))
            elif upper < 0.5:
                findings.append((1-p, b, a, d['games']-d['wins'], d['games'],
                                 ci, f'{b} > {a}'))
    findings.sort(reverse=True)
    if not findings:
        print('  (no pairings with significant differences at 95%)')
    for p, a, b, wins, games, ci, label in findings:
        print(f'  {label:<35s}  {wins}/{games}  '
              f'({100*p:.1f}% +/-{100*ci:.1f}%)')


def print_side_effects(side_stats, agents):
    print('\n' + '=' * 80)
    print('  SIDE EFFECTS (does the agent perform better as P1 or P2?)')
    print('=' * 80)
    print(f'  {"Agent":<16} {"P1 games":>8} {"P1 win%":>8} {"P2 games":>8} '
           f'{"P2 win%":>8} {"P1-P2":>8}')
    print(f'  {"-"*16} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for a in agents:
        s = side_stats[a]
        p1_pct = 100 * s['p1_wins'] / s['p1_games'] if s['p1_games'] else 0
        p2_pct = 100 * s['p2_wins'] / s['p2_games'] if s['p2_games'] else 0
        diff = p1_pct - p2_pct
        print(f'  {a:<16} {s["p1_games"]:>8d} {p1_pct:>7.1f}% '
               f'{s["p2_games"]:>8d} {p2_pct:>7.1f}% {diff:>+7.1f}%')


def print_overall(overall, agents):
    print('\n' + '=' * 80)
    print('  OVERALL PER-AGENT STATS')
    print('=' * 80)
    print(f'  {"Agent":<16} {"Games":>6} {"Wins":>5} {"Win%":>7} '
           f'{"Turns/g":>8} {"Cap+":>6} {"Cap-":>6}')
    print(f'  {"-"*16} {"-"*6} {"-"*5} {"-"*7} {"-"*8} {"-"*6} {"-"*6}')
    # Sort by win rate
    sorted_agents = sorted(agents, key=lambda a:
                             overall[a]['wins'] / overall[a]['games']
                             if overall[a]['games'] else 0, reverse=True)
    for a in sorted_agents:
        s = overall[a]
        win_pct = 100 * s['wins'] / s['games'] if s['games'] else 0
        turns = s['turns_sum'] / s['games'] if s['games'] else 0
        cap_for = s['captures_for'] / s['games'] if s['games'] else 0
        cap_against = s['captures_against'] / s['games'] if s['games'] else 0
        print(f'  {a:<16} {s["games"]:>6d} {s["wins"]:>5d} {win_pct:>6.1f}% '
               f'{turns:>7.1f} {cap_for:>5.1f} {cap_against:>5.1f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='input', default='results/games.csv',
                    help='Input CSV from tournament_grid.py')
    p.add_argument('--out', default='results',
                    help='Output directory for analysis files')
    args = p.parse_args()

    games_path = Path(args.input)
    if not games_path.exists():
        print(f'ERROR: {games_path} not found')
        return

    games = load_games(games_path)
    print(f'Loaded {len(games)} games from {games_path}')

    matrix, agents = win_rate_matrix(games)
    side_stats = side_effect_per_agent(games)
    overall = overall_per_agent(games)

    print_overall(overall, agents)
    print_matrix(matrix, agents)
    print_significance(matrix, agents)
    print_side_effects(side_stats, agents)


if __name__ == '__main__':
    main()
