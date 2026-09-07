"""
tournament_grid.py - Parallel round-robin tournament for tab agents.

Runs every pairing in the agent registry, both sides, N games each.
Workers are pickled by name; agents are instantiated inside each
worker for clean RNG isolation.

Usage:
    # Default: full grid, 400 games per pairing, all cores
    python tournament_grid.py

    # Quick sanity: 30 games, 8 cores
    python tournament_grid.py --games 30 --workers 8

    # Specific subset: only LOOK headlines
    python tournament_grid.py --agents LOOK PSA GA-Expert GA-Original

    # Resume from a partial run
    python tournament_grid.py --games 400 --resume results/games.csv

Output:
    results/games.csv      per-game records (resumable)
    results/summary.csv    win-rate table with 95% CIs
    results/summary.txt    human-readable summary

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
(AI assistance disclosed in the manuscript acknowledgments)
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# --- Agent registry -----------------------------------------------------------
# Each entry: name -> factory (no args). Factory is called inside the worker
# so agents are fresh per game and not shared across workers.

# Board width for this run. Set by main(); workers inherit it through the
# process fork / spawn of the pool. R2.6 asks whether results survive off 4x8.
COLUMNS = 8


def _agent_factories() -> Dict[str, Callable]:
    """Lazy imports inside the function to keep top-level fast."""
    from tab_look import LookAgent
    from tab_psa import PhaseSamplingAgent
    from tab_mcts import MCTSAgent, MCTSTreeAgent

    def _td_look(use_lookahead):
        import os
        from tab_tdlook import TDLookAgent, load_net
        ck = os.environ.get('TAB_TD_CKPT', '../03_Results/checkpoints_td/best.pt')
        return TDLookAgent(load_net(ck), lookahead_seqs=3, sample_n=80,
                             opp_n_candidates=15, use_lookahead=use_lookahead)
    from tab_ai import GAAgent, RandomAgent

    return {
        # -- LOOK and ablations -----------------------------------------
        'LOOK': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=False),  # uses original FitnessEvaluator
        'LOOK-no-LA': lambda: LookAgent(
            columns=COLUMNS, sample_n=80, use_lookahead=False, use_rules=False),
        'LOOK-my-eval': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False),
        'LOOK-cont-phase': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=False, use_continuous_phase=True),
        'LOOK-no-enum': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=False, enum_threshold=0),  # always sample

        # -- Autopsy: which LookEvaluator term causes the 34% collapse? -
        # Baseline LOOK-my-eval: binary hide, discrete phase,
        # danger=10, stacking=quadratic, material=15, frozen_dist=True.
        # Each variant changes ONE thing toward FitnessEvaluator defaults.
        'LOOK-A1-material5': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            material_scale=5.0),  # FitnessEvaluator weight
        'LOOK-A2-no-frozendist': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            frozen_distinction=False),  # +3 per soldier in own home
        'LOOK-A3-simple-stack': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            stacking_mode='simple'),  # -3*cnt for ALL stacks
        'LOOK-A4-strong-danger': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            danger_scale=30.0),  # 3x stronger danger penalty
        'LOOK-A5-stacking-off': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            stacking_mode='off'),  # no stacking penalty
        # -- R3.14: the three components the original autopsy never ablated
        'LOOK-A7-no-hiding': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            use_home_hiding=False),
        'LOOK-A8-no-central': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            use_central_bonus=False),
        'LOOK-A9-no-entry': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            use_entry_pressure=False),
        'LOOK-A6-all-fixed': lambda: LookAgent(
            columns=COLUMNS, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
            use_rules=True, binary_hide=True, use_continuous_phase=False,
            material_scale=5.0, frozen_distinction=False,
            stacking_mode='simple', danger_scale=30.0),

        # -- Baselines --------------------------------------------------
        'PSA': lambda: PhaseSamplingAgent(columns=COLUMNS, n_samples=80),
        # PSA with the rescue-aware quadratic stacking penalty replaced by the
        # flat linear -3c (the TLE -> TLE-X substitution, applied to the
        # independently designed evaluator).
        'PSA-X': lambda: PhaseSamplingAgent(columns=COLUMNS, n_samples=80,
                                            stacking_mode='simple'),
        # PSA with the stacking term deleted outright (the TLE-N ablation).
        'PSA-N': lambda: PhaseSamplingAgent(columns=COLUMNS, n_samples=80,
                                            stacking_mode='off'),
        'GA-Expert': lambda: GAAgent('beginner', columns=COLUMNS, fitness_type='expert'),
        'GA-Original': lambda: GAAgent('beginner', columns=COLUMNS, fitness_type='original'),
        'GA-Fuzzy': lambda: GAAgent('beginner', columns=COLUMNS, fitness_type='fuzzy'),

        # -- Stronger baselines (slower) --------------------------------
        'GA-Expert-adv': lambda: GAAgent('advanced', columns=COLUMNS, fitness_type='expert'),
        'GA-Original-adv': lambda: GAAgent('advanced', columns=COLUMNS, fitness_type='original'),

        # -- MCTS (piece 2: flat Monte Carlo, UCB1 over allocations) -----
        # Budget = rollouts per decision; this is the compute knob for the
        # budget-matched comparison the reviewers asked for (R1.1, R3.11).
        'MCTS-200': lambda: MCTSAgent(n_simulations=200, rollout='pure'),
        'MCTS-800': lambda: MCTSAgent(n_simulations=800, rollout='pure'),
        'MCTS-3000': lambda: MCTSAgent(n_simulations=3000, rollout='pure'),
        'MCTS-200-h': lambda: MCTSAgent(n_simulations=200, rollout='heavy'),
        'MCTS-800-h': lambda: MCTSAgent(n_simulations=800, rollout='heavy'),
        'MCTS-3000-h': lambda: MCTSAgent(n_simulations=3000, rollout='heavy'),

        # -- MCTS piece 3: chance-node tree over stick throws -------------
        # Same budget knob and rollout policies as the flat agents, so the
        # tree-vs-flat difference isolates search depth (R3's question).
        'MCTST-800-h': lambda: MCTSTreeAgent(n_simulations=800, rollout='heavy',
                                               max_depth=4),
        'MCTST-800-h-d2': lambda: MCTSTreeAgent(n_simulations=800, rollout='heavy',
                                                  max_depth=2),
        'MCTST-800-h-d6': lambda: MCTSTreeAgent(n_simulations=800, rollout='heavy',
                                                  max_depth=6),
        'MCTST-3000-h': lambda: MCTSTreeAgent(n_simulations=3000, rollout='heavy',
                                                max_depth=4),

        # -- E2: budget-matched ladder (R1.1, R3.11) ---------------------
        # Four compute levels per design family. The knob differs by family,
        # so actual per-decision compute is MEASURED (instrument.py) rather
        # than assumed; the figure plots win rate against measured compute.
        'BM-TLA-1': lambda: LookAgent(columns=COLUMNS, lookahead_seqs=1, sample_n=10,
                                        opp_n_candidates=5, use_rules=False,
                                        enum_threshold=0),
        'BM-TLA-2': lambda: LookAgent(columns=COLUMNS, lookahead_seqs=3, sample_n=80,
                                        opp_n_candidates=15, use_rules=False),
        'BM-TLA-3': lambda: LookAgent(columns=COLUMNS, lookahead_seqs=5, sample_n=300,
                                        opp_n_candidates=30, use_rules=False),
        'BM-TLA-4': lambda: LookAgent(columns=COLUMNS, lookahead_seqs=10, sample_n=1000,
                                        opp_n_candidates=60, use_rules=False),

        'BM-TLAG-1': lambda: LookAgent(columns=COLUMNS, sample_n=10, use_lookahead=False,
                                         use_rules=False, enum_threshold=0),
        'BM-TLAG-2': lambda: LookAgent(columns=COLUMNS, sample_n=80, use_lookahead=False,
                                         use_rules=False),
        'BM-TLAG-3': lambda: LookAgent(columns=COLUMNS, sample_n=400, use_lookahead=False,
                                         use_rules=False),
        'BM-TLAG-4': lambda: LookAgent(columns=COLUMNS, sample_n=2000, use_lookahead=False,
                                         use_rules=False),

        'BM-GA-1': lambda: GAAgent('beginner', columns=COLUMNS, population_size=20,
                                     generations=5, fitness_type='original'),
        'BM-GA-2': lambda: GAAgent('beginner', columns=COLUMNS, population_size=50,
                                     generations=10, fitness_type='original'),
        'BM-GA-3': lambda: GAAgent('beginner', columns=COLUMNS, population_size=100,
                                     generations=25, fitness_type='original'),
        'BM-GA-4': lambda: GAAgent('beginner', columns=COLUMNS, population_size=200,
                                     generations=50, fitness_type='original'),

        # -- Learned agents (TD-lambda self-play, tab_td.py) --------------
        # TLA-NN runs the IDENTICAL search to TLA-S; only the evaluator differs
        # (learned value net vs handcrafted LookEvaluator). TD-Greedy is the
        # same network with lookahead disabled, isolating what search adds.
        'TLA-NN': lambda: _td_look(True),
        'TD-Greedy': lambda: _td_look(False),

        # -- Floor ------------------------------------------------------
        'Random': lambda: RandomAgent(),
    }


# Default agent set (skip the slowest for quick runs; user can override)
DEFAULT_AGENTS = [
    'LOOK', 'LOOK-no-LA', 'LOOK-my-eval', 'LOOK-no-enum',
    'PSA', 'GA-Expert', 'GA-Original',
]


# --- Worker function (must be at module scope for pickling) -------------------

def _play_one_game_worker(args: tuple) -> dict:
    """
    Worker: play one game between two named agents with a fixed seed.

    args = (game_id, agent_a_name, agent_b_name, seed)
    Returns the game record (also written by the parent to CSV).
    """
    game_id, a_name, b_name, seed = args[:4]
    if len(args) > 4:
        globals()['COLUMNS'] = args[4]
    from tab_runner import play_one_game

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**31 - 1))
    except ImportError:
        pass

    factories = _agent_factories()
    agent_a = factories[a_name]()
    agent_b = factories[b_name]()
    result = play_one_game(agent_a, agent_b, max_turns=300, seed=seed,
                             columns=COLUMNS)

    return {
        'game_id': game_id,
        'agent_p1': a_name,
        'agent_p2': b_name,
        'seed': seed,
        'winner': result['winner'],
        'turns': result['turns'],
        'time_sec': round(result['time_sec'], 3),
        'p1_alive': result['p1_alive'],
        'p2_alive': result['p2_alive'],
        'p1_captures': result['p1_captures'],
        'p2_captures': result['p2_captures'],
    }


# --- Pairing builder ----------------------------------------------------------

def build_pairings(agent_names: List[str], games_per_side: int,
                     seed_base: int = 42) -> List[tuple]:
    """
    Build the work queue: every unique pair x 2 sides x games_per_side games.
    Returns list of (game_id, agent_a_name, agent_b_name, seed).
    """
    pairings = []
    gid = 0
    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):
            if a == b:
                continue
            if i > j:
                continue  # avoid duplicate pair (we do both sides explicitly below)
            # Side A=P1
            for k in range(games_per_side):
                pairings.append((gid, a, b, seed_base + gid))
                gid += 1
            # Side B=P1 (the "swap")
            for k in range(games_per_side):
                pairings.append((gid, b, a, seed_base + gid))
                gid += 1
    return pairings


# --- Resume support -----------------------------------------------------------

def load_completed(csv_path: Path) -> set:
    """Return set of game_ids already written to CSV."""
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(int(row['game_id']))
    return done


# --- Summary computation ------------------------------------------------------

def compute_summary(csv_path: Path) -> List[dict]:
    """
    Read games.csv, return list of pair summaries with win rate + 95% CI.
    Each row: agent_a, agent_b, games, a_wins, b_wins, draws, w_pct, ci.
    """
    rows = defaultdict(lambda: {'a_wins': 0, 'b_wins': 0, 'draws': 0,
                                  'turns_sum': 0, 'time_sum': 0.0})

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            p1, p2 = r['agent_p1'], r['agent_p2']
            # Canonicalize: alphabetical order for the "pair key"
            a, b = sorted([p1, p2])
            key = (a, b)
            winner = int(r['winner'])
            if winner == 0:
                rows[key]['draws'] += 1
            elif winner == 1:  # P1 won
                if p1 == a:
                    rows[key]['a_wins'] += 1
                else:
                    rows[key]['b_wins'] += 1
            else:  # P2 won
                if p2 == a:
                    rows[key]['a_wins'] += 1
                else:
                    rows[key]['b_wins'] += 1
            rows[key]['turns_sum'] += int(r['turns'])
            rows[key]['time_sum'] += float(r['time_sec'])

    summary = []
    for (a, b), d in sorted(rows.items()):
        total = d['a_wins'] + d['b_wins'] + d['draws']
        if total == 0:
            continue
        p = d['a_wins'] / total
        ci_95 = 1.96 * math.sqrt(p * (1 - p) / total) if 0 < p < 1 else 0.0
        summary.append({
            'agent_a': a,
            'agent_b': b,
            'games': total,
            'a_wins': d['a_wins'],
            'b_wins': d['b_wins'],
            'draws': d['draws'],
            'a_win_pct': round(100 * p, 1),
            'ci_95_pct': round(100 * ci_95, 1),
            'avg_turns': round(d['turns_sum'] / total, 1),
            'avg_time_sec': round(d['time_sum'] / total, 3),
        })
    return summary


def write_summary(summary: List[dict], txt_path: Path, csv_path: Path):
    # CSV
    if summary:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    # Human-readable
    lines = []
    lines.append('=' * 90)
    lines.append('  tab TOURNAMENT SUMMARY')
    lines.append('=' * 90)
    lines.append(f'  Generated: {datetime.now().isoformat()}')
    lines.append('')
    lines.append(f'  {"Agent A":<20} {"Agent B":<20} {"Games":>6} '
                  f'{"A wins":>6} {"B wins":>6} {"Draws":>5} {"A win %":>9} '
                  f'{"95% CI":>9} {"Turns":>6} {"Time/g":>7}')
    lines.append(f'  {"-"*20} {"-"*20} {"-"*6} {"-"*6} {"-"*6} {"-"*5} '
                  f'{"-"*9} {"-"*9} {"-"*6} {"-"*7}')
    for r in summary:
        lines.append(f'  {r["agent_a"]:<20} {r["agent_b"]:<20} '
                      f'{r["games"]:>6} {r["a_wins"]:>6} {r["b_wins"]:>6} '
                      f'{r["draws"]:>5} {r["a_win_pct"]:>8.1f}% '
                      f'{"+/-"+str(r["ci_95_pct"]):>9} '
                      f'{r["avg_turns"]:>6.1f} {r["avg_time_sec"]:>6.2f}s')
    lines.append('')
    text = '\n'.join(lines)
    with open(txt_path, 'w') as f:
        f.write(text + '\n')
    print(text)


# --- Main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='tab agent tournament grid.')
    p.add_argument('--games', type=int, default=400,
                    help='Games per side per pairing (so total = 2N). Default 400.')
    p.add_argument('--workers', type=int, default=None,
                    help='Worker processes. Default = cpu_count().')
    p.add_argument('--agents', nargs='+', default=None,
                    help='Agent names to include. Default = built-in list.')
    p.add_argument('--list-agents', action='store_true',
                    help='Print all available agents and exit.')
    p.add_argument('--out', type=str, default='results',
                    help='Output directory. Default ./results')
    p.add_argument('--columns', type=int, default=8,
                    help='board width (4 x COLUMNS). R2.6 board-size sweep.')
    p.add_argument('--seed', type=int, default=42,
                    help='Seed base. Default 42.')
    p.add_argument('--resume', action='store_true',
                    help='Skip games already in games.csv.')
    p.add_argument('--checkpoint-every', type=int, default=200,
                    help='Flush CSV every N games. Default 200.')
    args = p.parse_args()

    factories = _agent_factories()
    if args.list_agents:
        print('Available agents:')
        for name in factories.keys():
            print(f'  {name}')
        return

    agent_names = args.agents if args.agents else DEFAULT_AGENTS
    unknown = [a for a in agent_names if a not in factories]
    if unknown:
        print(f'ERROR: unknown agents: {unknown}', file=sys.stderr)
        print(f'Available: {list(factories.keys())}', file=sys.stderr)
        sys.exit(1)

    workers = args.workers or mp.cpu_count()
    globals()['COLUMNS'] = args.columns
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_csv = out_dir / 'games.csv'
    summary_csv = out_dir / 'summary.csv'
    summary_txt = out_dir / 'summary.txt'

    # Build pairings
    pairings = build_pairings(agent_names, args.games, seed_base=args.seed)
    total_games = len(pairings)
    n_pairings = len(agent_names) * (len(agent_names) - 1) // 2
    print(f'Agents: {agent_names}')
    print(f'Pairings: {n_pairings} unique pairs x 2 sides x {args.games} games '
           f'= {total_games} total games')
    print(f'Workers: {workers}')
    print(f'Output:  {out_dir.resolve()}/')

    # Resume support
    completed = load_completed(games_csv) if args.resume else set()
    pending = [tuple(pa) + (args.columns,) for pa in pairings
                 if pa[0] not in completed]
    if completed:
        print(f'Resuming: {len(completed)} games already done, '
               f'{len(pending)} remaining')

    # Set up CSV (append if resuming, fresh otherwise)
    file_exists = games_csv.exists() and args.resume
    csv_handle = open(games_csv, 'a' if file_exists else 'w', newline='')
    fieldnames = ['game_id', 'agent_p1', 'agent_p2', 'seed', 'winner',
                   'turns', 'time_sec', 'p1_alive', 'p2_alive',
                   'p1_captures', 'p2_captures']
    writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        csv_handle.flush()

    # Run in parallel
    t_start = time.time()
    done = len(completed)
    last_flush = 0
    print(f'\nStarting tournament at {datetime.now().isoformat()}')
    print('Progress: ', end='', flush=True)

    try:
        with mp.Pool(processes=workers) as pool:
            for record in pool.imap_unordered(_play_one_game_worker, pending,
                                                chunksize=4):
                writer.writerow(record)
                done += 1
                if done - last_flush >= args.checkpoint_every:
                    csv_handle.flush()
                    last_flush = done
                    elapsed = time.time() - t_start
                    rate = (done - len(completed)) / elapsed if elapsed > 0 else 0
                    eta = (total_games - done) / rate if rate > 0 else 0
                    print(f'\n[{done}/{total_games}] '
                           f'rate={rate:.1f} g/s  eta={eta/60:.1f} min',
                           end='', flush=True)
                else:
                    if done % 20 == 0:
                        print('.', end='', flush=True)
    except KeyboardInterrupt:
        print('\n[INTERRUPTED - partial results saved]')
    finally:
        csv_handle.close()

    total_time = time.time() - t_start
    print(f'\n\nDone. {done} games total. Wall time: {total_time/60:.1f} min')
    print(f'Throughput: {done/total_time:.1f} games/sec')

    # Compute & write summary
    print(f'\nComputing summary from {games_csv} ...')
    summary = compute_summary(games_csv)
    write_summary(summary, summary_txt, summary_csv)
    print(f'\nFiles written:')
    print(f'  {games_csv}')
    print(f'  {summary_csv}')
    print(f'  {summary_txt}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # safer cross-platform
    main()
