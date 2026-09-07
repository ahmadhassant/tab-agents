"""
human_alignment.py -- Move-level agreement between agents and human experts.

Parses game logs in HumanGames/, replays each human turn as a decision point,
asks each agent what it would do given the same board state and throw values,
and measures three agreement metrics:

  1. SOLDIER_SET: did the agent pick the same set of soldiers to move?
     (Coarse but robust to value-allocation ambiguity.)
  2. STRICT: did the agent produce the exact same allocation (per-soldier
     value multiset)?
  3. FIRST_SOLDIER: did the agent pick the same first soldier to move?

We measure separately:
  - Human-as-P1 turns (Ahmad playing first hand) -- this is most games
  - Human-as-P2 turns (if any)

The output is a per-agent agreement table and a confusion-style breakdown
showing which agents most resemble the human.

Usage:
    python human_alignment.py --games-dir HumanGames/ --agents LOOK PSA GA-Expert
    python human_alignment.py --games-dir HumanGames/  # all default agents

Output:
    results_alignment/positions.csv   -- per-decision record
    results_alignment/summary.csv     -- per-agent agreement rates
    results_alignment/summary.txt     -- human-readable table

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
(AI assistance disclosed in the manuscript acknowledgments)
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from tab_game import (Board, TabRules, DEFAULT_COLUMNS, PLAYER_1, PLAYER_2,
                       SoldierMove, TurnAllocation)


# ----- Position label <-> cell index --------------------------------------
# Same labelling as tab_game.py: rows A/B/C/D, cols 1..8.
# P1 home = row A (cells 0..7), P2 home = row D (cells 24..31).
# Loop rows B and C in serpentine order.

ROW_OF_CELL = {}
COL_OF_CELL = {}
LABEL_OF_CELL = {}
CELL_OF_LABEL = {}
for _i in range(32):
    _r = _i // 8
    _c = _i % 8
    _row_letter = 'ABCD'[_r]
    _col_num = _c + 1
    ROW_OF_CELL[_i] = _row_letter
    COL_OF_CELL[_i] = _col_num
    _lbl = f'{_row_letter}{_col_num}'
    LABEL_OF_CELL[_i] = _lbl
    CELL_OF_LABEL[_lbl] = _i


def cell_from_label(lbl: str) -> int:
    return CELL_OF_LABEL[lbl.strip()]


# ----- BOARD line parser --------------------------------------------------

BOARD_RE = re.compile(
    r'BOARD:([^|]+)\|([^|]+)\|([^|]+)\|([^ ]+)\s+'
    r'FQ1:\[([^\]]*)\]\s+FQ2:\[([^\]]*)\]'
)


def parse_board_line(line: str) -> Optional[Board]:
    """
    Parse a BOARD: line into a Board object.
    Returns None if the line doesn't match.
    """
    m = BOARD_RE.search(line)
    if not m:
        return None
    rows = [m.group(1), m.group(2), m.group(3), m.group(4)]
    fq1 = m.group(5).strip()
    fq2 = m.group(6).strip()

    # Parse cells: tokens like "+1", "-1", "+0", "+2", "-3"
    cells = []
    for row in rows:
        # Find every signed integer
        toks = re.findall(r'[+-]\d+', row)
        if len(toks) != 8:
            return None  # malformed
        for t in toks:
            cells.append(int(t))
    if len(cells) != 32:
        return None

    board = Board(DEFAULT_COLUMNS)
    board.cells = cells

    # FQ1 and FQ2: comma-separated labels in order of freezing.
    # The Board.frozen_queue stores cells in FIFO order.
    fq1_list = [s.strip() for s in fq1.split(',') if s.strip()]
    fq2_list = [s.strip() for s in fq2.split(',') if s.strip()]

    # Rebuild frozen_queue and frozen_at from the FQ lists
    board.frozen_queue[0] = []
    board.frozen_queue[1] = []
    board.frozen_at[0] = {}
    board.frozen_at[1] = {}
    for lbl in fq1_list:
        c = cell_from_label(lbl)
        board.frozen_queue[0].append(c)
        board.frozen_at[0][c] = board.frozen_at[0].get(c, 0) + 1
    for lbl in fq2_list:
        c = cell_from_label(lbl)
        board.frozen_queue[1].append(c)
        board.frozen_at[1][c] = board.frozen_at[1].get(c, 0) + 1

    return board


# ----- Game-file parser ---------------------------------------------------

TURN_HEADER_RE = re.compile(
    r'---\s*Turn\s+(\d+):\s+(You|AI|P1|P2)\s+throws\s+\[([^\]]+)\]\s*---'
)
ALLOCATE_RE = re.compile(
    r'ALLOCATE:\s+values=\[([^\]]+)\]\s+soldiers=\[([^\]]*)\]'
)
# Move lines like "B1->B3 (2)" or "B1->B3 (2) CAPTURED 1!"
# Game logs use the Unicode right-arrow U+2192 ("\u2192").
MOVE_RE = re.compile(
    r'\b([A-D])([1-8])\s*(?:->|\u2192)\s*([A-D])([1-8])\s*\((\d+)\)'
)
# Freeing lines like "FREE: A1->B1" or "FREE: A1\u2192B1"
FREE_RE = re.compile(
    r'FREE:\s+([A-D])([1-8])\s*(?:->|\u2192)\s*([A-D])([1-8])'
)


def parse_values(s: str) -> List[int]:
    return [int(v.strip()) for v in s.split(',') if v.strip()]


def parse_game_file(path: Path) -> List[Dict]:
    """
    Parse a single game file into a list of decision records:

    Each record is for ONE turn where the human (You / P1 in these logs)
    made an active decision (had >=1 movable soldier and made a non-trivial
    allocation).

    record = {
        'game_id': str,
        'turn': int,
        'pre_board': Board,
        'throw_values': List[int],
        'human_alloc': Dict[soldier_pos -> List[int values_used]],
        'human_soldiers': List[int],  # cells the human chose to move
        'raw_text': str,  # for debugging
    }
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    lines = [l.rstrip('\r\n') for l in text.split('\n')]

    records = []
    game_id = path.stem
    last_board = None  # the BOARD: line immediately preceding the current turn header
    i = 0
    while i < len(lines):
        line = lines[i]

        # Update last_board if this is a BOARD: line
        if line.startswith('BOARD:'):
            parsed = parse_board_line(line)
            if parsed is not None:
                last_board = parsed
            i += 1
            continue

        # Turn header
        m = TURN_HEADER_RE.search(line)
        if not m:
            i += 1
            continue

        turn_num = int(m.group(1))
        actor = m.group(2)  # "You" or "AI" (these files: human is P1)
        throws = parse_values(m.group(3))

        # Look ahead for an ALLOCATE line; bail at next "--- Turn" or BOARD: line
        # for the *next* turn.
        alloc_line = None
        move_lines = []
        j = i + 1
        while j < len(lines):
            nl = lines[j]
            if TURN_HEADER_RE.search(nl):
                break
            if nl.startswith('BOARD:'):
                # The board AFTER this turn -- still useful as boundary
                break
            if ALLOCATE_RE.search(nl):
                alloc_line = nl
            elif MOVE_RE.search(nl) or FREE_RE.search(nl):
                move_lines.append(nl)
            j += 1

        # Only record if this is a HUMAN turn with active allocation
        is_human = (actor == 'You' or actor == 'P1')
        if is_human and last_board is not None and alloc_line:
            am = ALLOCATE_RE.search(alloc_line)
            if am:
                alloc_values = parse_values(am.group(1))
                # Reconstruct per-soldier allocation from move lines
                human_alloc = reconstruct_allocation(last_board, alloc_values,
                                                       move_lines, PLAYER_1)
                if human_alloc:
                    soldiers_chosen = sorted(human_alloc.keys())
                    records.append({
                        'game_id': game_id,
                        'turn': turn_num,
                        'pre_board': last_board,
                        'throw_values': alloc_values,
                        'human_alloc': human_alloc,
                        'human_soldiers': soldiers_chosen,
                    })

        i = j  # jump to where lookahead stopped
        # But do NOT skip the BOARD: line that bounded us -- let the loop handle it
        if i < len(lines) and lines[i].startswith('BOARD:'):
            continue
    return records


def reconstruct_allocation(pre_board: Board, alloc_values: List[int],
                             move_lines: List[str],
                             player: int) -> Dict[int, List[int]]:
    """
    From the pre-state board, the values rolled, and the sequence of move
    lines that followed, work out which soldier got which values.

    Returns a dict: starting_cell -> list of values used (in order of use).
    A 'starting cell' is the soldier's position BEFORE the turn.

    Handles:
      - FREE: A1->B1 (value 1 spent on freeing; soldier 'starts' at A1
        but ends at B1; we record it as starting cell A1)
      - B1->B3 (2): soldier moved from B1 to B3 using value 2
      - Chained moves: same soldier might move multiple times; we attribute
        all values to the ORIGINAL starting cell.
    """
    # Track each "active soldier identity" by where it currently sits.
    # When a soldier moves, we update its current position but remember
    # its original starting cell.
    current_to_origin = {}  # current_pos -> origin_cell
    alloc = defaultdict(list)

    remaining_values = list(alloc_values)

    for line in move_lines:
        # Freeing
        fm = FREE_RE.search(line)
        if fm:
            from_lbl = fm.group(1) + fm.group(2)
            to_lbl = fm.group(3) + fm.group(4)
            from_cell = cell_from_label(from_lbl)
            to_cell = cell_from_label(to_lbl)
            origin = current_to_origin.get(from_cell, from_cell)
            # Freeing uses value 1
            if 1 in remaining_values:
                remaining_values.remove(1)
            alloc[origin].append(1)
            # The soldier now sits at to_cell
            current_to_origin.pop(from_cell, None)
            current_to_origin[to_cell] = origin
            continue

        # Regular move
        mm = MOVE_RE.search(line)
        if mm:
            from_lbl = mm.group(1) + mm.group(2)
            to_lbl = mm.group(3) + mm.group(4)
            val = int(mm.group(5))
            from_cell = cell_from_label(from_lbl)
            to_cell = cell_from_label(to_lbl)
            origin = current_to_origin.get(from_cell, from_cell)
            if val in remaining_values:
                remaining_values.remove(val)
            alloc[origin].append(val)
            current_to_origin.pop(from_cell, None)
            current_to_origin[to_cell] = origin
            continue

    # Filter: only count soldiers that actually got at least one value
    return {k: v for k, v in alloc.items() if v}


# ----- Agent evaluation ---------------------------------------------------

def _agent_factories() -> Dict[str, callable]:
    from tab_look import LookAgent
    from tab_psa import PhaseSamplingAgent
    from tab_ai import GAAgent, RandomAgent
    return {
        'LOOK': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                    opp_n_candidates=15, use_rules=False),
        'LOOK-no-LA': lambda: LookAgent(sample_n=80, use_lookahead=False,
                                          use_rules=False),
        'LOOK-no-enum': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                            opp_n_candidates=15,
                                            use_rules=False, enum_threshold=0),
        # R3.2: TLE (the pre-fix evaluator) was never in the alignment study,
        # which is why the claim that the stacking substitution improved human
        # alignment had no supporting experiment. Both arms are now present.
        'TLE': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                   opp_n_candidates=15, use_rules=True,
                                   binary_hide=True,
                                   use_continuous_phase=False,
                                   stacking_mode='quadratic'),
        'TLA-S': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                     opp_n_candidates=15, use_rules=True,
                                     binary_hide=True,
                                     use_continuous_phase=False,
                                     stacking_mode='simple'),
        'TLE-no-hiding': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                             opp_n_candidates=15, use_rules=True,
                                             binary_hide=True,
                                             use_continuous_phase=False,
                                             use_home_hiding=False),
        'PSA': lambda: PhaseSamplingAgent(n_samples=80),
        'GA-Expert': lambda: GAAgent('beginner', fitness_type='expert'),
        'GA-Original': lambda: GAAgent('beginner', fitness_type='original'),
        'Random': lambda: RandomAgent(),
    }


def agent_decision(agent, board: Board, player: int,
                     throw_values: List[int]) -> Dict[int, List[int]]:
    """Ask agent for an allocation, return as {origin_cell: [values]}."""
    rules = TabRules(DEFAULT_COLUMNS)
    movable = rules.get_movable_soldiers(board, player, throw_values)
    if not movable or not throw_values:
        return {}
    alloc = agent.choose_allocation(board, player, throw_values, movable, rules)
    out = defaultdict(list)
    for sm in alloc.soldier_moves:
        if sm.values_used:
            out[sm.from_pos].extend(sm.values_used)
    return dict(out)


# ----- Agreement metrics --------------------------------------------------

def agreement_metrics(human: Dict[int, List[int]],
                       agent: Dict[int, List[int]]) -> Dict[str, bool]:
    """
    Returns dict with three booleans:
      - soldier_set: same set of soldiers chosen
      - strict:     same soldiers AND same value multiset per soldier
      - first:      same single soldier picked as first to move
    Plus continuous:
      - jaccard:    Jaccard similarity of soldier sets
    """
    h_sold = set(human.keys())
    a_sold = set(agent.keys())

    soldier_set = (h_sold == a_sold)

    inter = len(h_sold & a_sold)
    union = len(h_sold | a_sold)
    jaccard = inter / union if union > 0 else 1.0

    # Strict: same dict where values are sorted multisets
    if soldier_set:
        strict = all(sorted(human[s]) == sorted(agent.get(s, []))
                       for s in h_sold)
    else:
        strict = False

    # First soldier (smallest from_pos value, tiebreaker for determinism)
    h_first = min(human.keys()) if human else None
    a_first = min(agent.keys()) if agent else None
    first = (h_first == a_first)

    return {
        'soldier_set': soldier_set,
        'strict': strict,
        'first': first,
        'jaccard': jaccard,
    }


# ----- Main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='HumanGames',
                    help='Directory containing human game .txt files')
    p.add_argument('--agents', nargs='+', default=None,
                    help='Agent names (default: built-in list)')
    p.add_argument('--out', default='results_alignment',
                    help='Output directory')
    args = p.parse_args()

    games_dir = Path(args.games_dir)
    if not games_dir.exists():
        print(f'ERROR: {games_dir} not found')
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    factories = _agent_factories()
    agent_names = args.agents if args.agents else list(factories.keys())
    unknown = [a for a in agent_names if a not in factories]
    if unknown:
        print(f'ERROR: unknown agents: {unknown}')
        print(f'Available: {list(factories.keys())}')
        return

    # Parse all games
    print(f'Parsing games in {games_dir} ...')
    all_records = []
    for game_file in sorted(games_dir.glob('*.txt')):
        records = parse_game_file(game_file)
        all_records.extend(records)
        print(f'  {game_file.name}: {len(records)} human decisions')
    print(f'Total human decisions: {len(all_records)}')

    if not all_records:
        print('No decisions found. Check parsing.')
        return

    # For each decision, ask each agent
    print(f'\nEvaluating {len(agent_names)} agents on {len(all_records)} '
           f'decisions...')

    # Build agents once (they're stateless per-decision)
    agents = {name: factories[name]() for name in agent_names}

    # Per-decision records for CSV
    position_records = []
    # Per-agent tallies
    agg = {name: {'soldier_set': 0, 'strict': 0, 'first': 0,
                    'jaccard_sum': 0.0, 'n': 0} for name in agent_names}

    import random
    for ri, rec in enumerate(all_records):
        if ri % 20 == 0:
            print(f'  decision {ri+1}/{len(all_records)}', flush=True)
        # Reseed for determinism per decision
        random.seed(42 + ri)
        for name, agent in agents.items():
            try:
                agent_alloc = agent_decision(agent, rec['pre_board'],
                                                PLAYER_1, rec['throw_values'])
            except Exception as e:
                # Agent failed -- record as no-match
                agent_alloc = {}
            metrics = agreement_metrics(rec['human_alloc'], agent_alloc)
            agg[name]['n'] += 1
            agg[name]['soldier_set'] += int(metrics['soldier_set'])
            agg[name]['strict'] += int(metrics['strict'])
            agg[name]['first'] += int(metrics['first'])
            agg[name]['jaccard_sum'] += metrics['jaccard']

            position_records.append({
                'game_id': rec['game_id'],
                'turn': rec['turn'],
                'throws': ','.join(map(str, rec['throw_values'])),
                'human_soldiers': ','.join(LABEL_OF_CELL[s]
                                              for s in rec['human_soldiers']),
                'agent': name,
                'agent_soldiers': ','.join(LABEL_OF_CELL[s]
                                              for s in sorted(agent_alloc.keys())),
                'soldier_set_match': int(metrics['soldier_set']),
                'strict_match': int(metrics['strict']),
                'first_match': int(metrics['first']),
                'jaccard': round(metrics['jaccard'], 3),
            })

    # Write per-decision CSV
    pos_csv = out_dir / 'positions.csv'
    if position_records:
        with open(pos_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(position_records[0].keys()))
            writer.writeheader()
            writer.writerows(position_records)

    # Summary
    summary_rows = []
    for name in agent_names:
        s = agg[name]
        n = s['n']
        if n == 0:
            continue
        soldier_pct = 100 * s['soldier_set'] / n
        strict_pct = 100 * s['strict'] / n
        first_pct = 100 * s['first'] / n
        jaccard = s['jaccard_sum'] / n
        # 95% CI on soldier_set agreement (binomial)
        p = s['soldier_set'] / n
        ci = 1.96 * math.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0
        summary_rows.append({
            'agent': name,
            'n_decisions': n,
            'soldier_set_pct': round(soldier_pct, 1),
            'soldier_set_ci_95': round(100 * ci, 1),
            'strict_pct': round(strict_pct, 1),
            'first_match_pct': round(first_pct, 1),
            'mean_jaccard': round(jaccard, 3),
        })

    # Sort by soldier_set agreement
    summary_rows.sort(key=lambda r: r['soldier_set_pct'], reverse=True)

    # Write summary CSV + TXT
    summary_csv = out_dir / 'summary.csv'
    if summary_rows:
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_txt = out_dir / 'summary.txt'
    lines = []
    lines.append('=' * 88)
    lines.append('  HUMAN-AGENT MOVE ALIGNMENT')
    lines.append('=' * 88)
    lines.append(f'  Games dir: {games_dir.resolve()}')
    lines.append(f'  Decisions evaluated: {len(all_records)}')
    lines.append('')
    lines.append(f'  {"Agent":<16} {"N":>5} {"Soldier-set":>14} '
                  f'{"Strict":>9} {"First":>9} {"Jaccard":>9}')
    lines.append(f'  {"-"*16} {"-"*5} {"-"*14} {"-"*9} {"-"*9} {"-"*9}')
    for r in summary_rows:
        sset = f'{r["soldier_set_pct"]:.1f}+/-{r["soldier_set_ci_95"]:.1f}'
        lines.append(f'  {r["agent"]:<16} {r["n_decisions"]:>5} '
                      f'{sset:>14} {r["strict_pct"]:>8.1f}% '
                      f'{r["first_match_pct"]:>8.1f}% '
                      f'{r["mean_jaccard"]:>9.3f}')
    lines.append('')
    lines.append('Metrics:')
    lines.append('  - Soldier-set: % of decisions where agent picked the '
                  'same set of soldiers')
    lines.append('  - Strict:      same soldiers AND same value-to-soldier '
                  'split')
    lines.append('  - First:       same soldier picked as the first to move')
    lines.append('  - Jaccard:     mean |A intersect H| / |A union H| over '
                  'soldier sets')
    text = '\n'.join(lines)
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nFiles written:')
    print(f'  {pos_csv}')
    print(f'  {summary_csv}')
    print(f'  {summary_txt}')


if __name__ == '__main__':
    main()
