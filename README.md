# ṭāb: engine, agents and replication package

Rules engine, agents, tournament harness and complete raw results for:

> A. B. Hassanat, G. A. Altarawneh, O. Alhabashneh and A. S. Tarawneh, "Tactical
> Lookahead and Heuristic Calibration
> in the Ancient Board Game of ṭāb: A Systematic Study of Agent Design,"
> *IEEE Transactions on Games* (under review).

**Play the agents in a browser — no install:** download
[`web/tab_game.html`](web/tab_game.html) and double-click it. The trained network is
embedded in the file; it works offline. Both dropdowns carry exactly the eight agents
reported in the paper — TLA-NN (rank 1), TD-Greedy (2), TLA-S (3), TLA (4), TLA-B (5),
GA-Original (6), TLA-G (8) and PSA (10) — each labelled with its tournament rank.

**Playing ṭāb online:** the game is also live for the ṭāb community, in Arabic and
English, on the Visit Karak cultural-heritage portal —
<https://visitkarak.com/tab-game.php>. That deployment is maintained separately from
this research package and carries its own agent set, so use the build above to
reproduce anything reported in the paper.

Every number in the paper can be regenerated from this repository with the commands
below. Pure Python; `numpy` is needed only for the learned agents and `torch` only to
retrain them.

---

## Contents

| Path | What it is |
|---|---|
| `code/` | Engine (`tab_game.py`), all agents, tournament harness, analysis scripts |
| `results/` | Raw per-game CSVs and summaries behind every table and figure |
| `human_logs/` | The 11 expert game logs behind the 342-decision alignment study |
| `web/` | Browser-playable build with the trained network embedded |
| `checkpoints/` | The TD(λ) network the paper reports (`best.pt`) |

## Install

```bash
git clone https://github.com/ahmadhassant/tab-agents.git
cd tab-agents
python -m pip install -r requirements.txt
python code/tab_game.py            # engine self-check
python code/tournament_grid.py --list-agents
```

Python 3.10+. Results are seed-determined: the same `--seed`, agent set and `--games`
reproduce the same games on any number of workers, because agents are constructed
inside each worker process and nothing is shared between games. Lower `--workers` to
match your machine.

`--games N` is **per side**: a pairing plays `N` games with A as P1 and `N` with A as
P2, so `--games 500` means 1,000 games per pairing.

---

## Reproducing each table and figure

Run everything from `code/`.  Each run writes `games.csv` (one row per game),
`summary.csv` and `summary.txt` into `--out`.

### Table II, Table III, Figures 1–3 — main tournament (28,000 games)

```bash
python tournament_grid.py --games 500 --workers 55 --seed 42 \
  --out ../results/tournament_main_28k \
  --agents LOOK-A3-simple-stack LOOK LOOK-no-enum LOOK-no-LA \
           GA-Original GA-Expert GA-Fuzzy PSA
```

### Table IV, Figure 4 — complete autopsy (132,000 games)

Every LookEvaluator axis ablated singly, against two reference opponents.

```bash
python tournament_grid.py --games 1000 --workers 20 --seed 20260903 \
  --out ../results/autopsy_full \
  --agents LOOK-my-eval LOOK-A1-material5 LOOK-A2-no-frozendist \
           LOOK-A3-simple-stack LOOK-A4-strong-danger LOOK-A5-stacking-off \
           LOOK-A6-all-fixed LOOK-A7-no-hiding LOOK-A8-no-central \
           LOOK-A9-no-entry GA-Expert LOOK
python analyze_autopsy.py
```

`results/autopsy_400g/` is the superseded 400-game pilot (seed base 42), kept only
for provenance. The paper quotes `autopsy_full/`.

### Section VII-B — the same stacking term in an independently designed evaluator (20,000 games)

```bash
python tournament_grid.py --games 1000 --workers 55 --seed 20260907 \
  --out ../results/psa_autopsy \
  --agents PSA PSA-X PSA-N GA-Expert LOOK-A3-simple-stack
python measure_psa_stack.py --games 400 --workers 55   # how often each variant changes the move
```

`PSA-X` replaces PSA's rescue-aware quadratic stacking penalty with the flat linear
-3c; `PSA-N` deletes the term. Both are `stacking_mode` settings on
`PhaseSamplingAgent`.

### Table V — human move alignment (342 decisions)

```bash
python human_alignment.py --games-dir ../human_logs \
  --out ../results/alignment_with_tle \
  --agents TLA-S TLE LOOK LOOK-no-enum LOOK-no-LA PSA GA-Expert GA-Original Random
python stats_alignment.py       # game-level clustered bootstrap, McNemar
```

### Table VII — confirmatory tournament, held-out seeds (66,000 games)

Run after all evaluator development was frozen. This is the independent evaluation.

```bash
export TAB_TD_CKPT=../checkpoints/best.pt      # Windows: set TAB_TD_CKPT=..\checkpoints\best.pt
python tournament_grid.py --games 500 --workers 55 --seed 90210 \
  --out ../results/final_roster \
  --agents LOOK-A3-simple-stack LOOK LOOK-no-enum LOOK-no-LA \
           GA-Original GA-Expert GA-Fuzzy PSA TLA-NN TD-Greedy MCTS-800-h Random
python final_standings.py       # full 12-agent table
python final_subset.py          # balanced 10-agent table
```

### Table VIII — matched-compute ladder (66,000 games)

Four compute budgets each for TLA, TLA-G and the GA.

```bash
python measure_compute.py       # per-decision evaluator calls and wall time
python tournament_grid.py --games 500 --workers 45 --seed 9090 \
  --out ../results/budget_sweep_p \
  --agents BM-TLA-1 BM-TLA-2 BM-TLA-3 BM-TLA-4 \
           BM-TLAG-1 BM-TLAG-2 BM-TLAG-3 BM-TLAG-4 \
           BM-GA-1 BM-GA-2 BM-GA-3 BM-GA-4
python analyze_budget.py
```

### Section IX — MCTS budget ladder and search depth

```bash
# budget ladder (18,000 games)
python tournament_grid.py --games 200 --workers 60 --seed 777 \
  --out ../results/mcts_ladder \
  --agents MCTS-200 MCTS-800 MCTS-200-h MCTS-800-h \
           GA-Original GA-Expert LOOK LOOK-A3-simple-stack PSA Random

# flat search vs chance-node tree, two depths (15,000 games)
python tournament_grid.py --games 500 --workers 60 --seed 2026 \
  --out ../results/mcts_tree_p \
  --agents MCTST-800-h MCTST-800-h-d2 MCTST-800-h-d6 MCTS-800-h \
           LOOK-A3-simple-stack GA-Original

python test_mcts_parity.py      # engine-parity harness for the ported MCTS
```

### Section IX — learned agents

```bash
python tab_td.py                # retrain TD(lambda) self-play (long)
python tournament_grid.py --games 500 --workers 55 --seed 31415 \
  --out ../results/tlann \
  --agents TLA-NN TD-Greedy LOOK LOOK-A3-simple-stack LOOK-no-LA \
           MCTS-800-h GA-Original GA-Expert PSA Random
python analyze_tlann.py
python verify_distil.py
python floorcheck_dqn.py        # learned-agent floor check
```

### Section XI — board-size sweep (60,000 games)

```bash
for C in 7 8 9 11; do
  python tournament_grid.py --games 500 --workers 14 --seed 4040 --columns $C \
    --out ../results/board_4x${C}_p \
    --agents LOOK-A3-simple-stack LOOK LOOK-my-eval LOOK-no-LA GA-Expert GA-Original
done
python analyze_boards.py
```

### Supporting measurements quoted in the text

```bash
python measure_fire_rate.py     # Sec. VII   exemption-clause fire rate
python mech_probe3.py           # Sec. VII-A call-site census of the stacking penalty
python measure_forks.py         # Sec. III-I fork frequency, TLA family
python measure_forks_psa.py     # Sec. III-I PSA internal vs executed fork choice
```

### Verifying the paper against the data

```bash
python audit_manuscript.py      # recompute every Table II and prose number
python audit2.py                # autopsy, Bradley–Terry fit, alignment
```

---

## Seed bases used in the paper

| Experiment | Seed base |
|---|---|
| Main tournament, autopsy pilot | 42 |
| Complete autopsy (132k) | 20260903 |
| Confirmatory tournament | 90210 |
| Matched-compute ladder | 9090 |
| MCTS budget ladder | 777 |
| MCTS tree/depth | 2026 |
| Board-size sweep | 4040 |
| Learned-agent runs | 31415, 27182 |

Game *k* of a pairing uses `seed + k`.

## Agent names

Registry keys differ from the display names used in the paper:

| Paper | Registry key |
|---|---|
| TLA | `LOOK` |
| TLA-G | `LOOK-no-LA` |
| TLA-B | `LOOK-no-enum` |
| TLE | `LOOK-my-eval` |
| TLA-S | `LOOK-A3-simple-stack` |
| PSA, GA-Original, GA-Expert, GA-Fuzzy, Random | same |
| MCTS-*, MCTST-* (tree), TLA-NN, TD-Greedy | same |
| Autopsy variants TLE-M/F/D/N/X/A and the three added axes | `LOOK-A1…A9`, `LOOK-A6-all-fixed` |

`python tournament_grid.py --list-agents` prints the live registry.

## Licence

Code is MIT (`LICENSE`). Results, human game logs and figures are CC BY 4.0
(`LICENSE-DATA`). Please cite the paper — see `CITATION.cff`.
