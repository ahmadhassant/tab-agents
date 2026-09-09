# tab Agent Comparison Bundle

Code for a systematic comparison of tab-playing agents.
For the IEEE Transactions on Games paper on agent design principles.

## What's in here

```
tab_look_bundle/
|-- tab_game.py            engine - Board, TabRules, simulate_allocation
|-- tab_ai.py              existing agents - FitnessEvaluator, GAAgent, etc.
|-- tab_look.py            NEW - LOOK agent (one-ply lookahead + ablations)
|-- tab_psa.py            NEW - Python port of PSA for fair comparison
|-- tab_runner.py          headless game loop with fixed fork-picker
|-- tournament_grid.py     parallel round-robin runner (this is the main tool)
|-- analyze_results.py     post-tournament analysis (matrix, CIs, side effects)
|-- README.md              this file
`-- __init__.py
```

## Quick check (5 min, any machine)

```bash
python tournament_grid.py --games 5 --workers 4 \
    --agents LOOK PSA GA-Expert --out results/smoke
```

This runs 3 pairings x 10 games each (60 games total). Verifies the bundle
works before launching the real run.

## Headline tournament (you, on 64 cores)

```bash
python tournament_grid.py --games 200 --workers 64
```

That's 7 default agents x 21 pairings x 400 games (200 per side) = **8 400 games**.
At ~3 s/game on one core, parallelized 64-way -> roughly **8 minutes** wall time.
The 95% CI tightens to about +/-5%, so 60% vs 50% becomes detectable.

Output lands in `./results/`:

- `games.csv` - one row per game (agent_p1, agent_p2, winner, turns, ...)
- `summary.csv` - pair-level summary with win rates and 95% CIs
- `summary.txt` - same, human-readable

If you want more confidence, push `--games 500` (~ 21 000 games, ~22 min on 64 cores).

## Full ablation grid (paper experiment)

```bash
python tournament_grid.py --games 200 --workers 64 --agents \
    LOOK LOOK-no-LA LOOK-my-eval LOOK-cont-phase LOOK-no-enum \
    PSA GA-Expert GA-Original GA-Fuzzy
```

9 agents -> 36 unique pairings -> 14 400 games. ~15 min on 64 cores.

This is the data the paper hangs on. Each LOOK-* variant isolates one
component (lookahead, evaluator choice, phase model, enumeration), so the
matrix tells us which design choice matters and by how much.

## Slow-but-strong baselines

Include `GA-Expert-adv` and `GA-Original-adv` for stronger GA opponents
(advanced level = 200 population, 100 generations). These are ~10x slower:

```bash
python tournament_grid.py --games 100 --workers 64 --agents \
    LOOK PSA GA-Expert-adv GA-Original-adv
```

~6 min on 64 cores.

## Resume after interruption

If the run dies (Ctrl-C, OOM, power blip), the CSV is flushed every 200
games. Re-run with `--resume` to pick up where you left off:

```bash
python tournament_grid.py --games 200 --workers 64 --resume
```

The script reads `games.csv`, skips completed `game_id`s, and continues.

## Analyze the results

```bash
python analyze_results.py --in results/games.csv
```

Prints four tables:

1. **Overall per-agent stats** - total games, wins, average turns, captures
2. **Win-rate matrix** - head-to-head win % with 95% CIs
3. **Statistical significance** - pairings where CI lower bound clears 50%
4. **Side effects** - does each agent play better as P1 or P2?

## Agents in the registry

| Name              | Description                                              |
|-------------------|----------------------------------------------------------|
| `LOOK`            | FitnessEvaluator + biased sampling + enumerate + 1-ply LA |
| `LOOK-no-LA`      | Same, lookahead off (ablation)                           |
| `LOOK-my-eval`    | Uses LookEvaluator instead (binary hide, discrete phase) |
| `LOOK-cont-phase` | LookEvaluator with continuous phase + smooth hide        |
| `LOOK-no-enum`    | Always sample, never enumerate                           |
| `LOOK-A1-material5`     | LOOK-my-eval with material weight 5 (was 15)       |
| `LOOK-A2-no-frozendist` | Same, but +3 per soldier in own home (no frozen distinction) |
| `LOOK-A3-simple-stack`  | Same, but -3*cnt for ALL stacks (FitnessEvaluator style) |
| `LOOK-A4-strong-danger` | Same, but danger_scale=30 instead of 10            |
| `LOOK-A5-stacking-off`  | Same, but no stacking penalty at all               |
| `LOOK-A6-all-fixed`     | All four autopsy corrections combined              |
| `PSA`            | Python port of HTML PSA agent                           |
| `GA-Expert`       | GAAgent('beginner', fitness_type='expert')               |
| `GA-Original`     | GAAgent('beginner', fitness_type='original')             |
| `GA-Fuzzy`        | GAAgent('beginner', fitness_type='fuzzy')                |
| `GA-Expert-adv`   | GAAgent('advanced', fitness_type='expert')               |
| `GA-Original-adv` | GAAgent('advanced', fitness_type='original')             |
| `Random`          | Uniform random allocation (floor agent)                  |

Use `python tournament_grid.py --list-agents` to print the live registry.

## Autopsy experiment (why does LookEvaluator lose 34%?)

LOOK-my-eval lost ~66% to every other smart agent in the 21K tournament.
The autopsy tests *which* LookEvaluator term causes the collapse by
turning off ONE design choice at a time (toward FitnessEvaluator defaults):

```bash
python tournament_grid.py --games 200 --workers 64 --agents \
    LOOK-my-eval LOOK-A1-material5 LOOK-A2-no-frozendist \
    LOOK-A3-simple-stack LOOK-A4-strong-danger LOOK-A5-stacking-off \
    LOOK-A6-all-fixed GA-Expert LOOK
```

9 agents -> 36 pairings -> 14400 games -> ~15 min on 64 cores.
If any LOOK-A* variant climbs from LOOK-my-eval's ~34% baseline toward ~50%
against GA-Expert, that term was the culprit. LOOK-A6 (all four corrections)
should match plain LOOK if the autopsy is conclusive.

## Human-alignment study

Measures how often each agent picks the same move as the human (Ahmad +
playtesters across 11 games, 342 decisions total):

```bash
python human_alignment.py --games-dir HumanGames \
    --agents LOOK PSA GA-Expert GA-Original Random
```

Runs in ~3 minutes on a single core. Produces `results_alignment/`:
- `positions.csv`: per-decision records (game, turn, throws, human move, agent move, match flags)
- `summary.csv`: per-agent agreement rates with 95% CIs
- `summary.txt`: human-readable summary

Three metrics:
- **Soldier-set**: same set of soldiers chosen (coarse, robust)
- **Strict**: same per-soldier value allocation (fine, demanding)
- **First**: same first soldier picked
- **Jaccard**: continuous similarity over soldier sets

## Reproducibility

The tournament uses seeded RNG. Default seed base = 42; pass `--seed` to
change. Same seed + same agent set + same `--games` = identical results.

## Implementation notes

- **Fork picker is fixed** (`tab_runner.py`): `cap x 10 + enemy_home x 5`.
  Same for every agent. Keeps comparisons fair - fork-picking is a
  rules-level decision, not an agent's strategy.
- **Side swap** is done explicitly: half the games have A as P1, half as P2.
  This neutralizes the P1-goes-first effect.
- **Workers are isolated**: each game runs in its own process, agents are
  instantiated inside the worker (not pickled across).
- **Draws** (game hits max_turns) are counted by surviving soldier majority.
  In practice this almost never happens.

## What we found in preliminary 30-game tests

(See `prelim_results.md` if included for full data.)

- LOOK with the original `FitnessEvaluator` beats PSA (63%, n=30, +/-17%)
- LOOK matches GA-Expert (53%) and beats GA-Original (67%)
- Lookahead's contribution is ~0% in 30-game samples - dice variance
  dominates the one-ply horizon. Need bigger sample to detect a small effect.
- LOOK with the new `LookEvaluator` (smooth phase, dice-prior danger)
  performs *worse* than with `FitnessEvaluator`. The 7-rule original is
  already well-calibrated.

The 64-core run should resolve all of these to +/-5% CIs.

## Author & Contact

* **Authors:** Ahmad B. Hassanat, Ghada A. Altarawneh, and Ahmad S. Tarawneh (Mutah University, Jordan)
* **Code Support:** Claude (Anthropic)
* **Date:** May 2026
