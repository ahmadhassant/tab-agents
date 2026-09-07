"""
measure_compute.py - per-decision compute table for every agent (E1).

Plays each agent against a fixed reference opponent and reports the compute
each one spends per decision. This is the table R1.1 and R3.11 asked for.

Usage:
  python measure_compute.py --games 20
"""
import argparse

import instrument
from tab_runner import play_one_game


def build_agents():
    from tab_ai import GAAgent, RandomAgent
    from tab_look import LookAgent
    from tab_psa import PhaseSamplingAgent
    from tab_mcts import MCTSAgent, MCTSTreeAgent
    return {
        'GA-Original': lambda: GAAgent('beginner', fitness_type='original'),
        'GA-Expert': lambda: GAAgent('beginner', fitness_type='expert'),
        'GA-Fuzzy': lambda: GAAgent('beginner', fitness_type='fuzzy'),
        'TLA': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                   opp_n_candidates=15, use_rules=False),
        'TLA-G': lambda: LookAgent(sample_n=80, use_lookahead=False,
                                     use_rules=False),
        'TLA-B': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                     opp_n_candidates=15, use_rules=False,
                                     enum_threshold=0),
        'TLA-S': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                     opp_n_candidates=15, use_rules=True,
                                     binary_hide=True,
                                     use_continuous_phase=False,
                                     stacking_mode='simple'),
        'PSA': lambda: PhaseSamplingAgent(n_samples=80),
        'MCTS-200-h': lambda: MCTSAgent(n_simulations=200, rollout='heavy'),
        'MCTS-800-h': lambda: MCTSAgent(n_simulations=800, rollout='heavy'),
        'MCTST-800-h': lambda: MCTSTreeAgent(n_simulations=800,
                                               rollout='heavy', max_depth=4),
        'Random': lambda: RandomAgent(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--seed', type=int, default=31337)
    args = ap.parse_args()

    instrument.patch_engine()
    from tab_ai import GAAgent

    summaries = []
    for name, factory in build_agents().items():
        agent = instrument.wrap(factory(), name)
        for g in range(args.games):
            opp = GAAgent('beginner', fitness_type='original')
            play_one_game(agent, opp, seed=args.seed + g)
        s = agent.summary()
        summaries.append(s)
        print(f"  done {name:<14} {s['decisions']:5d} decisions, "
                f"{s['moves'][0]:.0f} moves/dec, {s['ms'][0]:.1f} ms/dec",
                flush=True)

    print()
    print('PER-DECISION COMPUTE  (opponent: GA-Original, '
            f'{args.games} games each)')
    print(instrument.format_rows(summaries))


if __name__ == '__main__':
    main()
