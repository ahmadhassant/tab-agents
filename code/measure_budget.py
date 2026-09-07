"""measure_budget.py - measured per-decision compute for the E2 ladder agents."""
import instrument
from tab_runner import play_one_game


def main():
    instrument.patch_engine()
    from tab_ai import GAAgent
    from tab_look import LookAgent

    defs = {
        'BM-TLA-1': lambda: LookAgent(lookahead_seqs=1, sample_n=10,
                                        opp_n_candidates=5, use_rules=False,
                                        enum_threshold=0),
        'BM-TLA-2': lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                        opp_n_candidates=15, use_rules=False),
        'BM-TLA-3': lambda: LookAgent(lookahead_seqs=5, sample_n=300,
                                        opp_n_candidates=30, use_rules=False),
        'BM-TLA-4': lambda: LookAgent(lookahead_seqs=10, sample_n=1000,
                                        opp_n_candidates=60, use_rules=False),
        'BM-TLAG-1': lambda: LookAgent(sample_n=10, use_lookahead=False,
                                         use_rules=False, enum_threshold=0),
        'BM-TLAG-2': lambda: LookAgent(sample_n=80, use_lookahead=False,
                                         use_rules=False),
        'BM-TLAG-3': lambda: LookAgent(sample_n=400, use_lookahead=False,
                                         use_rules=False),
        'BM-TLAG-4': lambda: LookAgent(sample_n=2000, use_lookahead=False,
                                         use_rules=False),
        'BM-GA-1': lambda: GAAgent('beginner', population_size=20,
                                     generations=5, fitness_type='original'),
        'BM-GA-2': lambda: GAAgent('beginner', population_size=50,
                                     generations=10, fitness_type='original'),
        'BM-GA-3': lambda: GAAgent('beginner', population_size=100,
                                     generations=25, fitness_type='original'),
        'BM-GA-4': lambda: GAAgent('beginner', population_size=200,
                                     generations=50, fitness_type='original'),
    }
    rows = []
    for name, f in defs.items():
        a = instrument.wrap(f(), name)
        for g in range(30):
            play_one_game(a, GAAgent('beginner', fitness_type='original'),
                            seed=61000 + g)
        s = a.summary()
        rows.append(s)
        print(f"  {name:<12} {s['moves'][0]:9.0f} moves/dec  "
                f"{s['ms'][0]:7.2f} ms/dec", flush=True)
    print()
    print('E2 LADDER - MEASURED PER-DECISION COMPUTE')
    print(instrument.format_rows(rows))


if __name__ == '__main__':
    main()
