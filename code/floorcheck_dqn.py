"""
floorcheck_dqn.py - Are the existing self-play DQN checkpoints worth anything?

RUN THIS ON THE WORKSTATION (needs PyTorch; it is not installed on the desktop).

Background
----------
05_Server_Runs/engine_snapshots/fuzz_variants/tab_engine (21)/ and (22)/
contain a genuine self-play DQN: replay buffer, target network, gamma=0.99,
and saved checkpoints across stage-1 rounds 1-5 and stage-2 s2r1-s2r10.
Its tab_game.py is AST-identical to 02_Code/tab_game.py (verified 2 Sep 2026),
so those checkpoints were trained on the game we are still playing.

What is unknown is whether they are any GOOD. The network is a small MLP over
24 hand-engineered features, and its evaluator wrapper blanks frozen_queue,
frozen_at and no_reentry - so it cannot see freezing or taint. This script
measures the consequence before we invest in retraining.

Decision rule
-------------
  >= 50% vs GA-Original  -> the checkpoints are a usable baseline; report them.
  40-50%                 -> weak but honest; retrain on the 191-dim encoder.
  <  40%                 -> discard; train a fresh self-play agent.

Usage
-----
  python floorcheck_dqn.py --games 200
  python floorcheck_dqn.py --games 200 --engine-dir "..\\05_Server_Runs\\engine_snapshots\\fuzz_variants\\tab_engine (22)\\tab_engine"
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ENGINE = (HERE.parent / '05_Server_Runs' / 'engine_snapshots' /
                    'fuzz_variants' / 'tab_engine (21)' / 'tab_engine')


def load_module(name, path):
    """Import a module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200,
                    help='games per pairing (200 gives about +/-7pp)')
    ap.add_argument('--engine-dir', type=str, default=str(DEFAULT_ENGINE))
    ap.add_argument('--checkpoints', nargs='+', default=None,
                    help='specific .pth files; default = all in engine-dir')
    ap.add_argument('--seed', type=int, default=4242)
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir)
    if not engine_dir.exists():
        sys.exit(f"engine dir not found: {engine_dir}")

    try:
        import torch
    except ImportError:
        sys.exit("PyTorch not installed. Run this on the workstation.")

    # 02_Code first on the path, so tab_game/tab_ai/tab_look resolve to the
    # CURRENT engine. tab_dqn is then loaded explicitly from the snapshot.
    sys.path.insert(0, str(HERE))
    from tab_ai import GAAgent
    from tab_look import LookAgent
    from tab_runner import play_one_game

    sys.path.append(str(engine_dir))
    tab_dqn = load_module('tab_dqn_snapshot', engine_dir / 'tab_dqn.py')
    if not getattr(tab_dqn, 'HAS_TORCH', False):
        sys.exit("tab_dqn reports HAS_TORCH=False - check the torch import.")

    ckpts = ([Path(c) for c in args.checkpoints] if args.checkpoints
             else sorted(engine_dir.glob('*.pth')))
    if not ckpts:
        sys.exit(f"no .pth checkpoints in {engine_dir}")

    opponents = {
        'GA-Original': lambda: GAAgent('beginner', fitness_type='original'),
        'GA-Expert':   lambda: GAAgent('beginner', fitness_type='expert'),
        'TLA-S':       lambda: LookAgent(lookahead_seqs=3, sample_n=80,
                                           opp_n_candidates=15, use_rules=True,
                                           binary_hide=True,
                                           use_continuous_phase=False,
                                           stacking_mode='simple'),
    }

    def build_dqn_agent(ckpt):
        net = tab_dqn.TabDQN(input_dim=24, hidden_dims=(128, 64, 32))
        blob = torch.load(str(ckpt), map_location='cpu', weights_only=False)
        state = blob.get('model_state', blob.get('model', blob))
        net.load_state_dict(state)
        net.eval()
        vc = tab_dqn.SoldierValueCalculator(8)
        return tab_dqn.ExhaustiveAgent(
            8, evaluator=tab_dqn.DQNEvaluator(net, vc))

    print("=" * 74)
    print(f"  DQN floor-check   engine: {engine_dir.name}   "
            f"{args.games} games/pairing")
    print("=" * 74)
    results = {}
    for ckpt in ckpts:
        try:
            build_dqn_agent(ckpt)
        except Exception as e:
            print(f"\n{ckpt.name}: LOAD FAILED - {e}")
            continue
        print(f"\n{ckpt.name}")
        for opp_name, opp_fn in opponents.items():
            wins = 0
            t0 = time.time()
            for g in range(args.games):
                # side-swap so first-player advantage cancels
                if g % 2 == 0:
                    r = play_one_game(build_dqn_agent(ckpt), opp_fn(),
                                        seed=args.seed + g)
                    wins += (r['winner'] == 1)
                else:
                    r = play_one_game(opp_fn(), build_dqn_agent(ckpt),
                                        seed=args.seed + g)
                    wins += (r['winner'] == -1)
            pct = 100.0 * wins / args.games
            ci = 196.0 * (pct / 100 * (1 - pct / 100) / args.games) ** 0.5
            print(f"   vs {opp_name:14s} {wins:4d}/{args.games} = "
                    f"{pct:5.1f}% +/-{ci:.1f}   ({time.time()-t0:.0f}s)")
            results[(ckpt.name, opp_name)] = pct

    print("\n" + "=" * 74)
    best = max((v for (c, o), v in results.items() if o == 'GA-Original'),
                 default=0.0)
    print(f"  Best checkpoint vs GA-Original: {best:.1f}%")
    if best >= 50:
        print("  -> USABLE. Report these checkpoints as the learned baseline.")
    elif best >= 40:
        print("  -> WEAK. Retrain this architecture on tab_env's 191-dim state.")
    else:
        print("  -> DISCARD. Train a fresh self-play agent on the current engine.")
    print("=" * 74)


if __name__ == '__main__':
    main()
