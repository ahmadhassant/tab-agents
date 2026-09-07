"""
tab_rl2.py - self-play value network, revision 2.

What changed from tab_rl.py, and why
------------------------------------
Run 1 (tab_rl.py) showed MSE falling steadily while win rate FELL from 13.3%
(round 1) to 8.3% (round 3). Loss down, strength down is the signature of a
network learning the base rate rather than move quality.

Diagnosis: labelling every state with the final game outcome (+-1) is an
extremely high-variance target in a dice-dominated race. One allocation barely
shifts P(win), so the regression converges toward the mean and argmax over
near-identical scores becomes close to arbitrary.

Two changes:

1. TACTICAL TARGET. Label with the immediate, low-variance signal the encoder
   was designed for - tab_env.py's own docstring specifies
   "label = K - L (kills this turn minus losses from opponent's response)".
   The target is a blend:

       target = w * clip((K - L)/2, -1, 1)  +  (1 - w) * outcome

   so the network still carries long-run value but gets a dense per-decision
   gradient. w is --w-tactical (default 0.7).

2. OPPONENT BOOTSTRAP. Pure self-play from a random initialisation gives almost
   no gradient signal early, because both sides play noise. The first
   --bootstrap-rounds rounds play against GA-Original, so the recorded replies
   (and therefore L) come from a competent opponent. Later rounds are self-play.

Usage:
  python tab_rl2.py --rounds 10 --games 400 --workers 50 --out checkpoints_rl2
"""

import argparse
import json
import os
import random
import time

import numpy as np

import tab_env
from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_mcts import apply_allocation, resolve_freeing
from tab_rl import make_net, NetAgent, STATE_DIM


def _alloc_lists(alloc, movable):
    lists = [[] for _ in movable]
    idx = {p: i for i, p in enumerate(movable)}
    for sm in alloc.soldier_moves:
        if sm.from_pos in idx:
            lists[idx[sm.from_pos]] = list(sm.values_used)
    return lists


def play_labeled_game(net, seed, epsilon, opponent=None, w_tactical=0.7,
                        max_turns=300, columns=8, device='cpu'):
    """
    Play one game and label every decision our learner makes with
    w*(K-L)/2 + (1-w)*outcome.

    opponent=None  -> self-play (the net plays both sides, both sides labelled)
    opponent=agent -> the learner is P1, `opponent` is P2 (only P1 labelled)
    """
    random.seed(seed)
    board = Board(columns)
    rules = TabRules(columns)
    learner = {PLAYER_1: NetAgent(net, epsilon=epsilon, device=device)}
    if opponent is None:
        learner[-PLAYER_1] = NetAgent(net, epsilon=epsilon, device=device)

    pending = []       # (state, player, K, my_alive_after)
    records = []       # (state, player, K, L)

    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        opp = -player

        # settle any pending record for this player: losses since their move
        for rec in [p for p in pending if p[1] == player]:
            state, pl, K, alive_after = rec
            L = alive_after - board.count_alive(pl)
            records.append((state, pl, K, max(L, 0)))
            pending.remove(rec)

        seq = StickDice.throw_turn()
        remaining = resolve_freeing(board, player, seq, rules)
        movable = rules.get_movable_soldiers(board, player, remaining)
        if movable and remaining:
            agent = learner.get(player)
            opp_alive_before = board.count_alive(opp)
            if agent is not None:
                before = len(agent.trace)
                alloc = agent.choose_allocation(board, player, remaining,
                                                  movable, rules)
                apply_allocation(board, _alloc_lists(alloc, movable),
                                   movable, player, rules)
                if len(agent.trace) > before:
                    state = agent.trace[-1][0]
                    K = opp_alive_before - board.count_alive(opp)
                    pending.append((state, player, max(K, 0),
                                      board.count_alive(player)))
            else:
                alloc = opponent.choose_allocation(board, player, remaining,
                                                     movable, rules)
                apply_allocation(board, _alloc_lists(alloc, movable),
                                   movable, player, rules)

        if board.is_game_over() is not None:
            break
        turn += 1

    winner = board.is_game_over()
    if winner is None:
        p1, p2 = board.count_alive(1), board.count_alive(-1)
        winner = 1 if p1 > p2 else (-1 if p2 > p1 else 0)

    # settle anything still pending (game ended before the reply)
    for state, pl, K, alive_after in pending:
        L = alive_after - board.count_alive(pl)
        records.append((state, pl, K, max(L, 0)))

    out = []
    for state, pl, K, L in records:
        outcome = 0.0 if winner == 0 else (1.0 if winner == pl else -1.0)
        tactical = max(-1.0, min(1.0, (K - L) / 2.0))
        out.append((state, w_tactical * tactical +
                      (1.0 - w_tactical) * outcome))
    return out, winner


# ============================================================================
# Parallel generation
# ============================================================================

_W = {}


def _init(ckpt, device, use_opponent):
    import torch
    net = make_net()
    if ckpt and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
    net.eval()
    _W['net'] = net
    _W['device'] = device
    _W['use_opponent'] = use_opponent


def _game(args):
    seed, eps, w = args
    try:
        opp = None
        if _W['use_opponent']:
            from tab_ai import GAAgent
            opp = GAAgent('beginner', fitness_type='original')
        return play_labeled_game(_W['net'], seed, eps, opponent=opp,
                                   w_tactical=w, device=_W['device'])
    except Exception:
        return [], 0


def generate(ckpt, n_games, workers, seed0, eps, w, use_opponent):
    import multiprocessing as mp
    args = [(seed0 + i, eps, w) for i in range(n_games)]
    data, wins = [], {1: 0, -1: 0, 0: 0}
    with mp.Pool(processes=workers, initializer=_init,
                   initargs=(ckpt, 'cpu', use_opponent)) as pool:
        for rows, wn in pool.imap_unordered(_game, args, chunksize=2):
            data.extend(rows)
            wins[wn] = wins.get(wn, 0) + 1
    return data, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--eval-games', type=int, default=60)
    ap.add_argument('--bootstrap-rounds', type=int, default=3,
                    help='rounds played against GA-Original before self-play')
    ap.add_argument('--w-tactical', type=float, default=0.7)
    ap.add_argument('--out', type=str, default='checkpoints_rl2')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    import torch
    from tab_rl import train, evaluate
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, 'net.pt')

    net = make_net()
    torch.save(net.state_dict(), ckpt)
    print(f'device={device}  params={sum(p.numel() for p in net.parameters())}'
            f'  w_tactical={args.w_tactical}  bootstrap={args.bootstrap_rounds}',
            flush=True)

    history, best = [], -1.0
    for rnd in range(1, args.rounds + 1):
        eps = max(0.05, 0.40 * (1.0 - (rnd - 1) / max(args.rounds - 1, 1)))
        boot = rnd <= args.bootstrap_rounds
        t0 = time.time()
        print(f'\n=== round {rnd}/{args.rounds}  eps={eps:.2f}  '
                f'{"vs GA-Original" if boot else "self-play"} ===', flush=True)
        data, wins = generate(ckpt, args.games, args.workers,
                                seed0=rnd * 100000 + 7, eps=eps,
                                w=args.w_tactical, use_opponent=boot)
        if not data:
            print('  no data, aborting', flush=True)
            break
        tg = [d[1] for d in data]
        print(f'  {len(data)} samples in {time.time()-t0:.0f}s  '
                f'target mean={np.mean(tg):+.3f} sd={np.std(tg):.3f}',
                flush=True)
        mse = train(net, data, device, epochs=args.epochs)
        torch.save(net.state_dict(), ckpt)
        cpu_net = make_net()
        cpu_net.load_state_dict(torch.load(ckpt, map_location='cpu',
                                             weights_only=True))
        cpu_net.eval()
        wr = evaluate(cpu_net, args.eval_games, 'cpu')
        flag = ''
        if wr > best:
            best = wr
            torch.save(net.state_dict(), os.path.join(args.out, 'best.pt'))
            flag = '  <-- best'
        print(f'  round {rnd}: mse={mse:.4f}  vs GA-Original = {wr*100:.1f}%'
                f'  ({time.time()-t0:.0f}s){flag}', flush=True)
        history.append({'round': rnd, 'mse': mse, 'winrate': wr,
                          'samples': len(data), 'bootstrap': boot})
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print('\n=== FINAL (tab_rl2) ===', flush=True)
    for h in history:
        tag = 'boot' if h['bootstrap'] else 'self'
        print(f"  round {h['round']:2d} [{tag}]: vs GA-Original "
                f"{h['winrate']*100:5.1f}%   mse={h['mse']:.4f}", flush=True)
    print(f'  best = {best*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
