"""
tab_td.py - TD(lambda) self-play value network for tab, at proper scale.

Why this run exists
-------------------
Our earlier learning attempts were badly under-resourced, and it would be wrong
to conclude from them that reinforcement learning cannot play tab. The decisive
counter-example is in this paper's own bibliography: TD-Gammon reached
world-class backgammon - a stochastic race game with variance comparable to
tab's - using a learned evaluator trained by TD(lambda) over ~1.5 million
self-play games, then shallow search at play time.

By comparison:

  tab_rl.py    3,200 self-play games, outcome-labelled Monte Carlo, 90k params
  tab_rl2.py   4,000 self-play games, tactical label, 90k params
  TD-Gammon    ~1,500,000 self-play games, TD(lambda), shallow search

We were roughly 400x short of the reference design, used a higher-variance
target, and played greedily from the raw network with no search. This file
fixes all three:

  1. SCALE.   Generation is cheap here - ~25 games/second on 50 cores - so
              100k+ games is hours, not weeks.
  2. TARGET.  TD(lambda) lambda-returns computed by backward recursion,
              bootstrapping on the network's own next-state value instead of
              regressing the raw final outcome. Much lower variance.
  3. SEARCH.  The learned evaluator is designed to drop into the SAME one-ply
              lookahead TLA uses, giving a clean comparison in which the search
              is held fixed and only the evaluator changes (handcrafted vs
              learned). That is TD-Gammon's architecture and the fairest test
              of the paper's central claim.

State encoding: tab_env.encode_pre(board, [], player, rules) -> 124 dims,
mirrored for P2, carrying cells, queue ranks, frozen counts, no-reentry taint
and started flags. Throw slots are left zero because this is a state value.

Usage:
  python tab_td.py --rounds 20 --games 5000 --workers 50 --lam 0.7
"""

import argparse
import json
import os
import random
import time

import numpy as np

import tab_env
from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_look import get_candidate_allocations
from tab_mcts import apply_allocation, resolve_freeing, to_turn_allocation

SDIM = 124


def encode(board, player, rules):
    return tab_env.encode_pre(board, [], player, rules)


def make_value_net(hidden=(256, 128, 64)):
    import torch.nn as nn
    layers, prev = [], SDIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1), nn.Tanh()]
    return nn.Sequential(*layers)


class TDAgent:
    """Greedy over candidates by V(resulting board). Optional epsilon."""

    def __init__(self, net=None, epsilon=0.0, enum_threshold=2000,
                   sample_n=60, max_candidates=24, columns=8):
        self.net = net
        self.epsilon = epsilon
        self.enum_threshold = enum_threshold
        self.sample_n = sample_n
        self.max_candidates = max_candidates
        self.rules = TabRules(columns)

    def pick(self, board, player, values, movable, rules):
        acts = get_candidate_allocations(values, movable, player, board, rules,
                                           enum_threshold=self.enum_threshold,
                                           sample_n=self.sample_n)
        if not acts:
            return None, None
        if len(acts) > self.max_candidates:
            acts = random.sample(acts, self.max_candidates)
        if len(acts) == 1:
            post = board.clone()
            apply_allocation(post, acts[0], movable, player, rules)
            return acts[0], post
        if self.net is None or random.random() < self.epsilon:
            a = random.choice(acts)
            post = board.clone()
            apply_allocation(post, a, movable, player, rules)
            return a, post
        feats = np.empty((len(acts), SDIM), dtype=np.float32)
        posts = []
        for i, a in enumerate(acts):
            post = board.clone()
            apply_allocation(post, a, movable, player, rules)
            posts.append(post)
            feats[i] = encode(post, player, rules)
        import torch
        with torch.no_grad():
            v = self.net(torch.from_numpy(feats)).squeeze(-1).numpy()
        i = int(np.argmax(v))
        return acts[i], posts[i]

    def choose_allocation(self, board, player, throw_values,
                            movable_soldiers, rules):
        from tab_game import TurnAllocation
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        a, _ = self.pick(board, player, throw_values, movable_soldiers, rules)
        if a is None:
            return TurnAllocation()
        return to_turn_allocation(a, movable_soldiers)


def selfplay(net, seed, epsilon, max_turns=300, columns=8, opponent=None):
    """Returns per-player trajectories of encoded post-move states + outcome."""
    random.seed(seed)
    board = Board(columns)
    rules = TabRules(columns)
    ag = TDAgent(net, epsilon=epsilon, columns=columns)
    traj = {PLAYER_1: [], -PLAYER_1: []}
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        rem = resolve_freeing(board, player, seq, rules)
        mov = rules.get_movable_soldiers(board, player, rem)
        if mov and rem:
            if opponent is not None and player != PLAYER_1:
                al = opponent.choose_allocation(board, player, rem, mov, rules)
                lists = [[] for _ in mov]
                idx = {q: i for i, q in enumerate(mov)}
                for sm in al.soldier_moves:
                    if sm.from_pos in idx:
                        lists[idx[sm.from_pos]] = list(sm.values_used)
                apply_allocation(board, lists, mov, player, rules)
            else:
                a, post = ag.pick(board, player, rem, mov, rules)
                if a is not None:
                    apply_allocation(board, a, mov, player, rules)
                    traj[player].append(encode(board, player, rules))
        if board.is_game_over() is not None:
            break
        turn += 1
    w = board.is_game_over()
    if w is None:
        p1, p2 = board.count_alive(1), board.count_alive(-1)
        w = 1 if p1 > p2 else (-1 if p2 > p1 else 0)
    return traj, w


_W = {}


def _init(ckpt, eps, hidden=(256, 128, 64), mix=0.0):
    import torch
    # One thread per worker: with ~55 processes on 64 cores the default torch
    # thread pool oversubscribes badly and dominates runtime (round 2 of the
    # 512-256-128 run took 1348s vs 40s for the smaller net).
    torch.set_num_threads(1)
    net = None
    _W['hidden'] = hidden
    _W['mix'] = mix
    if ckpt and os.path.exists(ckpt):
        net = make_value_net(hidden)
        net.load_state_dict(torch.load(ckpt, map_location='cpu',
                                         weights_only=True))
        net.eval()
    _W['net'] = net
    _W['eps'] = eps


def _game(seed):
    try:
        opp = None
        if _W.get('mix', 0.0) > 0 and (seed % 100) < int(_W['mix'] * 100):
            from tab_ai import GAAgent
            opp = GAAgent('beginner', fitness_type='original')
        return selfplay(_W['net'], seed, _W['eps'], opponent=opp)
    except Exception:
        return {1: [], -1: []}, 0


def generate(ckpt, n_games, workers, seed0, eps, hidden=(256, 128, 64),
               mix=0.0):
    import multiprocessing as mp
    out, wins = [], {1: 0, -1: 0, 0: 0}
    with mp.Pool(processes=workers, initializer=_init,
                   initargs=(ckpt, eps, hidden, mix)) as pool:
        for traj, w in pool.imap_unordered(_game, range(seed0, seed0 + n_games),
                                             chunksize=8):
            for player, states in traj.items():
                if states:
                    z = 0.0 if w == 0 else (1.0 if w == player else -1.0)
                    out.append((np.stack(states), z))
            wins[w] = wins.get(w, 0) + 1
    return out, wins


def lambda_targets(net, episodes, device, lam=0.7, batch=8192):
    """Backward lambda-return: G_t = (1-lam)*V(s_{t+1}) + lam*G_{t+1}, G_T = z."""
    import torch
    net.to(device).eval()
    allX = np.concatenate([e[0] for e in episodes]) if episodes else None
    with torch.no_grad():
        vs = []
        for i in range(0, len(allX), batch):
            xb = torch.from_numpy(allX[i:i + batch]).to(device)
            vs.append(net(xb).squeeze(-1).cpu().numpy())
        V = np.concatenate(vs)
    X, Y, off = [], [], 0
    for states, z in episodes:
        n = len(states)
        v = V[off:off + n]
        off += n
        g = np.empty(n, dtype=np.float32)
        nxt = z
        for t in range(n - 1, -1, -1):
            if t == n - 1:
                g[t] = z
            else:
                g[t] = (1 - lam) * v[t + 1] + lam * nxt
            nxt = g[t]
        X.append(states); Y.append(g)
    return np.concatenate(X), np.concatenate(Y)


def train(net, X, Y, device, epochs=3, batch=4096, lr=1e-3):
    import torch
    import torch.nn.functional as F
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X),
                                          torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    last = 0.0
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.mse_loss(net(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        last = tot / max(n, 1)
    net.eval()
    return last


def evaluate(net, n, seed0, fitness='original'):
    from tab_ai import GAAgent
    from tab_runner import play_one_game
    w = 0
    for g in range(n):
        me = TDAgent(net, epsilon=0.0)
        opp = GAAgent('beginner', fitness_type=fitness)
        if g % 2 == 0:
            r = play_one_game(me, opp, seed=seed0 + g); w += (r['winner'] == 1)
        else:
            r = play_one_game(opp, me, seed=seed0 + g); w += (r['winner'] == -1)
    return w / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--games', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lam', type=float, default=0.7)
    ap.add_argument('--eval-games', type=int, default=100)
    ap.add_argument('--out', type=str, default='checkpoints_td')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--hidden', type=str, default='256,128,64')
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--mix', type=float, default=0.0,
                    help='fraction of games played against GA-Original')
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, 'td.pt')
    hidden = tuple(int(x) for x in args.hidden.split(','))
    net = make_value_net(hidden)
    torch.save(net.state_dict(), ckpt)
    total_games = 0
    print(f'device={device}  params={sum(p.numel() for p in net.parameters())}'
            f'  lambda={args.lam}  target={args.rounds*args.games} games',
            flush=True)

    hist, best = [], -1.0
    for rnd in range(1, args.rounds + 1):
        eps = max(0.05, 0.30 * (1 - (rnd - 1) / max(args.rounds - 1, 1)))
        t0 = time.time()
        eps_used = eps if rnd > 1 else 1.0     # round 1: pure random bootstrap
        episodes, wins = generate(ckpt, args.games, args.workers,
                                    seed0=rnd * 200000, eps=eps_used,
                                    hidden=hidden, mix=args.mix)
        total_games += args.games
        gen = time.time() - t0
        X, Y = lambda_targets(net, episodes, device, lam=args.lam)
        mse = train(net, X, Y, device, epochs=args.epochs)
        torch.save(net.state_dict(), ckpt)
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            cpu = make_value_net(hidden)
            cpu.load_state_dict(torch.load(ckpt, map_location='cpu',
                                             weights_only=True))
            cpu.eval()
            wr = evaluate(cpu, args.eval_games, seed0=777000 + rnd * 1000)
        else:
            wr = float('nan')
        flag = ''
        if wr == wr and wr > best:
            best = wr
            torch.save(net.state_dict(), os.path.join(args.out, 'best.pt'))
            flag = '  <-- best'
        print(f'  round {rnd:2d}/{args.rounds}  eps={eps_used:.2f}  '
                f'games={total_games:6d}  states={len(X):7d}  '
                f'gen={gen:5.0f}s  mse={mse:.4f}  '
                f'vs GA-Original={wr*100:5.1f}%{flag}', flush=True)
        hist.append({'round': rnd, 'games': total_games, 'mse': mse,
                       'winrate': wr})
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=2)

    print('\n=== FINAL (TD-lambda) ===', flush=True)
    for h in hist:
        print(f"  {h['games']:7d} games : {h['winrate']*100:5.1f}%", flush=True)
    print(f'  best = {best*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
