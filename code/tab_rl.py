"""
tab_rl.py - self-play value network for tab, on the CURRENT engine.

Why this exists
---------------
The six self-play DQN checkpoints inherited from tab_engine (21) scored 2-10%
against every reference agent (03_Results/floorcheck.log). The audit found the
cause: their DQNEvaluator rebuilt a bare Board and blanked frozen_queue,
frozen_at and no_reentry, so the network could not see freezing or taint - it
could not represent the game it was playing.

This retrains the same idea on tab_env.py's 191-dim encoding, which DOES carry
queue ranks [32:48], frozen counts [48:50], and both the pre- and post-move
no-reentry taint [52:84], [158:190].

Design (TD-Gammon lineage, which the paper already cites):
  - The network scores a STATE-ACTION pair: encode(pre_board, throw, post_board).
  - The agent scores every candidate allocation and plays the best.
  - Training target is the final game outcome from the mover's perspective
    (+1 win, -1 loss, 0 draw) - Monte Carlo value regression on self-play.
  - Rounds alternate: generate games with the current net, train, repeat.

Usage:
  python tab_rl.py --rounds 8 --games 400 --workers 55 --out checkpoints_rl
"""

import argparse
import os
import random
import time

import numpy as np

import tab_env
from tab_game import Board, TabRules, StickDice, TurnAllocation, PLAYER_1
from tab_look import get_candidate_allocations
from tab_mcts import apply_allocation, resolve_freeing, to_turn_allocation

STATE_DIM = tab_env.STATE_DIM


# ============================================================================
# Network
# ============================================================================

def make_net(hidden=(256, 128, 64)):
    import torch.nn as nn
    layers, prev = [], STATE_DIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
        prev = h
    layers += [nn.Linear(prev, 1), nn.Tanh()]
    return nn.Sequential(*layers)


# ============================================================================
# Agent
# ============================================================================

class NetAgent:
    """Scores each candidate allocation with the value net; plays the best."""

    def __init__(self, net=None, epsilon=0.0, enum_threshold=2000,
                   sample_n=80, max_candidates=48, columns=8, device='cpu'):
        self.net = net
        self.epsilon = epsilon
        self.enum_threshold = enum_threshold
        self.sample_n = sample_n
        self.max_candidates = max_candidates
        self.columns = columns
        self.device = device
        self.rules = TabRules(columns)
        self.turn = 0
        self.trace = []          # (state_vector, player) for training

    def _candidates(self, board, player, values, movable):
        acts = get_candidate_allocations(
            values, movable, player, board, self.rules,
            enum_threshold=self.enum_threshold, sample_n=self.sample_n)
        if len(acts) > self.max_candidates:
            acts = random.sample(acts, self.max_candidates)
        return acts

    def choose_allocation(self, board, player, throw_values,
                            movable_soldiers, rules):
        self.turn += 1
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        acts = self._candidates(board, player, throw_values, movable_soldiers)
        if not acts:
            return TurnAllocation()

        feats = np.empty((len(acts), STATE_DIM), dtype=np.float32)
        for i, a in enumerate(acts):
            post = board.clone()
            apply_allocation(post, a, movable_soldiers, player, rules)
            feats[i] = tab_env.encode_state(board, throw_values, post,
                                              self.turn, player, rules)

        if self.net is None or random.random() < self.epsilon:
            idx = random.randrange(len(acts))
        else:
            import torch
            with torch.no_grad():
                x = torch.from_numpy(feats).to(self.device)
                scores = self.net(x).squeeze(-1).cpu().numpy()
            idx = int(np.argmax(scores))

        self.trace.append((feats[idx].copy(), player))
        return to_turn_allocation(acts[idx], movable_soldiers)


# ============================================================================
# Self-play game generation
# ============================================================================

def play_selfplay_game(net, seed, epsilon, device='cpu', max_turns=300,
                         columns=8):
    """Both sides use the same net. Returns list of (state, target)."""
    random.seed(seed)
    board = Board(columns)
    rules = TabRules(columns)
    a1 = NetAgent(net, epsilon=epsilon, columns=columns, device=device)
    a2 = NetAgent(net, epsilon=epsilon, columns=columns, device=device)

    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        agent = a1 if player == PLAYER_1 else a2
        seq = StickDice.throw_turn()
        remaining = resolve_freeing(board, player, seq, rules)
        movable = rules.get_movable_soldiers(board, player, remaining)
        if movable and remaining:
            alloc = agent.choose_allocation(board, player, remaining,
                                              movable, rules)
            # re-derive the chosen allocation lists from the TurnAllocation
            lists = [[] for _ in movable]
            pos_index = {p: i for i, p in enumerate(movable)}
            for sm in alloc.soldier_moves:
                if sm.from_pos in pos_index:
                    lists[pos_index[sm.from_pos]] = list(sm.values_used)
            apply_allocation(board, lists, movable, player, rules)
        if board.is_game_over() is not None:
            break
        turn += 1

    winner = board.is_game_over()
    if winner is None:
        p1, p2 = board.count_alive(1), board.count_alive(-1)
        winner = 1 if p1 > p2 else (-1 if p2 > p1 else 0)

    out = []
    for agent in (a1, a2):
        for state, player in agent.trace:
            target = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            out.append((state, target))
    return out, winner


# ============================================================================
# Parallel generation
# ============================================================================

_W = {}


def _worker_init(ckpt, device):
    import torch
    net = make_net()
    if ckpt and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.eval()
    _W['net'] = net
    _W['device'] = device


def _worker_game(args):
    seed, epsilon = args
    try:
        return play_selfplay_game(_W['net'], seed, epsilon, _W['device'])
    except Exception as e:                       # keep one bad game from
        return [], 0                              # killing the round


def generate(ckpt, n_games, workers, seed0, epsilon):
    import multiprocessing as mp
    args = [(seed0 + i, epsilon) for i in range(n_games)]
    data, wins = [], {1: 0, -1: 0, 0: 0}
    with mp.Pool(processes=workers, initializer=_worker_init,
                   initargs=(ckpt, 'cpu')) as pool:
        for rows, w in pool.imap_unordered(_worker_game, args, chunksize=2):
            data.extend(rows)
            wins[w] = wins.get(w, 0) + 1
    return data, wins


# ============================================================================
# Training
# ============================================================================

def train(net, data, device, epochs=6, batch=1024, lr=1e-3):
    import torch
    import torch.nn as nn
    X = torch.from_numpy(np.stack([d[0] for d in data]))
    y = torch.from_numpy(np.array([d[1] for d in data], dtype=np.float32))
    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    last = 0.0
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = net(xb).squeeze(-1)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        last = tot / max(n, 1)
        print(f'    epoch {ep+1}/{epochs}  mse={last:.4f}', flush=True)
    net.eval()
    return last


def evaluate(net, n_games, device, opponent='GA-Original', seed0=777000):
    """Head-to-head against a reference agent, side-swapped."""
    from tab_ai import GAAgent
    from tab_runner import play_one_game
    wins = 0
    for g in range(n_games):
        opp = GAAgent('beginner', fitness_type='original')
        me = NetAgent(net, epsilon=0.0, device=device)
        if g % 2 == 0:
            r = play_one_game(me, opp, seed=seed0 + g)
            wins += (r['winner'] == 1)
        else:
            r = play_one_game(opp, me, seed=seed0 + g)
            wins += (r['winner'] == -1)
    return wins / n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--workers', type=int, default=55)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--eval-games', type=int, default=60)
    ap.add_argument('--out', type=str, default='checkpoints_rl')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, 'net.pt')

    net = make_net()
    torch.save(net.state_dict(), ckpt)
    print(f'device={device}  params={sum(p.numel() for p in net.parameters())}',
            flush=True)

    history = []
    for rnd in range(1, args.rounds + 1):
        eps = max(0.05, 0.50 * (1.0 - (rnd - 1) / max(args.rounds - 1, 1)))
        t0 = time.time()
        print(f'\n=== round {rnd}/{args.rounds}  epsilon={eps:.2f} ===',
                flush=True)
        data, wins = generate(ckpt, args.games, args.workers,
                                seed0=rnd * 100000, epsilon=eps)
        gen_s = time.time() - t0
        print(f'  generated {len(data)} samples from {args.games} games '
                f'in {gen_s:.0f}s  (P1 {wins.get(1,0)} / P2 {wins.get(-1,0)} / '
                f'draw {wins.get(0,0)})', flush=True)
        if not data:
            print('  no data, aborting'); break
        mse = train(net, data, device, epochs=args.epochs)
        torch.save(net.state_dict(), ckpt)
        net_cpu = make_net()
        net_cpu.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        net_cpu.eval()
        wr = evaluate(net_cpu, args.eval_games, 'cpu')
        print(f'  round {rnd}: mse={mse:.4f}  vs GA-Original = {wr*100:.1f}%  '
                f'({time.time()-t0:.0f}s total)', flush=True)
        history.append({'round': rnd, 'mse': mse, 'winrate': wr,
                          'samples': len(data)})
        torch.save(net.state_dict(),
                     os.path.join(args.out, f'net_round{rnd}.pt'))
        import json
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print('\n=== FINAL ===', flush=True)
    for h in history:
        print(f"  round {h['round']}: vs GA-Original {h['winrate']*100:5.1f}%  "
                f"mse={h['mse']:.4f}  samples={h['samples']}", flush=True)


if __name__ == '__main__':
    main()
