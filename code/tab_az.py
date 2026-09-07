"""
tab_az.py - AlphaZero-style learned agent for tab.

Why this design, after two failures
-----------------------------------
tab_rl.py  (outcome label)  : MSE 0.88->0.73, win rate 13%->5%.
tab_rl2.py (tactical label) : MSE 0.06, win rate 3.3%.

Both regress a scalar on the action actually taken, so they never observe the
counterfactual - what the OTHER candidates at that decision were worth. With a
weak policy they only ever visit bad states and learn an accurate map of bad
play. No label change fixes that; the signal has to compare candidates.

AlphaZero's answer, adapted to a variable-size action space:

  * At each decision, MCTS (the flat agent validated in tab_mcts.py) searches
    the candidate allocations and returns VISIT COUNTS.
  * The network scores each candidate from its 191-dim (pre, throw, post)
    encoding, producing one logit per candidate. Softmax WITHIN the decision
    gives a distribution over that decision's candidates, so the action space
    can vary in size from turn to turn.
  * Policy loss = cross-entropy against the MCTS visit distribution.
    Value loss  = MSE against the final game outcome.
  * Search is the policy-improvement operator; the network distils it and then
    plays without search, which is what makes the final agent fast.

This gives the per-decision comparative signal the two regression runs lacked.

Usage:
  python tab_az.py --rounds 8 --games 150 --sims 600 --workers 50
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

import tab_env
from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_look import get_candidate_allocations
from tab_mcts import (apply_allocation, resolve_freeing, rollout,
                        to_turn_allocation, ROLLOUT_POLICIES)

STATE_DIM = tab_env.STATE_DIM
MAX_CAND = 16          # candidates kept per decision (pad/mask to this)


# ============================================================================
# Network: shared trunk, one scalar policy logit + one value per candidate
# ============================================================================

def make_az_net(hidden=(256, 128)):
    import torch
    import torch.nn as nn

    class AZNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], STATE_DIM
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.policy = nn.Linear(prev, 1)
            self.value = nn.Linear(prev, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.policy(z).squeeze(-1), torch.tanh(self.value(z)).squeeze(-1)

    return AZNet()


# ============================================================================
# MCTS search that reports visit counts
# ============================================================================

def mcts_visits(board, player, values, movable, rules, n_sims=600,
                  rollout_name='heavy', enum_threshold=2000, sample_n=80):
    """Flat Monte Carlo with UCB1; returns (candidates, visits, encodings)."""
    acts = get_candidate_allocations(values, movable, player, board, rules,
                                       enum_threshold=enum_threshold,
                                       sample_n=sample_n)
    if not acts:
        return [], None, None
    if len(acts) > MAX_CAND:
        acts = random.sample(acts, MAX_CAND)

    policy = ROLLOUT_POLICIES[rollout_name]
    k = len(acts)
    children, feats = [], np.empty((k, STATE_DIM), dtype=np.float32)
    for i, a in enumerate(acts):
        child = board.clone()
        apply_allocation(child, a, movable, player, rules)
        children.append(child)
        feats[i] = tab_env.encode_state(board, values, child, 0, player, rules)

    if k == 1:
        return acts, np.array([1.0], dtype=np.float32), feats

    visits = [0] * k
    wins = [0.0] * k
    budget = max(n_sims, k)
    c_uct = math.sqrt(2)
    for t in range(budget):
        if t < k:
            arm = t
        else:
            logN = math.log(t)
            arm, best = 0, float('-inf')
            for i in range(k):
                u = wins[i] / visits[i] + c_uct * math.sqrt(logN / visits[i])
                if u > best:
                    best, arm = u, i
        term = children[arm].is_game_over()
        w = term if term is not None else rollout(
            children[arm], -player, rules, policy=policy, max_turns=200)
        visits[arm] += 1
        wins[arm] += 1.0 if w == player else (0.5 if w == 0 else 0.0)

    v = np.array(visits, dtype=np.float32)
    return acts, v / v.sum(), feats


# ============================================================================
# Agent: plays by the network's policy logits, no search
# ============================================================================

class AZAgent:
    """Scores candidates with the policy head and plays the argmax."""

    def __init__(self, net=None, temperature=0.0, enum_threshold=2000,
                   sample_n=80, columns=8, device='cpu'):
        self.net = net
        self.temperature = temperature
        self.enum_threshold = enum_threshold
        self.sample_n = sample_n
        self.rules = TabRules(columns)
        self.device = device

    def choose_allocation(self, board, player, throw_values,
                            movable_soldiers, rules):
        from tab_game import TurnAllocation
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        acts = get_candidate_allocations(
            throw_values, movable_soldiers, player, board, rules,
            enum_threshold=self.enum_threshold, sample_n=self.sample_n)
        if not acts:
            return TurnAllocation()
        if len(acts) > MAX_CAND:
            acts = random.sample(acts, MAX_CAND)
        if len(acts) == 1 or self.net is None:
            return to_turn_allocation(random.choice(acts) if self.net is None
                                        else acts[0], movable_soldiers)

        feats = np.empty((len(acts), STATE_DIM), dtype=np.float32)
        for i, a in enumerate(acts):
            post = board.clone()
            apply_allocation(post, a, movable_soldiers, player, rules)
            feats[i] = tab_env.encode_state(board, throw_values, post, 0,
                                              player, rules)
        import torch
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(feats).to(self.device))
            logits = logits.cpu().numpy()
        if self.temperature > 0:
            p = np.exp((logits - logits.max()) / self.temperature)
            idx = int(np.random.choice(len(acts), p=p / p.sum()))
        else:
            idx = int(np.argmax(logits))
        return to_turn_allocation(acts[idx], movable_soldiers)


# ============================================================================
# Self-play with search
# ============================================================================

def selfplay_game(net, seed, n_sims, temperature=1.0, device='cpu',
                    max_turns=300, columns=8):
    """Returns list of (feats[k,191], pi[k], player) plus the winner."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    board = Board(columns)
    rules = TabRules(columns)
    recs = []
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        remaining = resolve_freeing(board, player, seq, rules)
        movable = rules.get_movable_soldiers(board, player, remaining)
        if movable and remaining:
            acts, pi, feats = mcts_visits(board, player, remaining, movable,
                                            rules, n_sims=n_sims)
            if acts:
                recs.append((feats, pi, player))
                if temperature > 0 and len(acts) > 1:
                    idx = int(np.random.choice(len(acts), p=pi))
                else:
                    idx = int(np.argmax(pi))
                apply_allocation(board, acts[idx], movable, player, rules)
        if board.is_game_over() is not None:
            break
        turn += 1

    winner = board.is_game_over()
    if winner is None:
        p1, p2 = board.count_alive(1), board.count_alive(-1)
        winner = 1 if p1 > p2 else (-1 if p2 > p1 else 0)
    return recs, winner


_W = {}


def _init(ckpt, device, n_sims):
    _W['device'] = device
    _W['n_sims'] = n_sims


def _game(args):
    seed, temp = args
    try:
        return selfplay_game(None, seed, _W['n_sims'], temperature=temp,
                               device=_W['device'])
    except Exception:
        return [], 0


def generate(ckpt, n_games, workers, seed0, n_sims, temperature):
    import multiprocessing as mp
    args = [(seed0 + i, temperature) for i in range(n_games)]
    out, wins = [], {1: 0, -1: 0, 0: 0}
    with mp.Pool(processes=workers, initializer=_init,
                   initargs=(ckpt, 'cpu', n_sims)) as pool:
        for recs, w in pool.imap_unordered(_game, args, chunksize=1):
            for feats, pi, player in recs:
                target_v = 0.0 if w == 0 else (1.0 if w == player else -1.0)
                out.append((feats, pi, target_v))
            wins[w] = wins.get(w, 0) + 1
    return out, wins


# ============================================================================
# Training
# ============================================================================

def train_az(net, data, device, epochs=8, batch=256, lr=1e-3):
    import torch
    import torch.nn.functional as F
    n = len(data)
    X = np.zeros((n, MAX_CAND, STATE_DIM), dtype=np.float32)
    P = np.zeros((n, MAX_CAND), dtype=np.float32)
    M = np.zeros((n, MAX_CAND), dtype=np.float32)
    V = np.zeros((n,), dtype=np.float32)
    for i, (feats, pi, v) in enumerate(data):
        k = len(pi)
        X[i, :k] = feats
        P[i, :k] = pi
        M[i, :k] = 1.0
        V[i] = v
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X),
                                          torch.from_numpy(P),
                                          torch.from_numpy(M),
                                          torch.from_numpy(V))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    last = (0.0, 0.0)
    for ep in range(epochs):
        pl_sum, vl_sum, cnt = 0.0, 0.0, 0
        for xb, pb, mb, vb in dl:
            xb, pb, mb, vb = (xb.to(device), pb.to(device),
                                mb.to(device), vb.to(device))
            b, k, d = xb.shape
            logits, values = net(xb.reshape(b * k, d))
            logits = logits.reshape(b, k)
            values = values.reshape(b, k)
            logits = logits.masked_fill(mb == 0, -1e9)
            logp = torch.log_softmax(logits, dim=1)
            ploss = -(pb * logp).sum(dim=1).mean()
            # value head trained on the visited (highest-pi) candidate
            best = pb.argmax(dim=1)
            vpred = values.gather(1, best.unsqueeze(1)).squeeze(1)
            vloss = F.mse_loss(vpred, vb)
            loss = ploss + vloss
            opt.zero_grad(); loss.backward(); opt.step()
            pl_sum += ploss.item() * b; vl_sum += vloss.item() * b; cnt += b
        last = (pl_sum / cnt, vl_sum / cnt)
        print(f'    epoch {ep+1}/{epochs}  policy={last[0]:.4f} '
                f'value={last[1]:.4f}', flush=True)
    net.eval()
    return last


def evaluate_az(net, n_games, device, seed0=555000):
    from tab_ai import GAAgent
    from tab_runner import play_one_game
    wins = 0
    for g in range(n_games):
        opp = GAAgent('beginner', fitness_type='original')
        me = AZAgent(net, temperature=0.0, device=device)
        if g % 2 == 0:
            r = play_one_game(me, opp, seed=seed0 + g); wins += (r['winner'] == 1)
        else:
            r = play_one_game(opp, me, seed=seed0 + g); wins += (r['winner'] == -1)
    return wins / n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--games', type=int, default=150)
    ap.add_argument('--sims', type=int, default=600)
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--eval-games', type=int, default=60)
    ap.add_argument('--out', type=str, default='checkpoints_az')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, 'az.pt')
    net = make_az_net()
    torch.save(net.state_dict(), ckpt)
    print(f'device={device}  params={sum(p.numel() for p in net.parameters())}'
            f'  sims={args.sims}  MAX_CAND={MAX_CAND}', flush=True)

    pool_data, history, best = [], [], -1.0
    for rnd in range(1, args.rounds + 1):
        temp = 1.0 if rnd <= args.rounds // 2 else 0.5
        t0 = time.time()
        print(f'\n=== round {rnd}/{args.rounds}  temp={temp} ===', flush=True)
        data, wins = generate(ckpt, args.games, args.workers,
                                seed0=rnd * 50000 + 3, n_sims=args.sims,
                                temperature=temp)
        if not data:
            print('  no data'); break
        pool_data.extend(data)
        pool_data = pool_data[-120000:]         # replay window
        print(f'  {len(data)} decisions in {time.time()-t0:.0f}s '
                f'(pool {len(pool_data)})', flush=True)
        pl, vl = train_az(net, pool_data, device, epochs=args.epochs)
        torch.save(net.state_dict(), ckpt)
        cpu = make_az_net()
        cpu.load_state_dict(torch.load(ckpt, map_location='cpu',
                                         weights_only=True))
        cpu.eval()
        wr = evaluate_az(cpu, args.eval_games, 'cpu')
        flag = ''
        if wr > best:
            best = wr
            torch.save(net.state_dict(), os.path.join(args.out, 'best.pt'))
            flag = '  <-- best'
        print(f'  round {rnd}: policy={pl:.4f} value={vl:.4f}  '
                f'vs GA-Original = {wr*100:.1f}%  ({time.time()-t0:.0f}s){flag}',
                flush=True)
        history.append({'round': rnd, 'policy': pl, 'value': vl,
                          'winrate': wr, 'decisions': len(data)})
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print('\n=== FINAL (tab_az) ===', flush=True)
    for h in history:
        print(f"  round {h['round']:2d}: vs GA-Original {h['winrate']*100:5.1f}%"
                f"   policy={h['policy']:.4f}  value={h['value']:.4f}", flush=True)
    print(f'  best = {best*100:.1f}%', flush=True)


if __name__ == '__main__':
    main()
