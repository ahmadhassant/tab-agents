"""
tab_distil.py - behaviour-clone TLA-S into the network that failed at RL.

Purpose (this is a diagnostic, not an attempt at a strong agent)
---------------------------------------------------------------
Three learning designs failed: outcome-labelled self-play, tactical-labelled
self-play, and (by measurement, before building it) AlphaZero-style distillation
of MCTS. Section 6 of Baseline_Results_Sep02.md argues the common cause is that
rollout evaluation carries almost no signal in tab, not that the network or the
191-dim state are inadequate.

This isolates that claim. Same architecture, same encoding, same candidate
generator - only the teacher changes, from search to TLA-S. If the student now
learns, the bottleneck was the signal. If it still fails, the representation is
at fault and we must report that instead.

Usage:
  python tab_distil.py --games 400 --workers 50 --epochs 20
"""

import argparse
import json
import os
import random
import time

import numpy as np

import tab_env
from tab_game import Board, TabRules, StickDice, PLAYER_1
from tab_look import LookAgent, get_candidate_allocations
from tab_mcts import apply_allocation, resolve_freeing
from tab_az import make_az_net, MAX_CAND, STATE_DIM, AZAgent


def teacher():
    return LookAgent(lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
                       use_rules=True, binary_hide=True,
                       use_continuous_phase=False, stacking_mode='simple')


def _match(alloc, acts, movable):
    """Index of the candidate matching the teacher's TurnAllocation."""
    lists = [[] for _ in movable]
    idx = {p: i for i, p in enumerate(movable)}
    for sm in alloc.soldier_moves:
        if sm.from_pos in idx:
            lists[idx[sm.from_pos]] = sorted(sm.values_used)
    for i, a in enumerate(acts):
        if [sorted(x) for x in a] == lists:
            return i
    return -1


def collect_game(seed, columns=8, max_turns=300):
    """Play TLA-S vs a mix of opponents; record (candidates, teacher choice)."""
    random.seed(seed)
    from tab_ai import GAAgent, RandomAgent
    board = Board(columns)
    rules = TabRules(columns)
    tea = teacher()
    # vary the opponent so the state distribution is not degenerate
    r = seed % 3
    opp = (GAAgent('beginner', fitness_type='original') if r == 0 else
             GAAgent('beginner', fitness_type='expert') if r == 1 else
             RandomAgent())
    recs = []
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else -PLAYER_1
        seq = StickDice.throw_turn()
        remaining = resolve_freeing(board, player, seq, rules)
        movable = rules.get_movable_soldiers(board, player, remaining)
        if movable and remaining:
            if player == PLAYER_1:
                acts = get_candidate_allocations(remaining, movable, player,
                                                   board, rules,
                                                   enum_threshold=2000,
                                                   sample_n=80)
                if len(acts) > MAX_CAND:
                    acts = random.sample(acts, MAX_CAND)
                alloc = tea.choose_allocation(board, player, remaining,
                                                movable, rules)
                pick = _match(alloc, acts, movable)
                if pick < 0 and acts:
                    # teacher's choice was sampled out; skip this decision
                    pick = -1
                if pick >= 0 and len(acts) > 1:
                    feats = np.empty((len(acts), STATE_DIM), dtype=np.float32)
                    for i, a in enumerate(acts):
                        post = board.clone()
                        apply_allocation(post, a, movable, player, rules)
                        feats[i] = tab_env.encode_state(board, remaining, post,
                                                          0, player, rules)
                    recs.append((feats, pick))
                lists = [[] for _ in movable]
                idx = {p: i for i, p in enumerate(movable)}
                for sm in alloc.soldier_moves:
                    if sm.from_pos in idx:
                        lists[idx[sm.from_pos]] = list(sm.values_used)
                apply_allocation(board, lists, movable, player, rules)
            else:
                alloc = opp.choose_allocation(board, player, remaining,
                                                movable, rules)
                lists = [[] for _ in movable]
                idx = {p: i for i, p in enumerate(movable)}
                for sm in alloc.soldier_moves:
                    if sm.from_pos in idx:
                        lists[idx[sm.from_pos]] = list(sm.values_used)
                apply_allocation(board, lists, movable, player, rules)
        if board.is_game_over() is not None:
            break
        turn += 1
    return recs


def _game(seed):
    try:
        return collect_game(seed)
    except Exception:
        return []


def collect(n_games, workers, seed0):
    import multiprocessing as mp
    data = []
    with mp.Pool(processes=workers) as pool:
        for recs in pool.imap_unordered(_game, range(seed0, seed0 + n_games),
                                          chunksize=2):
            data.extend(recs)
    return data


def train(net, data, device, epochs=20, batch=256, lr=1e-3, val_frac=0.1):
    import torch
    n = len(data)
    X = np.zeros((n, MAX_CAND, STATE_DIM), dtype=np.float32)
    M = np.zeros((n, MAX_CAND), dtype=np.float32)
    Y = np.zeros((n,), dtype=np.int64)
    for i, (feats, pick) in enumerate(data):
        k = len(feats)
        X[i, :k] = feats
        M[i, :k] = 1.0
        Y[i] = pick
    idx = np.random.permutation(n)
    cut = int(n * (1 - val_frac))
    tr, va = idx[:cut], idx[cut:]

    def loader(ix, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X[ix]),
                                              torch.from_numpy(M[ix]),
                                              torch.from_numpy(Y[ix]))
        return torch.utils.data.DataLoader(ds, batch_size=batch,
                                             shuffle=shuffle)

    dtr, dva = loader(tr, True), loader(va, False)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        net.train()
        tot, cor, cnt = 0.0, 0, 0
        for xb, mb, yb in dtr:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            b, k, d = xb.shape
            logits, _ = net(xb.reshape(b * k, d))
            logits = logits.reshape(b, k).masked_fill(mb == 0, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * b
            cor += (logits.argmax(1) == yb).sum().item(); cnt += b
        net.eval()
        vc, vn = 0, 0
        with torch.no_grad():
            for xb, mb, yb in dva:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                b, k, d = xb.shape
                logits, _ = net(xb.reshape(b * k, d))
                logits = logits.reshape(b, k).masked_fill(mb == 0, -1e9)
                vc += (logits.argmax(1) == yb).sum().item(); vn += b
        hist.append({'epoch': ep + 1, 'loss': tot / cnt,
                       'train_acc': cor / cnt, 'val_acc': vc / max(vn, 1)})
        print(f"    epoch {ep+1}/{epochs}  loss={tot/cnt:.4f}  "
                f"train_acc={cor/cnt:.3f}  val_acc={vc/max(vn,1):.3f}",
                flush=True)
    return hist


def evaluate(net, n_games, device, fitness='original', seed0=888000):
    from tab_ai import GAAgent
    from tab_runner import play_one_game
    wins = 0
    for g in range(n_games):
        opp = GAAgent('beginner', fitness_type=fitness)
        me = AZAgent(net, temperature=0.0, device=device)
        if g % 2 == 0:
            r = play_one_game(me, opp, seed=seed0 + g); wins += (r['winner'] == 1)
        else:
            r = play_one_game(opp, me, seed=seed0 + g); wins += (r['winner'] == -1)
    return wins / n_games


def evaluate_vs_teacher(net, n_games, device, seed0=999000):
    from tab_runner import play_one_game
    wins = 0
    for g in range(n_games):
        me = AZAgent(net, temperature=0.0, device=device)
        tea = teacher()
        if g % 2 == 0:
            r = play_one_game(me, tea, seed=seed0 + g); wins += (r['winner'] == 1)
        else:
            r = play_one_game(tea, me, seed=seed0 + g); wins += (r['winner'] == -1)
    return wins / n_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--workers', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--eval-games', type=int, default=100)
    ap.add_argument('--out', type=str, default='checkpoints_distil')
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    print(f'device={device}  teacher=TLA-S', flush=True)

    t0 = time.time()
    data = collect(args.games, args.workers, seed0=4242)
    print(f'collected {len(data)} labelled decisions from {args.games} games '
            f'in {time.time()-t0:.0f}s', flush=True)
    if len(data) < 500:
        print('too little data'); return

    net = make_az_net()
    hist = train(net, data, device, epochs=args.epochs)
    torch.save(net.state_dict(), os.path.join(args.out, 'distil.pt'))

    cpu = make_az_net()
    cpu.load_state_dict(torch.load(os.path.join(args.out, 'distil.pt'),
                                     map_location='cpu', weights_only=True))
    cpu.eval()
    wr_ga = evaluate(cpu, args.eval_games, 'cpu', 'original')
    wr_gx = evaluate(cpu, args.eval_games, 'cpu', 'expert')
    wr_te = evaluate_vs_teacher(cpu, args.eval_games, 'cpu')

    print('\n=== DISTILLATION RESULT ===', flush=True)
    print(f'  imitation val accuracy : {hist[-1]["val_acc"]*100:.1f}%', flush=True)
    print(f'  vs GA-Original         : {wr_ga*100:.1f}%', flush=True)
    print(f'  vs GA-Expert           : {wr_gx*100:.1f}%', flush=True)
    print(f'  vs TLA-S (its teacher) : {wr_te*100:.1f}%', flush=True)
    with open(os.path.join(args.out, 'result.json'), 'w') as f:
        json.dump({'history': hist, 'vs_ga_original': wr_ga,
                     'vs_ga_expert': wr_gx, 'vs_teacher': wr_te}, f, indent=2)


if __name__ == '__main__':
    main()
