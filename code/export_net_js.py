"""
export_net_js.py - export the TD(lambda) value net for the browser build.

The full-precision weights are 4.9 MB of JSON, too heavy to inline in a
self-contained HTML page. We quantise to int8 with a per-output-row scale
(standard symmetric quantisation) and base64 the buffer, which brings it to
roughly 300 KB. Quantisation is only acceptable if it does not change which
allocation the agent picks, so this script MEASURES that rather than assuming
it: it reports the value error on sampled positions and, more importantly, the
argmax agreement between the full and quantised networks over real decisions.

Writes:
  web_bot/tab_net.json        quantised weights + architecture
  web_bot/tab_net_tests.json  fixtures (state -> exact 124-dim encoding, value)
"""
import argparse
import base64
import json
import os
import random

import numpy as np

import tab_env
from tab_game import Board, TabRules, PLAYER_1
import tab_mcts
from tab_look import get_candidate_allocations
from tab_td import make_value_net, encode, SDIM


def quantise(W):
    """Per-output-row symmetric int8. Returns (int8 bytes, scales)."""
    scales = np.abs(W).max(axis=1) / 127.0
    scales[scales == 0] = 1e-8
    q = np.rint(W / scales[:, None]).astype(np.int8)
    return q, scales


def dequantise(q, scales):
    return q.astype(np.float32) * scales[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='../03_Results/checkpoints_td2/best.pt')
    ap.add_argument('--out', default='web_bot')
    ap.add_argument('--tests', type=int, default=40)
    ap.add_argument('--argmax-checks', type=int, default=400)
    ap.add_argument('--dtype', default='f16', choices=['i8', 'f16', 'f32'],
                    help='weight precision for the browser build')
    args = ap.parse_args()

    import torch
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    hidden = []
    for k in sorted(sd.keys()):
        if k.endswith('.weight') and sd[k].dim() == 2:
            hidden.append(sd[k].shape[0])
    hidden = tuple(hidden[:-1])
    net = make_value_net(hidden)
    net.load_state_dict(sd)
    net.eval()
    print('architecture: %d -> %s -> 1' % (SDIM, hidden))

    layers, qlayers = [], []
    for m in net:
        if not hasattr(m, 'weight'):
            continue
        W = m.weight.detach().numpy()
        b = m.bias.detach().numpy()
        layers.append((W, b))
        if args.dtype == 'i8':
            q, sc = quantise(W)
            qlayers.append({'shape': list(W.shape), 'dtype': 'i8',
                              'q': base64.b64encode(q.tobytes()).decode('ascii'),
                              'scale': [float(x) for x in sc],
                              'b': [float(x) for x in b]})
        else:
            np_dt = np.float16 if args.dtype == 'f16' else np.float32
            qlayers.append({'shape': list(W.shape), 'dtype': args.dtype,
                              'q': base64.b64encode(W.astype(np_dt).tobytes()).decode('ascii'),
                              'b': [float(x) for x in b]})

    os.makedirs(args.out, exist_ok=True)
    meta = {'in_dim': SDIM, 'hidden': list(hidden), 'layers': qlayers,
              'activation': 'relu', 'output': 'tanh', 'quant': args.dtype}
    p = os.path.join(args.out, 'tab_net.json')
    json.dump(meta, open(p, 'w'))
    print('wrote %s (%.0f KB)' % (p, os.path.getsize(p) / 1024))

    # ---- pure-numpy forward passes, full vs quantised ----------------------
    def _load(L):
        if L['dtype'] == 'i8':
            q = np.frombuffer(base64.b64decode(L['q']), dtype=np.int8).reshape(L['shape'])
            return dequantise(q, np.array(L['scale'])), np.array(L['b'])
        dt = np.float16 if L['dtype'] == 'f16' else np.float32
        W = np.frombuffer(base64.b64decode(L['q']), dtype=dt).reshape(L['shape'])
        return W.astype(np.float32), np.array(L['b'])
    deq = [_load(L) for L in qlayers]

    def fwd(x, ls):
        for i, (W, b) in enumerate(ls):
            x = x @ W.T + b
            x = np.maximum(x, 0) if i < len(ls) - 1 else np.tanh(x)
        return x

    # ---- fixtures ----------------------------------------------------------
    rules = TabRules(8)
    cases, errs = [], []
    seed = 0
    while len(cases) < args.tests and seed < 4000:
        seed += 1
        random.seed(seed)
        b = Board(8)
        for t in range(random.randint(3, 60)):
            tab_mcts.step_turn(b, PLAYER_1 if t % 2 == 0 else -PLAYER_1,
                                 rules, tab_mcts.random_policy)
            if b.is_game_over() is not None:
                break
        if b.is_game_over() is not None:
            continue
        for player in (PLAYER_1, -PLAYER_1):
            vec = encode(b, player, rules)
            with torch.no_grad():
                v = float(net(torch.from_numpy(vec[None, :])).item())
            vq = float(fwd(vec[None, :].astype(np.float64), deq)[0, 0])
            errs.append(abs(v - vq))
            cases.append({
                'cells': list(b.cells), 'player': int(player),
                'frozen_queue_p1': list(b.frozen_queue[0]),
                'frozen_queue_p2': list(b.frozen_queue[1]),
                'frozen_at_p1': {str(k): int(x) for k, x in b.frozen_at[0].items()},
                'frozen_at_p2': {str(k): int(x) for k, x in b.frozen_at[1].items()},
                'no_reentry_p1': [int(k) for k, st in b.no_reentry[0].items() if any(st)],
                'no_reentry_p2': [int(k) for k, st in b.no_reentry[1].items() if any(st)],
                'player_started': [bool(b.player_started[0]), bool(b.player_started[1])],
                'encoding': [round(float(x), 6) for x in vec],
                'value': round(v, 6), 'value_q': round(vq, 6),
            })
            if len(cases) >= args.tests:
                break
    p2 = os.path.join(args.out, 'tab_net_tests.json')
    json.dump(cases, open(p2, 'w'))
    print('wrote %s (%d cases, %.0f KB)' % (p2, len(cases), os.path.getsize(p2)/1024))
    print('value error from quantisation: mean %.5f, max %.5f'
            % (np.mean(errs), np.max(errs)))

    # ---- the check that matters: does the CHOICE change? -------------------
    agree = tot = 0
    seed = 10000
    from tab_mcts import apply_allocation, resolve_freeing
    from tab_game import StickDice
    while tot < args.argmax_checks and seed < 20000:
        seed += 1
        random.seed(seed)
        b = Board(8)
        for t in range(random.randint(3, 50)):
            tab_mcts.step_turn(b, PLAYER_1 if t % 2 == 0 else -PLAYER_1,
                                 rules, tab_mcts.random_policy)
            if b.is_game_over() is not None:
                break
        if b.is_game_over() is not None:
            continue
        seq = StickDice.throw_turn()
        rem = resolve_freeing(b, PLAYER_1, seq, rules)
        mov = rules.get_movable_soldiers(b, PLAYER_1, rem)
        if not mov or not rem:
            continue
        acts = get_candidate_allocations(rem, mov, PLAYER_1, b, rules,
                                           enum_threshold=2000, sample_n=40)
        if len(acts) < 2:
            continue
        X = np.empty((len(acts), SDIM), dtype=np.float32)
        for i, a in enumerate(acts):
            post = b.clone()
            apply_allocation(post, a, mov, PLAYER_1, rules)
            X[i] = encode(post, PLAYER_1, rules)
        with torch.no_grad():
            vf = net(torch.from_numpy(X)).squeeze(-1).numpy()
        vq = fwd(X.astype(np.float64), deq)[:, 0]
        tot += 1
        agree += int(np.argmax(vf) == np.argmax(vq))
    print('argmax agreement full vs quantised: %d/%d = %.2f%%'
            % (agree, tot, 100.0 * agree / max(tot, 1)))


if __name__ == '__main__':
    main()
