"""
tab_tdlook.py - TLA-NN: TLA's one-ply lookahead with the LEARNED evaluator.

This is the decisive comparison for Section IX. TLA-S and TLA-NN run the
*identical* search - same candidate generation, same three retained opponent
throw sequences, same renormalization, same best-of-candidates opponent reply,
same fixed fork picker. The only difference is the evaluator:

    TLA-S   handcrafted LookEvaluator with the repaired stacking term
    TLA-NN  a value network trained by TD(lambda) self-play (tab_td.py)

Holding the search fixed and swapping only the evaluator isolates exactly the
question the paper makes a claim about, and it is TD-Gammon's architecture:
learned positional evaluation plus shallow search.
"""

import random
from typing import List

import numpy as np

import tab_env
from tab_game import Board, TabRules, TurnAllocation, PLAYER_1
from tab_look import get_candidate_allocations, LIKELY_OPPONENT_SEQUENCES
from tab_mcts import apply_allocation, resolve_freeing, to_turn_allocation
from tab_td import make_value_net, encode, SDIM


def load_net(path):
    """Load a value net, inferring the hidden layer sizes from the checkpoint.

    tab_td.py can be run with --hidden, so the 100k-game run (256,128,64) and
    the 500k-game run (512,256,128) have different shapes. Inferring avoids a
    silent size mismatch when a checkpoint is swapped in.
    """
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    hidden = []
    for k in sorted(sd.keys()):
        if k.endswith('.weight') and sd[k].dim() == 2:
            hidden.append(sd[k].shape[0])
    hidden = tuple(hidden[:-1])          # last Linear is the scalar output
    net = make_value_net(hidden)
    net.load_state_dict(sd)
    net.eval()
    return net


class TDLookAgent:
    """One-ply lookahead over a learned value function."""

    def __init__(self, net, lookahead_seqs=3, sample_n=80, opp_n_candidates=15,
                   enum_threshold=10000, max_candidates=24, use_lookahead=True,
                   columns=8):
        self.net = net
        self.n_seq = lookahead_seqs
        self.sample_n = sample_n
        self.opp_n = opp_n_candidates
        self.enum_threshold = enum_threshold
        self.max_candidates = max_candidates
        self.use_lookahead = use_lookahead
        self.rules = TabRules(columns)

    def _v(self, boards, player, rules):
        """Batched value of each board from `player`'s perspective."""
        import torch
        X = np.empty((len(boards), SDIM), dtype=np.float32)
        for i, b in enumerate(boards):
            X[i] = encode(b, player, rules)
        with torch.no_grad():
            return self.net(torch.from_numpy(X)).squeeze(-1).numpy()

    def _lookahead(self, board_after, me, rules):
        """Expected value after a modelled opponent reply, renormalized over
        the retained sequences - mirrors tab_look.lookahead_score exactly."""
        opp = -me
        term = board_after.is_game_over()
        if term is not None:
            return 1e6 if term == me else -1e6
        seqs = LIKELY_OPPONENT_SEQUENCES[:self.n_seq]
        total_p = sum(p for _, p in seqs)
        if total_p == 0:
            return float(self._v([board_after], me, rules)[0])
        expected = 0.0
        for seq, prob in seqs:
            ob = board_after.clone()
            remaining = resolve_freeing(ob, opp, list(seq), rules)
            movable = rules.get_movable_soldiers(ob, opp, remaining)
            if not movable or not remaining:
                expected += prob * float(self._v([ob], me, rules)[0])
                continue
            cands = get_candidate_allocations(remaining, movable, opp, ob, rules,
                                                enum_threshold=2000,
                                                sample_n=self.opp_n)
            if len(cands) > self.opp_n:
                cands = random.sample(cands, self.opp_n)
            sims = []
            for a in cands:
                s = ob.clone()
                apply_allocation(s, a, movable, opp, rules)
                sims.append(s)
            # opponent picks the reply best for THEM (self-play assumption)
            opp_vals = self._v(sims, opp, rules)
            best = sims[int(np.argmax(opp_vals))]
            expected += prob * float(self._v([best], me, rules)[0])
        return expected / total_p

    def choose_allocation(self, board, player, throw_values,
                            movable_soldiers, rules):
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        acts = get_candidate_allocations(throw_values, movable_soldiers, player,
                                           board, rules,
                                           enum_threshold=self.enum_threshold,
                                           sample_n=self.sample_n)
        if not acts:
            return TurnAllocation()
        if len(acts) > self.max_candidates:
            acts = random.sample(acts, self.max_candidates)
        if len(acts) == 1:
            return to_turn_allocation(acts[0], movable_soldiers)

        sims = []
        for a in acts:
            s = board.clone()
            apply_allocation(s, a, movable_soldiers, player, rules)
            sims.append(s)

        if not self.use_lookahead:
            vals = self._v(sims, player, rules)
            return to_turn_allocation(acts[int(np.argmax(vals))],
                                        movable_soldiers)

        best_i, best_s = 0, float('-inf')
        for i, s in enumerate(sims):
            sc = self._lookahead(s, player, rules)
            if sc > best_s:
                best_s, best_i = sc, i
        return to_turn_allocation(acts[best_i], movable_soldiers)
