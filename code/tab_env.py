"""
tab_env.py — Tāb Game RL Environment (v4 — Pre+Post Move State)
================================================================
State design: PRE-move board + throw + POST-move board (per candidate)

This solves both problems:
  ✓ Throw included → NN learns what moves are possible given the dice
  ✓ Post-move board differs per candidate → NN can distinguish allocations

How it works:
  Training: for each move made, record
    state = encode(pre_board, throw) + encode_delta(post_board) + [turn]
    label = K - L  (kills this turn minus losses from opponent's response)

  Inference (NNAgent): for each candidate allocation
    sim = simulate(board, candidate)
    state = encode(board, throw) + encode_delta(sim) + [turn]
    → model predicts quality label
    → pick candidate with best predicted label

State vector (191-dim):
  PRE-MOVE (124 values):
    [0:32]    board cells /8  (own=+, opp=−, mirrored for P2)
    [32:40]   own queue rank  (front=1.0, back=1/8, freed=0)
    [40:48]   opp queue rank
    [48:50]   [own_frozen/8, opp_frozen/8]
    [50:52]   allFreed flags [own, opp]
    [52:84]   no-reentry taint (32 bits, mirrored for P2)
    [84:86]   player_started [own, opp]
    [86:94]   hasActiveElsewhere per own frozen slot (8 bits)
    [94:124]  throw one-hot: 6 slots × 5 values [1,2,3,4,6]

  POST-MOVE DELTA (66 values):
    [124:156] post-move board cells /8  (same normalisation as pre)
    [156:158] post allFreed flags [own, opp]  (may change after capturing)
    [158:190] post no-reentry taint  (taint may be set after exiting enemy home)

  TURN (1 value):
    [190]     turn / MAX_TURNS

  TOTAL: 191
"""

import sys, os, random
_ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'tab_project', 'engine')
sys.path.insert(0, _ENGINE)

import numpy as np
from tab_game import (Board, TabRules, StickDice, TurnAllocation, SoldierMove,
                      PLAYER_1, PLAYER_2, DEFAULT_COLUMNS)
from tab_ai import RandomAgent, FitnessEvaluator

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLASSES   = 5
LABEL_NAMES = ['Worst(--)', 'Bad(-)', 'Neutral(0)', 'Good(+)', 'Best(++)']
MAX_TURNS   = 300
DICE_VALS   = [1, 2, 3, 4, 6]
MAX_DICE    = 6
STATE_DIM   = 191

# Mirror: A↔D rows, B↔C rows so P2's board looks geometrically like P1's
def _mirror_pos(p):
    if   p < 8:  return p + 24
    elif p < 16: return p + 8
    elif p < 24: return p - 8
    else:         return p - 24
_MIRROR = [_mirror_pos(p) for p in range(32)]

def cost_to_label(c):
    if   c <= -2: return 0
    elif c == -1: return 1
    elif c ==  0: return 2
    elif c ==  1: return 3
    else:         return 4

def _count(cells, player):
    return sum(abs(c) for c in cells if c * player > 0)

# ── Helper: mirror board cells ────────────────────────────────────────────────
def _mirror_cells(cells, player):
    """Return cells array from player's perspective (own=+, opp=−)."""
    sign = 1 if player == PLAYER_1 else -1
    if player == PLAYER_1:
        return [cells[i] * sign for i in range(32)]
    else:
        return [cells[_MIRROR[i]] * sign for i in range(32)]

# ── Pre-move features (124 dims) ──────────────────────────────────────────────
def encode_pre(board, throw, player, rules):
    """Encode pre-move board context + throw values."""
    C  = board.columns
    pi = board.player_index(player)
    oi = 1 - pi

    # Board cells (32)
    board_feat = np.array(_mirror_cells(board.cells, player),
                          dtype=np.float32) / 8.0

    # Queue ranks (8 + 8)
    def _qfeat(fq, fa, slots):
        rank_map = {pos: (len(fq)-i) for i, pos in enumerate(fq)}
        f = np.zeros(8, dtype=np.float32)
        for k, slot in enumerate(slots):
            if fa.get(slot, 0) > 0 and slot in rank_map:
                f[k] = rank_map[slot] / 8.0
        return f

    own_slots = list(range(C)) if player==PLAYER_1 else list(range(3*C,4*C))
    opp_slots = list(range(3*C,4*C)) if player==PLAYER_1 else list(range(C))
    own_q = _qfeat(board.frozen_queue[pi], board.frozen_at[pi], own_slots)
    opp_q = _qfeat(board.frozen_queue[oi], board.frozen_at[oi], opp_slots)

    # Frozen counts (2)
    fq_c = np.array([len(board.frozen_queue[pi])/8.0,
                     len(board.frozen_queue[oi])/8.0], dtype=np.float32)

    # allFreed flags (2)
    af = np.array([1.0 if rules.all_freed(player,  board) else 0.0,
                   1.0 if rules.all_freed(-player, board) else 0.0],
                  dtype=np.float32)

    # Taint (32)
    taint = np.zeros(32, dtype=np.float32)
    for pos, stack in board.no_reentry[pi].items():
        if any(stack):
            mp = pos if player==PLAYER_1 else _MIRROR[pos]
            if 0 <= mp < 32:
                taint[mp] = 1.0

    # player_started (2)
    started = np.array([float(board.player_started[pi]),
                        float(board.player_started[oi])], dtype=np.float32)

    # hasActiveElsewhere per slot (8)
    has_act = np.zeros(8, dtype=np.float32)
    if rules.has_active_elsewhere(player, board):
        for k, slot in enumerate(own_slots):
            if board.frozen_at[pi].get(slot, 0) > 0:
                has_act[k] = 1.0

    # Throw one-hot (6 slots × 5 values = 30)
    sv = sorted(throw)[:MAX_DICE] + [0]*(MAX_DICE - min(len(throw), MAX_DICE))
    throw_feat = np.zeros(MAX_DICE * len(DICE_VALS), dtype=np.float32)
    for si, v in enumerate(sv):
        if v in DICE_VALS:
            throw_feat[si*len(DICE_VALS) + DICE_VALS.index(v)] = 1.0

    return np.concatenate([board_feat, own_q, opp_q, fq_c, af,
                           taint, started, has_act, throw_feat])
    # 32+8+8+2+2+32+2+8+30 = 124

# ── Post-move features (66 dims) ─────────────────────────────────────────────
def encode_post(post_board, player, rules):
    """Encode post-move board delta — what changed after the move."""
    C  = post_board.columns
    pi = post_board.player_index(player)
    oi = 1 - pi

    # Post board cells (32)
    post_cells = np.array(_mirror_cells(post_board.cells, player),
                          dtype=np.float32) / 8.0

    # Post allFreed (2)
    post_af = np.array([1.0 if rules.all_freed(player,  post_board) else 0.0,
                        1.0 if rules.all_freed(-player, post_board) else 0.0],
                       dtype=np.float32)

    # Post taint (32)
    post_taint = np.zeros(32, dtype=np.float32)
    for pos, stack in post_board.no_reentry[pi].items():
        if any(stack):
            mp = pos if player==PLAYER_1 else _MIRROR[pos]
            if 0 <= mp < 32:
                post_taint[mp] = 1.0

    return np.concatenate([post_cells, post_af, post_taint])
    # 32+2+32 = 66

# ── Full state vector (191 dims) ──────────────────────────────────────────────
def encode_state(pre_board, throw, post_board, turn, player, rules):
    """
    Combine pre-move context + throw + post-move result.
    191-dim vector used for both training and inference.
    """
    pre  = encode_pre(pre_board, throw, player, rules)     # 124
    post = encode_post(post_board, player, rules)           #  66
    prog = np.array([min(turn, MAX_TURNS)/MAX_TURNS],
                    dtype=np.float32)                       #   1
    return np.concatenate([pre, post, prog])                # 191

# ── Simplified GA Agent ───────────────────────────────────────────────────────
class SimpleGAAgent:
    def __init__(self, columns=DEFAULT_COLUMNS, max_allocs=3000):
        from itertools import product as iprod
        self._iprod    = iprod
        self._eval     = FitnessEvaluator(columns=columns)
        self._fallback = RandomAgent()
        self._max      = max_allocs

    def choose_allocation(self, board, player, throw_values,
                          movable_soldiers, rules):
        n = len(movable_soldiers)
        if n == 0:
            return TurnAllocation()
        total = n ** len(throw_values)
        assigns = (list(self._iprod(range(n), repeat=len(throw_values)))
                   if total <= self._max
                   else [tuple(random.randrange(n) for _ in throw_values)
                         for _ in range(self._max)])
        best_ta, best_score = None, float('-inf')
        for assign in assigns:
            alloc = [[] for _ in range(n)]
            for vi, si in enumerate(assign):
                alloc[si].append(throw_values[vi])
            sim   = rules.simulate_allocation(board, alloc, movable_soldiers, player)
            score = self._eval.evaluate(sim.cells, player)
            if score > best_score:
                best_score = score
                ta = TurnAllocation()
                for si, vals in enumerate(alloc):
                    if vals:
                        ta.soldier_moves.append(
                            SoldierMove(from_pos=movable_soldiers[si],
                                        to_pos=movable_soldiers[si],
                                        values_used=vals))
                best_ta = ta
        return (best_ta or
                self._fallback.choose_allocation(board, player, throw_values,
                                                 movable_soldiers, rules))

# ── Apply Allocation ───────────────────────────────────────────────────────────
def apply_alloc(board, alloc, player, rules):
    if not alloc or not alloc.soldier_moves:
        return 0
    total_cap = 0
    pi = board.player_index(player)
    for sm in alloc.soldier_moves:
        pos = sm.from_pos
        for val in sm.values_used:
            if (val == 1 and rules.is_own_home(pos, player)
                    and board.frozen_at[pi].get(pos, 0) > 0
                    and rules.can_free(pos, player, board)):
                rules.free_specific_soldier(board, player, pos)
                continue
            dests = rules.compute_destinations(pos, val, player, board)
            if dests:
                new_pos, cap, _ = rules.apply_single_move(board, pos, val, player)
                total_cap += cap
                pos = new_pos
    return total_cap

# ── Episode Runner ─────────────────────────────────────────────────────────────
def play_episode(p1_agent, p2_agent, rules, collect_both=True):
    """
    Each record encodes:
      state  = pre_board + throw + post_board (after player's move)
      label  = cost_to_label(kills - losses_from_opponent_response)

    Training: agent made a move → record pre+throw+post → label with outcome
    Inference: for each candidate, simulate it → encode pre+throw+sim → score
    """
    board   = Board(rules.columns)
    p1_recs, p2_recs = [], []
    turn    = 0
    winner  = None

    while turn < MAX_TURNS:
        for (player, agent, recs_list) in [
                (PLAYER_1, p1_agent, p1_recs),
                (PLAYER_2, p2_agent, p2_recs)]:
            opp = -player
            should_record = (player == PLAYER_1) or collect_both

            throw_raw = StickDice.throw_turn()
            tv, _     = rules.get_throw_sequence_with_freeing(board, player, throw_raw)
            mov       = rules.get_movable_soldiers(board, player, tv) if tv else []

            if should_record:
                pre_board = board.clone()   # snapshot before move
                opp_n0    = _count(board.cells, opp)
                own_n0    = _count(board.cells, player)

            if mov and tv:
                alloc = agent.choose_allocation(board, player, tv, mov, rules)
                apply_alloc(board, alloc, player, rules)

            if should_record:
                # Post-move board is the current board after move applied
                post_board = board.clone()
                kills      = opp_n0 - _count(board.cells, opp)
                own_n1     = _count(board.cells, player)

                # Build state: pre + throw + post
                state = encode_state(pre_board, tv or [], post_board,
                                     turn, player, rules)

                if _count(board.cells, opp) == 0:
                    winner = player
                    recs_list.append(_rec(state, kills, 0, turn, player))
                    break

                # Opponent responds
                throw_o = StickDice.throw_turn()
                tv_o, _ = rules.get_throw_sequence_with_freeing(board, opp, throw_o)
                mov_o   = rules.get_movable_soldiers(board, opp, tv_o) if tv_o else []
                opp_agent = p2_agent if player==PLAYER_1 else p1_agent
                if mov_o and tv_o:
                    alloc_o = opp_agent.choose_allocation(board, opp, tv_o, mov_o, rules)
                    apply_alloc(board, alloc_o, opp, rules)

                losses = own_n1 - _count(board.cells, player)
                recs_list.append(_rec(state, kills, losses, turn, player))

                if _count(board.cells, player) == 0:
                    winner = opp
                    break
            else:
                if _count(board.cells, opp) == 0:
                    winner = player; break
                if _count(board.cells, player) == 0:
                    winner = opp; break

        if winner is not None:
            break
        turn += 1

    all_recs = p1_recs + (p2_recs if collect_both else [])
    for r in all_recs:
        r['winner'] = winner
    return all_recs

def _rec(state, kills, losses, turn, player):
    c = kills - losses
    return dict(state=state, kills=kills, losses=losses,
                cost=c, label=cost_to_label(c),
                turn=turn, player=player, winner=None)

# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from collections import Counter
    rules = TabRules()
    p1 = RandomAgent(); p2 = SimpleGAAgent()

    recs = play_episode(p1, p2, rules, collect_both=True)
    print(f'Episode: {len(recs)} records')
    print(f'State shape: {recs[0]["state"].shape}  (expect ({STATE_DIM},))')
    print(f'Labels: {dict(Counter(r["label"] for r in recs))}')
    assert recs[0]['state'].shape == (STATE_DIM,), \
        f'STATE_DIM mismatch: {recs[0]["state"].shape} vs {STATE_DIM}'

    # Verify candidates get different scores with a random model
    import torch, torch.nn.functional as F
    from tab_model import TabMoveNet
    from itertools import product as iprod

    model = TabMoveNet().eval()
    b = Board(); rules2 = TabRules()
    b.player_started=[True,True]; b.frozen_queue=[[0,1,2],[31,30,29,28,27,26,25,24]]
    b.frozen_at=[{0:1,1:1,2:1},{i:1 for i in range(24,32)}]
    b.cells=[0]*32; b.cells[8]=1; b.cells[12]=1
    tv = [1,3]; mov = rules2.get_movable_soldiers(b, PLAYER_1, tv)

    pre_feat = encode_pre(b, tv, PLAYER_1, rules2)
    scores = []
    for assign in list(iprod(range(len(mov)), repeat=len(tv)))[:4]:
        alloc = [[] for _ in range(len(mov))]
        for vi,si in enumerate(assign): alloc[si].append(tv[vi])
        sim = rules2.simulate_allocation(b, alloc, mov, PLAYER_1)
        state = encode_state(b, tv, sim, 0, PLAYER_1, rules2)
        x = torch.tensor(state[None], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = model(x)
            p = F.softmax(logits, dim=-1).numpy()[0]
        score = p[2]*0.1+p[3]-p[0]*2-p[1]+p[4]*2
        scores.append(round(score, 5))

    unique = len(set(scores))
    print(f'Candidate scores: {scores}')
    print(f'Unique scores: {unique}  (expect >1 — candidates distinguishable)')
    print('All checks passed.' if unique > 1 else 'WARNING: candidates indistinguishable')
