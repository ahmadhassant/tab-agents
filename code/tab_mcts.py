"""
tab_mcts.py - Monte Carlo Tree Search for tab.

PIECES 1-2 OF 4: rollout engine + flat Monte Carlo search (UCB1).

This file is built incrementally.

  Piece 1  rollout engine - plays a tab game forward from an arbitrary
           state under a pluggable policy, using EXACTLY the rules and the
           fork-picking heuristic that tab_runner.play_one_game uses.
  Piece 2  MCTSAgent - flat Monte Carlo search: UCB1 over the candidate
           allocations of the current turn, evaluated by rollout. This is
           the standard "flat MC bandit" formulation of MCTS.
  Piece 3  chance-node tree (planned) - expands opponent stick-throws as
           chance nodes so the search goes deeper than one turn.
  Piece 4  budget instrumentation + tournament registration (planned).

Why this is a separate, tested piece: an MCTS agent is only as valid as the
game its rollouts play. The earlier MCTS attempt shipped alongside an engine
that had no no-reentry rule, no LIFO taint and no fork handling, so none of
its results transferred. test_mcts_parity.py proves this rollout engine and
the tournament engine are the same game, seed for seed, before any search
code is written against it.

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
(AI assistance disclosed in the manuscript acknowledgments)
"""

import random
from typing import Callable, List, Optional

from tab_game import (Board, TabRules, StickDice, SoldierMove, TurnAllocation,
                        PLAYER_1, PLAYER_2)

# An allocation in "candidate" form: one list of throw-values per movable
# soldier, parallel to the movable_soldiers list. This is the same
# representation tab_look.get_candidate_allocations produces, so MCTS
# searches the identical action space as the TLA family.
Allocation = List[List[int]]

# A policy maps a decision point to one allocation.
Policy = Callable[[Board, int, List[int], List[int], TabRules], Allocation]


# ============================================================================
# Canonical fork picker - must stay identical to tab_runner.play_one_game
# ============================================================================

def pick_fork(board: Board, dests: List[int], player: int,
                rules: TabRules) -> int:
    """
    Uniform fork-picking heuristic: cap x 10 + enemy_home x 5.

    Fixed at engine level for every agent (Section III of the paper), so the
    tournament measures allocation choice, not fork choice.
    """
    if len(dests) <= 1:
        return 0
    best_score = float('-inf')
    choice = 0
    for ci, d in enumerate(dests):
        s = 0.0
        if board.cells[d] * player < 0:
            s += abs(board.cells[d]) * 10.0
        if rules.is_enemy_home(d, player):
            s += 5.0
        if s > best_score:
            best_score = s
            choice = ci
    return choice


# ============================================================================
# Turn mechanics
# ============================================================================

def resolve_freeing(board: Board, player: int, seq: List[int],
                      rules: TabRules) -> List[int]:
    """
    Consume 1s from the throw sequence to free frozen soldiers in queue order.
    Returns the remaining throw values. Mirrors tab_runner exactly.
    """
    remaining = list(seq)
    pi = board.player_index(player)
    ones_used = 0
    while 1 in remaining and ones_used < seq.count(1):
        if not board.frozen_queue[pi]:
            break
        next_pos = board.frozen_queue[pi][0]
        if rules.can_free(next_pos, player, board):
            rules.free_specific_soldier(board, player, next_pos)
            remaining.remove(1)
            ones_used += 1
        else:
            break
    return remaining


def apply_allocation(board: Board, allocation: Allocation,
                       soldiers: List[int], player: int,
                       rules: TabRules) -> None:
    """
    Apply an allocation to the board IN PLACE, using the canonical fork
    picker. Mirrors the apply loop in tab_runner.play_one_game.
    """
    pi = board.player_index(player)
    for i, values in enumerate(allocation):
        if not values:
            continue
        pos = soldiers[i]
        for val in values:
            # Freeing check: a 1 spent on a frozen soldier in own home frees it
            if val == 1 and rules.is_own_home(pos, player) and \
                    board.frozen_at[pi].get(pos, 0) > 0 and \
                    rules.can_free(pos, player, board):
                rules.free_specific_soldier(board, player, pos)
                continue
            total = abs(board.cells[pos])
            frozen = board.frozen_at[pi].get(pos, 0)
            if total - frozen <= 0:
                continue
            dests = rules.compute_destinations(pos, val, player, board)
            if not dests:
                continue
            choice = pick_fork(board, dests, player, rules)
            new_pos, _cap, _ = rules.apply_single_move(
                board, pos, val, player, choice=choice)
            pos = new_pos


def step_turn(board: Board, player: int, rules: TabRules, policy: Policy,
                seq: Optional[List[int]] = None) -> bool:
    """
    Play one full turn for `player` on `board`, in place.

    Args:
        seq: throw sequence to use. If None, one is thrown. Passing an
             explicit sequence is what lets the chance-node search in
             piece 3 expand a specific dice outcome.

    Returns True if an allocation was applied, False if the player had no
    legal move (a passed turn).
    """
    if seq is None:
        seq = StickDice.throw_turn()
    remaining = resolve_freeing(board, player, seq, rules)
    movable = rules.get_movable_soldiers(board, player, remaining)
    if not movable or not remaining:
        return False
    allocation = policy(board, player, remaining, movable, rules)
    apply_allocation(board, allocation, movable, player, rules)
    return True


# ============================================================================
# Policies
# ============================================================================

def random_policy(board: Board, player: int, values: List[int],
                    soldiers: List[int], rules: TabRules) -> Allocation:
    """
    Uniform random allocation: each throw value goes to a random movable
    soldier. Consumes randomness in the same order as tab_ai.RandomAgent,
    so seeded games are directly comparable.
    """
    n = len(soldiers)
    assignments: Allocation = [[] for _ in range(n)]
    for val in values:
        assignments[random.randint(0, n - 1)].append(val)
    return assignments


# ============================================================================
# Rollout
# ============================================================================

def rollout(board: Board, player_to_move: int, rules: TabRules,
              policy: Policy = random_policy,
              max_turns: int = 300) -> int:
    """
    Play `board` to completion from `player_to_move` and return the winner.

    Returns PLAYER_1 (1), PLAYER_2 (-1), or 0 for a draw. The board is
    cloned, so the caller's state is untouched.

    Terminal handling mirrors tab_runner: if max_turns is reached, the
    winner is decided on remaining soldier count.
    """
    sim = board.clone()
    player = player_to_move
    turn = 0
    winner = sim.is_game_over()
    while winner is None and turn < max_turns:
        step_turn(sim, player, rules, policy)
        winner = sim.is_game_over()
        if winner is not None:
            break
        player = -player
        turn += 1
    if winner is None:
        p1, p2 = sim.count_alive(PLAYER_1), sim.count_alive(PLAYER_2)
        winner = PLAYER_1 if p1 > p2 else (PLAYER_2 if p2 > p1 else 0)
    return winner


def play_game(policy1: Policy, policy2: Policy, max_turns: int = 300,
                columns: int = 8, seed: Optional[int] = None) -> dict:
    """
    Play one complete game between two policies.

    Deliberately mirrors tab_runner.play_one_game turn for turn. Its only
    purpose is the parity harness: with the same seed and equivalent
    policies, this and play_one_game must produce identical games.
    """
    if seed is not None:
        random.seed(seed)
    board = Board(columns)
    rules = TabRules(columns)
    turn = 0
    while turn < max_turns:
        player = PLAYER_1 if turn % 2 == 0 else PLAYER_2
        policy = policy1 if player == PLAYER_1 else policy2
        step_turn(board, player, rules, policy)
        winner = board.is_game_over()
        if winner is not None:
            break
        turn += 1
    final_winner = board.is_game_over()
    if final_winner is None:
        p1, p2 = board.count_alive(PLAYER_1), board.count_alive(PLAYER_2)
        final_winner = PLAYER_1 if p1 > p2 else (PLAYER_2 if p2 > p1 else 0)
    p1_alive = board.count_alive(PLAYER_1)
    p2_alive = board.count_alive(PLAYER_2)
    return {
        'winner': final_winner,
        'turns': turn + 1,
        'p1_alive': p1_alive,
        'p2_alive': p2_alive,
        'p1_captures': 8 - p2_alive,
        'p2_captures': 8 - p1_alive,
    }


if __name__ == '__main__':
    import time
    rules = TabRules(8)
    t0 = time.time()
    n = 200
    wins = {1: 0, -1: 0, 0: 0}
    for s in range(n):
        random.seed(10_000 + s)
        wins[rollout(Board(8), PLAYER_1, rules)] += 1
    dt = time.time() - t0
    print(f"{n} random rollouts in {dt:.2f}s "
            f"({n/dt:.0f} rollouts/s, {dt/n*1000:.1f} ms each)")
    print(f"  P1 {wins[1]}  P2 {wins[-1]}  draw {wins[0]}")


# ============================================================================
# PIECE 2: flat Monte Carlo search (UCB1 over candidate allocations)
# ============================================================================

import math

from tab_game import SoldierMove, TurnAllocation
from tab_look import get_candidate_allocations


def to_turn_allocation(allocation: Allocation,
                         soldiers: List[int]) -> TurnAllocation:
    """Convert candidate form to the TurnAllocation the engine consumes."""
    ta = TurnAllocation()
    for i, values in enumerate(allocation):
        if values:
            ta.soldier_moves.append(SoldierMove(from_pos=soldiers[i],
                                                  to_pos=soldiers[i],
                                                  values_used=list(values)))
    return ta


def heavy_policy(board: Board, player: int, values: List[int],
                   soldiers: List[int], rules: TabRules) -> Allocation:
    """
    Capture-greedy rollout policy: epsilon-greedy toward moves that capture.

    Deliberately uses NO tab-specific evaluation terms - only "does this move
    land on an enemy soldier", which is generic MCTS heavy-playout practice.
    This keeps MCTS an independent baseline: it borrows nothing from
    LookEvaluator or FitnessEvaluator, so a comparison against the TLA family
    remains a cross-family comparison.
    """
    n = len(soldiers)
    assignments: Allocation = [[] for _ in range(n)]
    for val in values:
        capturing = []
        for i, pos in enumerate(soldiers):
            dests = rules.compute_destinations(pos, val, player, board)
            if not dests:
                continue
            d = dests[pick_fork(board, dests, player, rules)]
            if board.cells[d] * player < 0:
                capturing.append(i)
        if capturing and random.random() < 0.8:
            assignments[random.choice(capturing)].append(val)
        else:
            assignments[random.randint(0, n - 1)].append(val)
    return assignments


ROLLOUT_POLICIES = {'pure': random_policy, 'heavy': heavy_policy}


class MCTSAgent:
    """
    Monte Carlo Tree Search agent (flat formulation).

    Each candidate allocation for the current turn is an arm of a bandit.
    UCB1 allocates the simulation budget across arms; each simulation applies
    the arm and plays the game out with the rollout policy. The arm with the
    most visits is played (robust child).

    The candidate set comes from tab_look.get_candidate_allocations, the same
    generator the TLA family uses, so MCTS searches an identical action space
    and win-rate differences are attributable to the search, not the moves
    available to it.

    Args:
        n_simulations: rollout budget per decision. This is the compute knob
            for the budget-matched comparison.
        c_uct: UCB1 exploration constant. sqrt(2) is the standard choice for
            rewards in [0, 1].
        rollout: 'pure' (uniform random) or 'heavy' (capture-greedy).
        max_candidates: cap on arms, so a wide turn does not spread the
            budget so thin that every arm has one visit.
    """

    PRESETS = {'mcts-200': 200, 'mcts-500': 500,
                 'mcts-1000': 1000, 'mcts-2000': 2000}

    def __init__(self, n_simulations: int = 1000, c_uct: float = math.sqrt(2),
                   rollout: str = 'pure', enum_threshold: int = 2000,
                   sample_n: int = 80, max_candidates: int = 64,
                   max_rollout_turns: int = 200, columns: int = 8):
        if rollout not in ROLLOUT_POLICIES:
            raise ValueError(f"rollout must be one of {list(ROLLOUT_POLICIES)}")
        self.n_simulations = n_simulations
        self.c_uct = c_uct
        self.rollout_name = rollout
        self.rollout_policy = ROLLOUT_POLICIES[rollout]
        self.enum_threshold = enum_threshold
        self.sample_n = sample_n
        self.max_candidates = max_candidates
        self.max_rollout_turns = max_rollout_turns
        self.columns = columns
        # Instrumentation for the compute-matched comparison (E1/E2)
        self.decisions = 0
        self.total_rollouts = 0
        self.total_candidates = 0

    def choose_allocation(self, board: Board, player: int,
                            throw_values: List[int],
                            movable_soldiers: List[int],
                            rules: TabRules) -> TurnAllocation:
        if not throw_values or not movable_soldiers:
            return TurnAllocation()

        candidates = get_candidate_allocations(
            throw_values, movable_soldiers, player, board, rules,
            enum_threshold=self.enum_threshold, sample_n=self.sample_n)
        if not candidates:
            return TurnAllocation()
        if len(candidates) > self.max_candidates:
            candidates = random.sample(candidates, self.max_candidates)

        self.decisions += 1
        self.total_candidates += len(candidates)

        if len(candidates) == 1:
            return to_turn_allocation(candidates[0], movable_soldiers)

        # Pre-apply each arm once; rollouts start from the resulting state.
        children = []
        for alloc in candidates:
            child = board.clone()
            apply_allocation(child, alloc, movable_soldiers, player, rules)
            children.append(child)

        k = len(candidates)
        visits = [0] * k
        wins = [0.0] * k
        budget = max(self.n_simulations, k)

        for t in range(budget):
            if t < k:
                arm = t                      # one visit to each arm first
            else:
                logN = math.log(t)
                arm, best = 0, float('-inf')
                for i in range(k):
                    u = wins[i] / visits[i] + \
                        self.c_uct * math.sqrt(logN / visits[i])
                    if u > best:
                        best, arm = u, i

            child = children[arm]
            terminal = child.is_game_over()
            if terminal is not None:
                winner = terminal
            else:
                winner = rollout(child, -player, rules,
                                   policy=self.rollout_policy,
                                   max_turns=self.max_rollout_turns)
            visits[arm] += 1
            wins[arm] += 1.0 if winner == player else (
                0.5 if winner == 0 else 0.0)

        self.total_rollouts += budget
        best_arm = max(range(k), key=lambda i: (visits[i], wins[i]))
        return to_turn_allocation(candidates[best_arm], movable_soldiers)

    def stats(self) -> dict:
        """Per-decision compute, for the budget-matched comparison."""
        d = max(self.decisions, 1)
        return {'decisions': self.decisions,
                  'rollouts_per_decision': self.total_rollouts / d,
                  'candidates_per_decision': self.total_candidates / d,
                  'rollout': self.rollout_name,
                  'n_simulations': self.n_simulations}


# ============================================================================
# PIECE 3: chance-node tree search
# ============================================================================
#
# Piece 2 is a flat bandit: it evaluates each allocation of the CURRENT turn
# by rollout and stops there. Reviewer 3 asked whether *deeper stochastic
# search* helps, which needs a tree that alternates:
#
#     decision node (a player allocates known throw values)
#            |
#     chance node   (the next player's stick throw is random)
#            |
#     decision node ...
#
# Chance nodes are sampled from the real StickDice distribution rather than
# a truncated table, so the tree is unbiased in the limit; children are keyed
# by the throw sequence actually drawn. Beyond `max_depth` plies the search
# hands off to the same rollout policy piece 2 uses, so the two agents differ
# only in how they spend their simulation budget.


class _Decision:
    """A node where `player` allocates `values` among `movable`."""
    __slots__ = ('board', 'player', 'values', 'movable', 'actions',
                   'children', 'N', 'W', 'depth')

    def __init__(self, board, player, values, movable, actions, depth):
        self.board = board
        self.player = player
        self.values = values
        self.movable = movable
        self.actions = actions          # list of allocations
        self.children = {}              # action index -> _Chance
        self.N = [0] * len(actions)
        self.W = [0.0] * len(actions)
        self.depth = depth


class _Chance:
    """A node where the stick throw for `player` has not yet been drawn."""
    __slots__ = ('board', 'player', 'children', 'depth')

    def __init__(self, board, player, depth):
        self.board = board
        self.player = player
        self.children = {}              # throw tuple -> _Decision | None
        self.depth = depth


class MCTSTreeAgent:
    """
    Chance-node MCTS.

    Same action space, rollout policies and budget knob as MCTSAgent, so the
    two are directly comparable: any difference is attributable to search
    depth rather than to moves available or playout quality.

    Args:
        n_simulations: rollouts per decision (the compute knob).
        max_depth: plies of tree above the rollout horizon. Each ply is one
            player's turn, so depth 4 looks two turns ahead for each side.
    """

    def __init__(self, n_simulations: int = 800, c_uct: float = math.sqrt(2),
                   rollout: str = 'pure', enum_threshold: int = 2000,
                   sample_n: int = 80, max_candidates: int = 32,
                   max_depth: int = 4, max_rollout_turns: int = 200,
                   columns: int = 8):
        if rollout not in ROLLOUT_POLICIES:
            raise ValueError(f"rollout must be one of {list(ROLLOUT_POLICIES)}")
        self.n_simulations = n_simulations
        self.c_uct = c_uct
        self.rollout_name = rollout
        self.rollout_policy = ROLLOUT_POLICIES[rollout]
        self.enum_threshold = enum_threshold
        self.sample_n = sample_n
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        self.max_rollout_turns = max_rollout_turns
        self.columns = columns
        self.decisions = 0
        self.total_rollouts = 0
        self.total_nodes = 0

    # -- helpers ---------------------------------------------------------
    def _actions(self, board, player, values, movable, rules):
        acts = get_candidate_allocations(
            values, movable, player, board, rules,
            enum_threshold=self.enum_threshold, sample_n=self.sample_n)
        if len(acts) > self.max_candidates:
            acts = random.sample(acts, self.max_candidates)
        return acts

    def _make_decision(self, board, player, rules, depth, seq=None):
        """Draw a throw for `player` and build a decision node, or None if
        the turn passes / the game is over."""
        if board.is_game_over() is not None:
            return None
        sim = board.clone()
        if seq is None:
            seq = StickDice.throw_turn()
        remaining = resolve_freeing(sim, player, seq, rules)
        movable = rules.get_movable_soldiers(sim, player, remaining)
        if not movable or not remaining:
            return None                              # passed turn
        acts = self._actions(sim, player, remaining, movable, rules)
        if not acts:
            return None
        self.total_nodes += 1
        return _Decision(sim, player, remaining, movable, acts, depth)

    def _reward(self, winner, player):
        return 1.0 if winner == player else (0.5 if winner == 0 else 0.0)

    # -- one simulation --------------------------------------------------
    def _simulate(self, node: '_Decision', rules) -> int:
        """Descend/expand from `node`, roll out, and return the winner."""
        path = []                                    # (decision, action_idx)
        cur = node
        while True:
            k = len(cur.actions)
            # pick an action: unvisited first, then UCB1
            unvisited = [i for i in range(k) if cur.N[i] == 0]
            if unvisited:
                idx = random.choice(unvisited)
            else:
                tot = sum(cur.N)
                logN = math.log(tot) if tot > 0 else 0.0
                idx, best = 0, float('-inf')
                for i in range(k):
                    u = cur.W[i] / cur.N[i] + \
                        self.c_uct * math.sqrt(logN / cur.N[i])
                    if u > best:
                        best, idx = u, i
            path.append((cur, idx))

            # apply the action -> opponent's chance node
            child = cur.children.get(idx)
            if child is None:
                nb = cur.board.clone()
                apply_allocation(nb, cur.actions[idx], cur.movable,
                                   cur.player, rules)
                child = _Chance(nb, -cur.player, cur.depth + 1)
                cur.children[idx] = child

            winner = child.board.is_game_over()
            if winner is not None:
                break
            if child.depth >= self.max_depth:
                winner = rollout(child.board, child.player, rules,
                                   policy=self.rollout_policy,
                                   max_turns=self.max_rollout_turns)
                break

            # chance node: draw a throw
            seq = tuple(StickDice.throw_turn())
            if seq in child.children:
                nxt = child.children[seq]
            else:
                nxt = self._make_decision(child.board, child.player, rules,
                                            child.depth + 1, seq=list(seq))
                child.children[seq] = nxt
            if nxt is None:
                # turn passed (or terminal): roll out from here
                winner = rollout(child.board, -child.player, rules,
                                   policy=self.rollout_policy,
                                   max_turns=self.max_rollout_turns)
                break
            cur = nxt

        for dec, idx in path:
            dec.N[idx] += 1
            dec.W[idx] += self._reward(winner, dec.player)
        return winner

    # -- agent interface -------------------------------------------------
    def choose_allocation(self, board: Board, player: int,
                            throw_values: List[int],
                            movable_soldiers: List[int],
                            rules: TabRules) -> TurnAllocation:
        if not throw_values or not movable_soldiers:
            return TurnAllocation()
        acts = self._actions(board, player, throw_values,
                               movable_soldiers, rules)
        if not acts:
            return TurnAllocation()
        self.decisions += 1
        if len(acts) == 1:
            return to_turn_allocation(acts[0], movable_soldiers)

        root = _Decision(board.clone(), player, list(throw_values),
                           list(movable_soldiers), acts, 0)
        budget = max(self.n_simulations, len(acts))
        for _ in range(budget):
            self._simulate(root, rules)
        self.total_rollouts += budget

        best = max(range(len(acts)), key=lambda i: (root.N[i], root.W[i]))
        return to_turn_allocation(acts[best], movable_soldiers)

    def stats(self) -> dict:
        d = max(self.decisions, 1)
        return {'decisions': self.decisions,
                  'rollouts_per_decision': self.total_rollouts / d,
                  'nodes_per_decision': self.total_nodes / d,
                  'rollout': self.rollout_name,
                  'max_depth': self.max_depth,
                  'n_simulations': self.n_simulations}
