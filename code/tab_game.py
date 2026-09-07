"""
tab Game Engine - FINAL version matching verified HTML.

Board: Row A(0-7) P1 home, B(8-15) battlefield, C(16-23) battlefield, D(24-31) P2 home.

TRACKS:
  P1: B1->B2->...->B8->C8->C7->...->C1  (enters at B1, fork after C1)
  P2: C8->C7->...->C1->B1->B2->...->B8  (enters at C8, fork after B8)

HOME ROW MOVEMENT (freed soldiers):
  P1: moves LEFT within A row. A5+3 -> A2. Past A1 -> exits to B1 then track.
  P2: moves RIGHT within D row. D3+3 -> D6. Past D8 -> exits to C8 then track.

FREEING: each tab moves one queued soldier one step toward exit.
  P1: A8->A7->...->A1->B1. P2: D1->D2->...->D8->C8.

AHMAD'S RULE 1: Can only enter enemy home if ALL soldiers freed (queue empty).
AHMAD'S RULE 2: Enemy home soldiers frozen while owner has active elsewhere
  (unfrozen soldiers in home row OR any soldiers on battlefield).
QUEUE CLEANUP: When a soldier is captured in its home row, remove from queue.

Authors: Ahmad B. Hassanat, Ghada A. Altarawneh and Ahmad S. Tarawneh - Mutah University, Jordan
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# =============================================================================
# Constants
# =============================================================================

DEFAULT_COLUMNS = 8
DEFAULT_ROWS = 4
PLAYER_1 = 1
PLAYER_2 = -1
STICK_VALUES = {0: 6, 1: 1, 2: 2, 3: 3, 4: 4}
THROW_AGAIN = {1, 4, 6}

# =============================================================================
# Tracks
# =============================================================================

def build_tracks(columns=DEFAULT_COLUMNS):
    C = columns
    row_b = list(range(C, 2 * C))
    row_c_rev = list(range(3 * C - 1, 2 * C - 1, -1))
    p1_track = row_b + row_c_rev
    p2_track = row_c_rev + row_b
    p1_enemy = list(range(3 * C, 4 * C))
    p2_enemy = list(range(C - 1, -1, -1))
    return p1_track, p2_track, p1_enemy, p2_enemy

P1_TRACK, P2_TRACK, P1_ENEMY, P2_ENEMY = build_tracks()
TRACK_LEN = len(P1_TRACK)

# =============================================================================
# Dice
# =============================================================================

class StickDice:
    @staticmethod
    def throw_once():
        flat = sum(1 for _ in range(4) if random.random() < 0.5)
        return STICK_VALUES[flat]

    @staticmethod
    def throw_turn():
        seq = []
        while True:
            v = StickDice.throw_once()
            seq.append(v)
            if v not in THROW_AGAIN:
                break
        return seq

# =============================================================================
# Board
# =============================================================================

class Board:
    def __init__(self, columns=DEFAULT_COLUMNS):
        self.columns = columns
        self.rows = DEFAULT_ROWS
        self.board_size = self.rows * columns
        self.cells = [0] * self.board_size
        for i in range(columns):
            self.cells[i] = 1
            self.cells[3 * columns + i] = -1
        self.frozen_at = [
            {i: 1 for i in range(columns)},
            {3 * columns + i: 1 for i in range(columns)},
        ]
        self.frozen_queue = [
            list(range(columns)),
            list(range(4 * columns - 1, 3 * columns - 1, -1)),
        ]
        self.player_started = [False, False]
        # NO RE-ENTRY RULE (LIFO): soldiers that exited enemy home cannot re-enter.
        # Dict per player: key=position, value=list of booleans [bottom...top].
        # True=tainted (was in enemy home), False=clean.
        # tab split: pop from top (LIFO - last arrived leaves first).
        # Group move: entire stack transfers to destination.
        self.no_reentry = [{}, {}]  # [P1_dict, P2_dict]

    def player_index(self, player):
        return 0 if player == PLAYER_1 else 1

    def clone(self):
        b = Board.__new__(Board)
        b.columns = self.columns
        b.rows = self.rows
        b.board_size = self.board_size
        b.cells = self.cells[:]
        b.frozen_at = [dict(d) for d in self.frozen_at]
        b.frozen_queue = [list(q) for q in self.frozen_queue]
        b.player_started = list(self.player_started)
        b.no_reentry = [{k: list(v) for k, v in d.items()} for d in self.no_reentry]
        return b

    def count_alive(self, player):
        if player == PLAYER_1:
            return sum(c for c in self.cells if c > 0)
        return sum(-c for c in self.cells if c < 0)

    def count_on_battlefield(self, player):
        count = 0
        for i in range(self.columns, 3 * self.columns):
            if self.cells[i] * player > 0:
                count += abs(self.cells[i])
        return count

    def is_game_over(self):
        p1 = self.count_alive(PLAYER_1)
        p2 = self.count_alive(PLAYER_2)
        if p2 == 0: return PLAYER_1
        if p1 == 0: return PLAYER_2
        return None

    def get_row(self, pos): return pos // self.columns
    def get_col(self, pos): return pos % self.columns
    def position_label(self, pos):
        return f"{chr(65 + self.get_row(pos))}{self.get_col(pos) + 1}"

    def display(self):
        cols = self.columns
        lines = ["      " + "  ".join(f"{i+1:3d}" for i in range(cols)),
                  "     " + "-" * (cols * 4 + 1)]
        for row in range(4):
            parts = []
            for col in range(cols):
                v = self.cells[row * cols + col]
                if v > 0: parts.append(f"+{abs(v)}" if abs(v) > 1 else " +")
                elif v < 0: parts.append(f"-{abs(v)}" if abs(v) > 1 else " -")
                else: parts.append(" .")
            lines.append(f"  {chr(65+row)} | {'  '.join(parts)} ")
        lines.append("     " + "-" * (cols * 4 + 1))
        lines.append(f"  P1: {self.count_alive(PLAYER_1)} | P2: {self.count_alive(PLAYER_2)}")
        return "\n".join(lines)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SoldierMove:
    from_pos: int
    to_pos: int = -1
    values_used: List[int] = field(default_factory=list)

@dataclass
class TurnAllocation:
    soldier_moves: List[SoldierMove] = field(default_factory=list)
    fitness: float = 0.0

# =============================================================================
# Rules Engine - matches HTML exactly
# =============================================================================

class TabRules:
    def __init__(self, columns=DEFAULT_COLUMNS):
        self.columns = columns
        self.rows = DEFAULT_ROWS
        self.board_size = self.rows * columns
        self.p1_track, self.p2_track, self.p1_enemy, self.p2_enemy = \
            build_tracks(columns)
        self.track_len = len(self.p1_track)

    def get_track(self, player):
        return self.p1_track if player == PLAYER_1 else self.p2_track

    def get_enemy_home(self, player):
        return self.p1_enemy if player == PLAYER_1 else self.p2_enemy

    def is_own_home(self, pos, player):
        if player == PLAYER_1: return 0 <= pos < self.columns
        return 3 * self.columns <= pos < self.board_size

    def is_enemy_home(self, pos, player):
        if player == PLAYER_1: return 3 * self.columns <= pos < self.board_size
        return 0 <= pos < self.columns

    def find_on_track(self, pos, player):
        track = self.get_track(player)
        try: return track.index(pos)
        except ValueError: return -1

    def find_on_enemy(self, pos, player):
        enemy = self.get_enemy_home(player)
        try: return enemy.index(pos)
        except ValueError: return -1

    def all_freed(self, player, board):
        """RULE 1: All soldiers must be freed before entering enemy home."""
        pi = board.player_index(player)
        return len(board.frozen_queue[pi]) == 0

    def has_active_elsewhere(self, player, board):
        """RULE 2: Check if player has unfrozen soldiers in home row or battlefield."""
        pi = board.player_index(player)
        C = self.columns
        hs = 0 if player == PLAYER_1 else 3 * C
        he = C if player == PLAYER_1 else 4 * C
        fz = board.frozen_at[pi]
        # Check home row for freed (unfrozen) soldiers
        for i in range(hs, he):
            if board.cells[i] * player > 0:
                total = abs(board.cells[i])
                frozen = fz.get(i, 0)
                if total > frozen:
                    return True
        # Check battlefield
        for i in range(C, 3 * C):
            if board.cells[i] * player > 0:
                return True
        return False

    def cleanup_captured(self, board, pos):
        """Remove captured soldier from frozen queue if it was in home row."""
        C = self.columns
        if pos < C:  # P1 home
            pi = 0
            qi = board.frozen_queue[pi]
            if pos in qi:
                qi.remove(pos)
            if pos in board.frozen_at[pi]:
                del board.frozen_at[pi][pos]
        elif pos >= 3 * C:  # P2 home
            pi = 1
            qi = board.frozen_queue[pi]
            if pos in qi:
                qi.remove(pos)
            if pos in board.frozen_at[pi]:
                del board.frozen_at[pi][pos]

    # -----------------------------------------------------------------
    # Movement - matches HTML moveSteps() exactly
    # -----------------------------------------------------------------

    def compute_destinations(self, from_pos, steps, player, board):
        """Returns [] (can't move), [pos] (one dest), or [pos1,pos2] (fork choice)."""
        C = self.columns

        # 1. OWN HOME ROW: freed soldier deploying
        # P1: moves LEFT (pos - steps), exits past A1 to battlefield
        # P2: moves RIGHT (pos + steps), exits past D8 to battlefield
        if self.is_own_home(from_pos, player):
            if steps <= 0:
                return []
            if player == PLAYER_1:
                target = from_pos - steps
                if target >= 0:
                    return [target]  # still in row A
                # Exited past A1 -> enters track at B1 (index 0)
                extra = steps - from_pos - 1  # steps after entering B1
                track = self.get_track(player)
                return [track[extra % self.track_len]]
            else:
                target = from_pos + steps
                if target <= 4 * C - 1:
                    return [target]  # still in row D
                # Exited past D8 -> enters track at C8 (index 0)
                extra = steps - (4 * C - 1 - from_pos) - 1
                track = self.get_track(player)
                return [track[extra % self.track_len]]

        # 2. ENEMY HOME: frozen if has active soldiers elsewhere
        if self.is_enemy_home(from_pos, player):
            if self.has_active_elsewhere(player, board):
                return []  # FROZEN
            enemy = self.get_enemy_home(player)
            e_idx = self.find_on_enemy(from_pos, player)
            if e_idx < 0:
                return []
            new_idx = e_idx + steps
            if new_idx < len(enemy):
                return [enemy[new_idx]]  # still in enemy home
            # Re-enter battlefield: D8->C8 or A1->B1 = track index 8
            extra = new_idx - len(enemy)
            track = self.get_track(player)
            return [track[(8 + extra) % self.track_len]]  # FIX: +8 offset

        # 3. BATTLE TRACK
        track = self.get_track(player)
        idx = self.find_on_track(from_pos, player)
        if idx < 0:
            return []
        new_idx = idx + steps
        if new_idx < self.track_len:
            return [track[new_idx]]

        # Fork! Passed end of track
        extra = new_idx - self.track_len
        loop_pos = track[extra % self.track_len]

        # RULE 1: Can only enter enemy home if all freed
        # NO RE-ENTRY RULE: if ANY soldier in this stack has been in enemy
        # home before, the ENTIRE group is blocked from entering.
        pi = board.player_index(player)
        stack = board.no_reentry[pi].get(from_pos, [])
        tainted = any(t for t in stack)  # any True = has tainted soldier
        if self.all_freed(player, board) and not tainted:
            enemy = self.get_enemy_home(player)
            enemy_pos = enemy[min(extra, len(enemy) - 1)]
            if loop_pos == enemy_pos:
                return [loop_pos]
            return [loop_pos, enemy_pos]
        else:
            return [loop_pos]  # must loop (not freed, or tainted)

    # -----------------------------------------------------------------
    # Apply move
    # -----------------------------------------------------------------

    def apply_single_move(self, board, from_pos, steps, player, choice=0):
        """Returns (new_pos, captured, entered_enemy)."""
        sign = 1 if player == PLAYER_1 else -1
        pi = board.player_index(player)
        total = abs(board.cells[from_pos])
        frozen = board.frozen_at[pi].get(from_pos, 0)
        movable = total - frozen
        if movable <= 0:
            return (from_pos, 0, False)
        move_count = 1 if (steps == 1 and movable > 1) else movable
        dests = self.compute_destinations(from_pos, steps, player, board)
        if not dests:
            return (from_pos, 0, False)
        new_pos = dests[min(choice, len(dests) - 1)]

        # NO RE-ENTRY: LIFO taint tracking
        nr = board.no_reentry[pi]
        from_stack = nr.get(from_pos, [])
        was_in_enemy = self.is_enemy_home(from_pos, player)

        # Determine which soldiers move (LIFO: pop from top of stack)
        # Stack top = last element = splits off first
        moving_taints = []
        if from_stack and from_pos != new_pos:
            if move_count >= len(from_stack):
                # Entire stack moves (or more soldiers than tracked)
                moving_taints = list(from_stack)
                nr.pop(from_pos, None)
            else:
                # LIFO: pop move_count soldiers from top
                moving_taints = from_stack[-move_count:]
                nr[from_pos] = from_stack[:-move_count]
                if not nr[from_pos]:
                    del nr[from_pos]

        board.cells[from_pos] -= move_count * sign
        captured = 0
        if board.cells[new_pos] * player < 0:
            captured = abs(board.cells[new_pos])
            self.cleanup_captured(board, new_pos)
            # Also clear opponent's taint stack at this position
            opp_pi = 1 - pi
            if new_pos in board.no_reentry[opp_pi]:
                del board.no_reentry[opp_pi][new_pos]
            board.cells[new_pos] = 0
        board.cells[new_pos] += move_count * sign

        # Update taint at destination
        if was_in_enemy and not self.is_enemy_home(new_pos, player):
            # Exiting enemy home -> ALL moving soldiers become tainted
            dest_stack = nr.get(new_pos, [])
            dest_stack.extend([True] * move_count)
            nr[new_pos] = dest_stack
        elif moving_taints and not self.is_enemy_home(new_pos, player):
            # Transfer taint stack to destination (append on top)
            dest_stack = nr.get(new_pos, [])
            dest_stack.extend(moving_taints)
            nr[new_pos] = dest_stack
        elif not from_stack and not was_in_enemy:
            # Clean soldiers moving - add False entries if destination has a stack
            # (they stack on top as clean)
            if new_pos in nr:
                nr[new_pos].extend([False] * move_count)

        # Clean up if all soldiers left from_pos
        if board.cells[from_pos] == 0 and from_pos in nr:
            del nr[from_pos]

        return (new_pos, captured, self.is_enemy_home(new_pos, player))

    # -----------------------------------------------------------------
    # Freeing
    # -----------------------------------------------------------------

    def free_destination(self, pos, player):
        """Where does a frozen soldier at pos go when freed one step?"""
        C = self.columns
        if player == PLAYER_1:
            return C if pos == 0 else pos - 1   # A1->B1, else left
        else:
            return 3 * C - 1 if pos == 4 * C - 1 else pos + 1  # D8->C8, else right

    def can_free(self, pos, player, board):
        """
        Can this frozen soldier be freed?
        YES if: (a) it's next in queue, OR
                (b) its free destination has an enemy (capture bypass)
        """
        pi = board.player_index(player)
        fa = board.frozen_at[pi]
        queue = board.frozen_queue[pi]
        if not queue:
            return False
        if board.cells[pos] * player <= 0:
            return False
        if fa.get(pos, 0) <= 0:
            return False
        # Next in queue - always allowed
        if queue[0] == pos:
            return True
        # Not in queue at all
        if pos not in queue:
            return False
        # Capture bypass: destination has enemy
        dest = self.free_destination(pos, player)
        if board.cells[dest] * player < 0:
            return True
        return False

    def free_frozen_soldier(self, board, player):
        """Move next queued soldier one step toward exit. Returns (old,new,cap) or None."""
        pi = board.player_index(player)
        queue = board.frozen_queue[pi]
        if not queue:
            return None
        old_pos = queue.pop(0)
        sign = 1 if player == PLAYER_1 else -1
        new_pos = self.free_destination(old_pos, player)
        fa = board.frozen_at[pi]
        fa[old_pos] = fa.get(old_pos, 1) - 1
        if fa[old_pos] <= 0:
            del fa[old_pos]
        board.cells[old_pos] -= sign
        captured = 0
        if board.cells[new_pos] * player < 0:
            captured = abs(board.cells[new_pos])
            self.cleanup_captured(board, new_pos)
            board.cells[new_pos] = 0
        board.cells[new_pos] += sign
        return (old_pos, new_pos, captured)

    def free_specific_soldier(self, board, player, pos):
        """Free a SPECIFIC frozen soldier (for capture bypass). Returns (old,new,cap)."""
        pi = board.player_index(player)
        sign = 1 if player == PLAYER_1 else -1
        queue = board.frozen_queue[pi]
        if pos in queue:
            queue.remove(pos)
        fa = board.frozen_at[pi]
        fa[pos] = fa.get(pos, 1) - 1
        if fa[pos] <= 0:
            del fa[pos]
        new_pos = self.free_destination(pos, player)
        board.cells[pos] -= sign
        captured = 0
        if board.cells[new_pos] * player < 0:
            captured = abs(board.cells[new_pos])
            self.cleanup_captured(board, new_pos)
            board.cells[new_pos] = 0
        board.cells[new_pos] += sign
        return (pos, new_pos, captured)

    def get_throw_sequence_with_freeing(self, board, player, throw_sequence):
        """
        Process throw sequence. tab (value 1) is treated as a CHOICE:
        - Auto-free ONLY the first tab if player hasn't started yet
        - After that, return ALL values (including 1s) for allocation
        - The AI/ExhaustiveAgent decides whether to free or move with each 1
        Returns (remaining_values, num_freed).
        """
        pi = board.player_index(player)
        remaining = list(throw_sequence)
        freed = 0

        # Must have at least one tab to start the game
        if not board.player_started[pi]:
            if 1 in remaining:
                board.player_started[pi] = True
                # Free one to get started, keep rest as choices
                remaining.remove(1)
                result = self.free_frozen_soldier(board, player)
                if result:
                    freed += 1
            else:
                # No tab -> can't play
                return ([], 0)

        # Return ALL remaining values (including 1s) for allocation
        # The agent decides whether to use 1s for freeing or moving
        return (remaining, freed)

    # -----------------------------------------------------------------
    # Movable soldiers
    # -----------------------------------------------------------------

    def get_movable_soldiers(self, board, player, throw_values):
        """Get positions of movable soldiers. Includes frozen with capture bypass."""
        pi = board.player_index(player)
        C = self.columns
        hs = 0 if player == PLAYER_1 else 3 * C
        he = C if player == PLAYER_1 else 4 * C
        os = 3 * C if player == PLAYER_1 else 0
        oe = 4 * C if player == PLAYER_1 else C
        ef = self.has_active_elsewhere(player, board)

        movable = []
        # Home row: freed (unfrozen) soldiers
        for i in range(hs, he):
            if board.cells[i] * player > 0:
                total = abs(board.cells[i])
                frozen = board.frozen_at[pi].get(i, 0)
                if total > frozen:
                    movable.append(i)

        # Home row: frozen soldiers that CAN be freed (queue order OR capture bypass)
        if throw_values and 1 in throw_values and board.frozen_queue[pi]:
            for i in range(hs, he):
                if i not in movable and self.can_free(i, player, board):
                    movable.append(i)

        # Battlefield
        for i in range(C, 3 * C):
            if board.cells[i] * player > 0:
                movable.append(i)
        # Enemy home: split with tab only if NOT frozen
        if not ef and throw_values and 1 in throw_values:
            for i in range(os, oe):
                if board.cells[i] * player > 0 and abs(board.cells[i]) > 1:
                    if i not in movable:
                        movable.append(i)
        if movable:
            return movable
        # Last resort: enemy home (unfrozen)
        if not ef:
            for i in range(os, oe):
                if board.cells[i] * player > 0:
                    movable.append(i)
        return movable

    # -----------------------------------------------------------------
    # Simulation helper for AI
    # -----------------------------------------------------------------

    def simulate_allocation(self, board, allocation, movable_soldiers, player):
        """Simulate allocation, return resulting board clone (for fitness eval)."""
        sim = board.clone()
        sign = 1 if player == PLAYER_1 else -1
        pi = sim.player_index(player)
        for i, values in enumerate(allocation):
            if i >= len(movable_soldiers) or not values:
                continue
            pos = movable_soldiers[i]
            for val in values:
                # Check if this is a FROZEN soldier being freed with tab
                if val == 1 and self.is_own_home(pos, player) and \
                   sim.frozen_at[pi].get(pos, 0) > 0 and \
                   self.can_free(pos, player, sim):
                    # Free this soldier
                    self.free_specific_soldier(sim, player, pos)
                    continue

                total = abs(sim.cells[pos])
                frozen = sim.frozen_at[pi].get(pos, 0)
                m = total - frozen
                if m <= 0:
                    continue
                dests = self.compute_destinations(pos, val, player, sim)
                if not dests:
                    continue
                if len(dests) > 1:
                    best_choice = 0
                    best_score = float('-inf')
                    for ci, d in enumerate(dests):
                        score = 0
                        if sim.cells[d] * player < 0:
                            score += abs(sim.cells[d]) * 10
                        if self.is_enemy_home(d, player):
                            score += 5
                        if score > best_score:
                            best_score = score
                            best_choice = ci
                    new_pos, cap, _ = self.apply_single_move(
                        sim, pos, val, player, choice=best_choice)
                else:
                    new_pos, cap, _ = self.apply_single_move(
                        sim, pos, val, player, choice=0)
                pos = new_pos
        return sim
