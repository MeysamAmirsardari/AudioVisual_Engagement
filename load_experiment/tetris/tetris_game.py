#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tetris_game.py — a complete, deterministic Tetris engine.

Real, modern Tetris: the seven tetrominoes, the Super Rotation System with wall
kicks, a 7-bag randomiser, gravity, lock delay, soft/hard drop, hold, a next-piece
queue, standard line-clear scoring, levels, and game over. It is renderer-agnostic
(it hands back the board as a grayscale image or as drawable cells), so it drives the
PsychoPy experiment and a stand-alone pygame window from the same code.

Determinism is the point: the simulation is a fixed-rate tick loop and every input is
stamped with the tick it happened on. A played game is therefore a (seed, inputs)
pair that can be replayed frame-for-frame — which is what the visual-embedding
reconstruction depends on. Three modes:

    mode="player"   you drive it: press(action) each frame; update(dt) runs gravity
    mode="replay"   hand it a recorded input list; update(dt) reproduces the game
    mode="ai"       a heuristic plays on its own (the passive-stimulus option)

Call update(dt) once per frame and to_image() (or render_cells()) to draw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Tetromino geometry — the four Super-Rotation-System states of each piece,
# as (row, col) cells inside the piece's bounding box (row increases downward).
# ---------------------------------------------------------------------------
SHAPES: dict[str, list[list[tuple[int, int]]]] = {
    "I": [[(1, 0), (1, 1), (1, 2), (1, 3)], [(0, 2), (1, 2), (2, 2), (3, 2)],
          [(2, 0), (2, 1), (2, 2), (2, 3)], [(0, 1), (1, 1), (2, 1), (3, 1)]],
    "O": [[(0, 0), (0, 1), (1, 0), (1, 1)]] * 4,
    "T": [[(0, 1), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (1, 2), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (2, 1)], [(0, 1), (1, 0), (1, 1), (2, 1)]],
    "S": [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 1), (1, 1), (1, 2), (2, 2)],
          [(1, 1), (1, 2), (2, 0), (2, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],
    "Z": [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 2), (1, 1), (1, 2), (2, 1)],
          [(1, 0), (1, 1), (2, 1), (2, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],
    "J": [[(0, 0), (1, 0), (1, 1), (1, 2)], [(0, 1), (0, 2), (1, 1), (2, 1)],
          [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (1, 1), (2, 0), (2, 1)]],
    "L": [[(0, 2), (1, 0), (1, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (2, 2)],
          [(1, 0), (1, 1), (1, 2), (2, 0)], [(0, 0), (0, 1), (1, 1), (2, 1)]],
}
PIECE_NAMES = list(SHAPES)
PIECE_ID = {name: i + 1 for i, name in enumerate(PIECE_NAMES)}   # 1..7 (0 = empty)

# SRS wall-kick offsets as (d_col, d_row) tried in order when rotating, keyed by
# (from_state, to_state). Standard tables (the reference y-up sign is flipped to our
# row-down board, so a table "+1 up" becomes d_row = -1).
_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (1, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (1, 2): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (2, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (2, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (3, 2): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (3, 0): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (0, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
}
_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
}

# Grayscale shades in PsychoPy image range [-1 (black) .. +1 (white)].
_BG, _FRAME, _SETTLED, _ACTIVE, _FLASH = -1.0, -0.35, 0.35, 0.85, 1.0
# RGB colours for the playable window (classic-ish palette), indexed by piece id.
COLORS = {0: (20, 22, 28), 1: (0, 240, 240), 2: (240, 240, 0), 3: (160, 0, 240),
          4: (0, 240, 0), 5: (240, 0, 0), 6: (0, 0, 240), 7: (240, 160, 0)}

# Gravity: internal ticks per one-cell drop, by level (level 0 slow -> fast). Runs
# at TICK_HZ; soft drop overrides it. Classic-style curve, capped.
TICK_HZ = 60.0
_GRAVITY_TICKS = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2]
_LINE_SCORE = {1: 100, 2: 300, 3: 500, 4: 800}


@dataclass
class _Active:
    name: str
    state: int
    row: int
    col: int


class TetrisGame:
    def __init__(self, cols: int = 10, rows: int = 20, seed: int = 0,
                 mode: str = "player", tick_hz: float = TICK_HZ,
                 lock_delay_ticks: int = 30, clear_flash_ticks: int = 8,
                 max_lock_resets: int = 15, on_top_out: str = "game_over",
                 start_level: int = 0, gravity_scale: float = 1.0,
                 tick_interval_s: float | None = None, flash_s: float | None = None):
        self.cols, self.rows = cols, rows
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick_hz = float(tick_hz)
        self.lock_delay_ticks = int(lock_delay_ticks)
        # difficulty: start_level sets the initial gravity; gravity_scale multiplies the
        # whole speed curve (smaller = faster). Both are recorded so replay matches.
        self.start_level = int(start_level)
        self.gravity_scale = float(gravity_scale)
        # legacy kwargs (old callers passed seconds); honour them if given
        self.clear_flash_ticks = (int(round(flash_s * self.tick_hz))
                                  if flash_s is not None else int(clear_flash_ticks))
        self.max_lock_resets = int(max_lock_resets)
        self.on_top_out = on_top_out

        self.board = np.zeros((rows, cols), dtype=np.uint8)   # 0 empty, else piece id
        self._bag: list[str] = []
        self.queue: list[str] = [self._draw() for _ in range(5)]
        self.hold: str | None = None
        self._hold_used = False

        # scoring / stats
        self.level = int(start_level)
        self.score = 0
        self.lines = 0
        self.pieces_placed = 0
        self.line_clear_events = 0
        self.rows_cleared = 0
        self.tetrises = 0
        self.combo = -1
        self.max_stack_height = 0
        self.resets = 0
        self.game_over = False

        # timing / tick state
        self.tick = 0
        self._accum = 0.0
        self._gravity_counter = 0
        self._lock_counter = 0
        self._lock_resets = 0
        self._resting = False
        self._clearing: list[int] = []
        self._clear_timer = 0

        # recording / replay. Inputs are buffered and applied at the next tick
        # boundary inside step(), so live play and replay run identical code.
        self._pending: list[str] = []
        self._inputs: list[list] = []                # [[tick, action], ...]
        self._events = {"clears": [], "game_over_tick": None}
        self._replay: dict[int, list[str]] = {}

        self._ai_plan: list[str] = []                # queued actions for mode="ai"
        self.active: _Active | None = None
        self._spawn()

    # ---- randomiser --------------------------------------------------------
    def _draw(self) -> str:
        if not self._bag:
            self._bag = PIECE_NAMES[:]
            self.rng.shuffle(self._bag)
        return self._bag.pop()

    # ---- piece geometry ----------------------------------------------------
    def _cells(self, a: _Active):
        return [(a.row + r, a.col + c) for r, c in SHAPES[a.name][a.state]]

    def _fits(self, a: _Active) -> bool:
        for r, c in self._cells(a):
            if c < 0 or c >= self.cols or r < 0 or r >= self.rows or self.board[r, c]:
                return False
        return True

    def _spawn(self):
        name = self.queue.pop(0)
        self.queue.append(self._draw())
        width = max(c for _, c in SHAPES[name][0]) + 1
        self.active = _Active(name, 0, 0, (self.cols - width) // 2)
        self._hold_used = False
        self._gravity_counter = self._lock_counter = self._lock_resets = 0
        self._resting = False
        if not self._fits(self.active):                # top-out
            if self.on_top_out == "reset":
                self.board[:] = 0
                self.resets += 1
                if not self._fits(self.active):
                    self.active.row = 0
            else:
                self.game_over = True
                self._events["game_over_tick"] = self.tick
        if self.mode == "ai":
            self._ai_plan = self._ai_actions()

    # ---- inputs ------------------------------------------------------------
    _ACTIONS = ("left", "right", "cw", "ccw", "flip", "soft", "hard", "hold")

    def press(self, action: str, record: bool = True):
        """Queue a player action (applied at the next tick) and record it. Buffering
        by one tick (<=1/tick_hz s) is what lets a played game replay bit-for-bit."""
        if self.game_over or self.active is None or self._clearing:
            return
        if record:
            self._inputs.append([self.tick, action])
        self._pending.append(action)

    def _apply(self, action: str):
        if self.active is None:
            return
        if action == "left":
            self._try_move(0, -1)
        elif action == "right":
            self._try_move(0, 1)
        elif action == "soft":
            if self._try_move(1, 0):
                self.score += 1
                self._gravity_counter = 0
        elif action == "hard":
            dropped = 0
            while self._try_move(1, 0, reset_lock=False):
                dropped += 1
            self.score += 2 * dropped
            self._lock()
        elif action in ("cw", "ccw", "flip"):
            self._try_rotate({"cw": 1, "ccw": -1, "flip": 2}[action])
        elif action == "hold":
            self._do_hold()

    def _try_move(self, dr: int, dc: int, reset_lock: bool = True) -> bool:
        cand = _Active(self.active.name, self.active.state,
                       self.active.row + dr, self.active.col + dc)
        if self._fits(cand):
            self.active = cand
            if reset_lock and dr == 0:
                self._reset_lock_delay()
            return True
        return False

    def _try_rotate(self, direction: int) -> bool:
        a = self.active
        if a.name == "O":
            return False
        new_state = (a.state + direction) % 4
        table = _I if a.name == "I" else _JLSTZ
        for dc, dr in table.get((a.state, new_state), [(0, 0)]):
            cand = _Active(a.name, new_state, a.row + dr, a.col + dc)
            if self._fits(cand):
                self.active = cand
                self._reset_lock_delay()
                return True
        return False

    def _reset_lock_delay(self):
        if self._resting and self._lock_resets < self.max_lock_resets:
            self._lock_counter = 0
            self._lock_resets += 1

    def _do_hold(self):
        if self._hold_used:
            return
        cur = self.active.name
        if self.hold is None:
            self.hold, self.active = cur, None
            self._spawn()
        else:
            swap = self.hold
            self.hold = cur
            width = max(c for _, c in SHAPES[swap][0]) + 1
            self.active = _Active(swap, 0, 0, (self.cols - width) // 2)
        self._hold_used = True
        self._gravity_counter = self._lock_counter = self._lock_resets = 0

    # ---- stepping ----------------------------------------------------------
    def update(self, dt: float):
        """Advance real time `dt`; runs whole fixed-rate ticks (frame-independent)."""
        if self.game_over:
            return
        self._accum += dt
        n = 0
        while self._accum >= 1.0 / self.tick_hz and not self.game_over and n < 12:
            self._accum -= 1.0 / self.tick_hz
            self.step()
            n += 1

    def step(self):
        """Advance exactly one simulation tick (applies this tick's inputs first)."""
        if self.game_over:
            return
        # gather this tick's actions from whichever source is driving the game
        if self.mode == "replay":
            actions = self._replay.get(self.tick, ())
        elif self.mode == "ai":
            actions = ()
            if self._ai_plan and not self._clearing:
                a = self._ai_plan.pop(0)
                self._inputs.append([self.tick, a])   # record so it replays uniformly
                actions = (a,)
        else:                                         # player: the buffered presses
            actions, self._pending = self._pending, []
        for action in actions:
            self._apply(action)

        if self.game_over:                            # an input topped the board out
            self.tick += 1
            return

        if self._clearing:
            self._clear_timer -= 1
            if self._clear_timer <= 0:
                self._finish_clear()
            self.tick += 1
            return

        if self.active is not None:
            if self._can_fall():
                self._resting = False
                self._gravity_counter += 1
                if self._gravity_counter >= self._gravity_ticks():
                    self._gravity_counter = 0
                    self._try_move(1, 0, reset_lock=False)
            else:                                       # resting -> lock delay
                self._resting = True
                self._lock_counter += 1
                if self._lock_counter >= self.lock_delay_ticks:
                    self._lock()
        self.tick += 1

    def _gravity_ticks(self) -> int:
        base = _GRAVITY_TICKS[min(self.level, len(_GRAVITY_TICKS) - 1)]
        return max(1, int(round(base * self.gravity_scale)))

    def _can_fall(self) -> bool:
        return self._fits(_Active(self.active.name, self.active.state,
                                  self.active.row + 1, self.active.col))

    def _lock(self):
        pid = PIECE_ID[self.active.name]
        for r, c in self._cells(self.active):
            self.board[r, c] = pid
        self.pieces_placed += 1
        self.max_stack_height = max(self.max_stack_height, self.stack_height())
        self.active = None
        full = [r for r in range(self.rows) if self.board[r].all()]
        if full:
            self._clearing = full
            self._clear_timer = max(1, self.clear_flash_ticks)
            self.line_clear_events += 1
            self.rows_cleared += len(full)
            self._events["clears"].append([self.tick, len(full)])
        else:
            self.combo = -1
            self._spawn()

    def _finish_clear(self):
        n = len(self._clearing)
        kept = np.delete(self.board, self._clearing, axis=0)
        self.board = np.vstack([np.zeros((n, self.cols), np.uint8), kept])
        self.lines += n
        if n == 4:
            self.tetrises += 1
        self.combo += 1
        self.score += _LINE_SCORE.get(n, 0) * (self.level + 1)
        if self.combo > 0:
            self.score += 50 * self.combo * (self.level + 1)
        self.level = max(self.start_level, self.lines // 10)
        self._clearing = []
        self._spawn()

    # ---- queries -----------------------------------------------------------
    def stack_height(self) -> int:
        for r in range(self.rows):
            if self.board[r].any():
                return self.rows - r
        return 0

    def ghost_cells(self):
        """Cells where the active piece would land (for the drop shadow)."""
        if self.active is None:
            return []
        a = _Active(self.active.name, self.active.state, self.active.row, self.active.col)
        while self._fits(_Active(a.name, a.state, a.row + 1, a.col)):
            a.row += 1
        return self._cells(a)

    # ---- rendering ---------------------------------------------------------
    def to_image(self, cell_px: int = 14, gap: int = 1) -> np.ndarray:
        """Grayscale frame (rows*cell x cols*cell) as float32 in [-1, 1]."""
        h, w = self.rows * cell_px, self.cols * cell_px
        arr = np.full((h, w), _BG, dtype=np.float32)

        def fill(r, c, val):
            arr[r * cell_px + gap:(r + 1) * cell_px - gap,
                c * cell_px + gap:(c + 1) * cell_px - gap] = val

        clearing = set(self._clearing)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c]:
                    fill(r, c, _FLASH if r in clearing else _SETTLED)
        if not self._clearing and self.active is not None:
            for r, c in self._cells(self.active):
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    fill(r, c, _ACTIVE)
        arr[0, :] = arr[-1, :] = arr[:, 0] = arr[:, -1] = _FRAME
        return arr

    def render_cells(self):
        """(board_id_grid, active_cells, ghost_cells, clearing_rows) for a colour draw."""
        return (self.board, self._cells(self.active) if self.active else [],
                self.ghost_cells(), list(self._clearing))

    @property
    def aspect(self) -> float:
        return self.cols / self.rows

    # ---- behaviour record / replay ----------------------------------------
    def record(self) -> dict:
        """Everything needed to replay the game bit-for-bit and to score behaviour."""
        return {
            "engine": "tetris_game/2", "mode": self.mode, "seed": self.seed,
            "cols": self.cols, "rows": self.rows, "tick_hz": self.tick_hz,
            "lock_delay_ticks": self.lock_delay_ticks,
            "clear_flash_ticks": self.clear_flash_ticks,
            "max_lock_resets": self.max_lock_resets, "on_top_out": self.on_top_out,
            "start_level": self.start_level, "gravity_scale": self.gravity_scale,
            "inputs": self._inputs, "n_ticks": self.tick, "events": self._events,
            "stats": {"score": self.score, "lines": self.lines, "level": self.level,
                      "pieces": self.pieces_placed, "line_clear_events": self.line_clear_events,
                      "rows_cleared": self.rows_cleared, "tetrises": self.tetrises,
                      "max_stack_height": self.max_stack_height, "resets": self.resets,
                      "duration_s": round(self.tick / self.tick_hz, 3),
                      "game_over": self.game_over},
        }

    @classmethod
    def from_record(cls, rec: dict) -> "TetrisGame":
        """Rebuild a REPLAY-mode game from a record (deterministic reproduction)."""
        g = cls(cols=int(rec["cols"]), rows=int(rec["rows"]), seed=int(rec["seed"]),
                mode="replay", tick_hz=float(rec.get("tick_hz", TICK_HZ)),
                lock_delay_ticks=int(rec.get("lock_delay_ticks", 30)),
                clear_flash_ticks=int(rec.get("clear_flash_ticks", 8)),
                max_lock_resets=int(rec.get("max_lock_resets", 15)),
                on_top_out=rec.get("on_top_out", "game_over"),
                start_level=int(rec.get("start_level", 0)),
                gravity_scale=float(rec.get("gravity_scale", 1.0)))
        for tk, action in rec.get("inputs", []):
            g._replay.setdefault(int(tk), []).append(action)
        return g

    # ---- heuristic autoplayer (mode="ai") ----------------------------------
    def _ai_actions(self) -> list[str]:
        """Plan the moves to reach the best placement for the current piece."""
        name = self.active.name
        best, best_score = None, -1e18
        for state in range(4 if name != "O" else 1):
            width = max(c for _, c in SHAPES[name][state]) + 1
            for col in range(0, self.cols - width + 1):
                land = self._drop_row(name, state, col)
                if land is None:
                    continue
                s = self._eval(name, state, col, land)
                if s > best_score:
                    best_score, best = s, (state, col)
        if best is None:
            return ["hard"]
        state, col = best
        plan = ["cw"] * ((state - 0) % 4)
        cur = (self.cols - (max(c for _, c in SHAPES[name][0]) + 1)) // 2
        plan += ["right" if col > cur else "left"] * abs(col - cur)
        plan.append("hard")
        return plan

    def _drop_row(self, name, state, col):
        a = _Active(name, state, 0, col)
        if not self._fits(a):
            return None
        while self._fits(_Active(name, state, a.row + 1, col)):
            a.row += 1
        return a.row

    def _eval(self, name, state, col, land):
        b = self.board.copy()
        for r, c in self._cells(_Active(name, state, land, col)):
            b[r, c] = 1
        full = [r for r in range(self.rows) if b[r].all()]
        lines = len(full)
        if lines:
            b = np.vstack([np.zeros((lines, self.cols), np.uint8),
                           np.delete(b, full, axis=0)])
        heights = [0] * self.cols
        holes = 0
        for c in range(self.cols):
            fill = np.nonzero(b[:, c])[0]
            if fill.size:
                heights[c] = self.rows - fill[0]
                holes += int((b[fill[0]:, c] == 0).sum())
        bump = sum(abs(heights[c] - heights[c + 1]) for c in range(self.cols - 1))
        return (-0.51 * sum(heights) + 0.76 * lines - 0.36 * holes - 0.18 * bump)


def _replay_to(rec: dict, n_ticks: int) -> "TetrisGame":
    g = TetrisGame.from_record(rec)
    for _ in range(n_ticks):
        g.step()
    return g


if __name__ == "__main__":
    # 1) an AI game is deterministic and its recorded inputs replay bit-for-bit;
    # 2) scripted player inputs replay bit-for-bit too.
    ai = TetrisGame(seed=7, mode="ai", on_top_out="reset")
    for _ in range(4000):
        ai.step()
    rec = ai.record()
    rep = _replay_to(rec, rec["n_ticks"])
    assert np.array_equal(ai.board, rep.board) and ai.score == rep.score, "AI replay mismatch"
    print(f"AI game: score={ai.score} lines={ai.lines} pieces={ai.pieces_placed} "
          f"clears={ai.line_clear_events} resets={ai.resets}  -> replay OK")

    rng = random.Random(3)
    pl = TetrisGame(seed=1, mode="player", on_top_out="reset")
    for _ in range(4000):
        if rng.random() < 0.30:
            pl.press(rng.choice(["left", "right", "cw", "ccw", "soft", "hard", "hold"]))
        pl.step()
    prec = pl.record()
    prep = _replay_to(prec, prec["n_ticks"])
    assert np.array_equal(pl.board, prep.board) and pl.score == prep.score, "player replay mismatch"
    print(f"player game: score={pl.score} pieces={pl.pieces_placed} "
          f"clears={pl.line_clear_events}  -> replay OK")
    print("engine determinism self-check passed")
