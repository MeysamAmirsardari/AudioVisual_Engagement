#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tetris_game.py — a self-playing, deterministic, black-and-white Tetris used as
the passive VISUAL stimulus of the cross-modal attention experiment.

Design goals (neuroscience-driven):
  * PASSIVE: an autonomous heuristic AI plays; the subject only watches.
  * GRAYSCALE: controlled luminance, no colour confound.
  * DETERMINISTIC: fully seeded (7-bag piece order + deterministic AI), so the
    exact game — and therefore the true number of line-clear events — is
    reproducible for a given (seed, duration, tick rate).
  * COUNTABLE EVENTS: cleared rows briefly FLASH before disappearing; these are
    the salient, discrete events the attend-visual task asks the subject to
    count (scored automatically against `line_clear_events`).
  * RENDERING-AGNOSTIC: the game exposes its state as a grayscale numpy array
    (values in [-1, 1], PsychoPy ImageStim convention); no PsychoPy dependency
    here, so it can be unit-tested and previewed headlessly.

The class is time-driven: call `update(dt_seconds)` each frame and then
`to_image()` to get the current frame.
"""

from __future__ import annotations

import random

import numpy as np

# ---------------------------------------------------------------------------
# Tetromino definitions (spawn orientation; (row, col), row increases downward)
# ---------------------------------------------------------------------------
_BASE = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}


def _normalize(cells):
    minr = min(r for r, _ in cells)
    minc = min(c for _, c in cells)
    return tuple(sorted((r - minr, c - minc) for r, c in cells))


def _rotate_cw(cells):
    return _normalize([(c, -r) for r, c in cells])


def _all_rotations(cells):
    rots, cur = [], _normalize(cells)
    for _ in range(4):
        if cur not in rots:
            rots.append(cur)
        cur = _rotate_cw(cur)
    return rots


# Precompute the unique rotation states for each piece.
_PIECES = {name: _all_rotations(cells) for name, cells in _BASE.items()}
_PIECE_NAMES = list(_PIECES.keys())

# Grayscale shades in PsychoPy image range [-1 (black) .. +1 (white)].
_BG = -1.0        # empty background
_FRAME = -0.35    # board frame
_SETTLED = 0.35   # locked blocks
_ACTIVE = 0.85    # the falling piece
_FLASH = 1.0      # rows about to clear (the countable event)

# Heuristic weights (Yiyuan Lee's well-known strong Tetris AI).
_W_HEIGHT, _W_LINES, _W_HOLES, _W_BUMP = -0.510066, 0.760666, -0.35663, -0.184483


class TetrisGame:
    def __init__(self, cols=10, rows=20, seed=0,
                 tick_interval_s=0.06, flash_s=0.18):
        self.cols, self.rows = cols, rows
        self.rng = random.Random(seed)
        self.tick_interval = float(tick_interval_s)
        self.flash_s = float(flash_s)

        self.board = np.zeros((rows, cols), dtype=np.uint8)
        self._bag: list[str] = []

        # Event counters exposed for the attention probe.
        self.line_clear_events = 0     # number of clear MOMENTS
        self.rows_cleared = 0          # total rows removed
        self.pieces_placed = 0
        self.resets = 0                # board top-outs (should be ~0 with the AI)
        self.max_stack_height = 0      # peak stack height in rows (the probe answer)

        # Timing / animation state.
        self._tick_accum = 0.0
        self._clearing: list[int] = []
        self._clear_timer = 0.0

        self._spawn()

    # -- piece helpers ----------------------------------------------------
    def _next_name(self) -> str:
        if not self._bag:
            self._bag = _PIECE_NAMES[:]
            self.rng.shuffle(self._bag)
        return self._bag.pop()

    def _cells(self, rot, r_off, c_off, name=None):
        rots = _PIECES[name or self.name]
        return [(r + r_off, c + c_off) for r, c in rots[rot]]

    def _valid(self, rot, r_off, c_off, name=None):
        for r, c in self._cells(rot, r_off, c_off, name):
            if c < 0 or c >= self.cols or r < 0 or r >= self.rows:
                return False
            if self.board[r, c]:
                return False
        return True

    def _clamp_col(self, rot, c_off, name=None):
        """Shift c_off so the piece stays within the horizontal bounds."""
        cols = [c + c_off for _, c in _PIECES[name or self.name][rot]]
        if min(cols) < 0:
            c_off -= min(cols)
        if max(cols) >= self.cols:
            c_off -= (max(cols) - self.cols + 1)
        return c_off

    def _spawn(self):
        self.name = self._next_name()
        width = max(c for _, c in _PIECES[self.name][0]) + 1
        self.rot = 0
        self.r_off = 0
        self.c_off = (self.cols - width) // 2
        if not self._valid(self.rot, self.r_off, self.c_off):
            # Top-out: clear the board and continue so the stimulus never freezes.
            self.board[:] = 0
            self.resets += 1
        self.target_rot, self.target_c = self._ai_choose()
        self.phase = "align"

    # -- AI ---------------------------------------------------------------
    def _ai_choose(self):
        """Pick the (rotation, col-offset) that maximises the heuristic score."""
        best, best_score = (0, self.c_off), -1e9
        for rot in range(len(_PIECES[self.name])):
            width = max(c for _, c in _PIECES[self.name][rot]) + 1
            for c_off in range(0, self.cols - width + 1):
                r_off = self._landing_row(rot, c_off)
                if r_off is None:
                    continue
                score = self._score_placement(rot, r_off, c_off)
                if score > best_score:
                    best_score, best = score, (rot, c_off)
        return best

    def _landing_row(self, rot, c_off):
        if not self._valid(rot, 0, c_off):
            return None
        r_off = 0
        while self._valid(rot, r_off + 1, c_off):
            r_off += 1
        return r_off

    def _score_placement(self, rot, r_off, c_off):
        b = self.board.copy()
        for r, c in self._cells(rot, r_off, c_off):
            b[r, c] = 1
        # lines cleared
        full = [r for r in range(self.rows) if b[r].all()]
        lines = len(full)
        if lines:
            b = np.delete(b, full, axis=0)
            b = np.vstack([np.zeros((lines, self.cols), np.uint8), b])
        # column heights
        heights = [0] * self.cols
        holes = 0
        for c in range(self.cols):
            col = b[:, c]
            filled = np.nonzero(col)[0]
            if filled.size:
                top = filled[0]
                heights[c] = self.rows - top
                holes += int((col[top:] == 0).sum())
        agg = sum(heights)
        bump = sum(abs(heights[c] - heights[c + 1]) for c in range(self.cols - 1))
        return (_W_HEIGHT * agg + _W_LINES * lines
                + _W_HOLES * holes + _W_BUMP * bump)

    # -- stepping ---------------------------------------------------------
    def update(self, dt: float):
        """Advance the game by `dt` seconds (frame-rate independent)."""
        if self._clearing:                     # currently flashing rows -> pause
            self._clear_timer -= dt
            if self._clear_timer <= 0:
                self._finish_clear()
            return
        self._tick_accum += dt
        # Cap catch-up so a long frame can't run away.
        steps = 0
        while self._tick_accum >= self.tick_interval and not self._clearing \
                and steps < 8:
            self._tick_accum -= self.tick_interval
            self._tick()
            steps += 1

    def _tick(self):
        if self.phase == "align":
            if self.rot != self.target_rot:
                self.rot = (self.rot + 1) % len(_PIECES[self.name])
                self.c_off = self._clamp_col(self.rot, self.c_off)
            elif self.c_off != self.target_c:
                self.c_off += 1 if self.target_c > self.c_off else -1
            else:
                self.phase = "fall"
            return
        # fall phase
        if self._valid(self.rot, self.r_off + 1, self.c_off):
            self.r_off += 1
        else:
            self._lock()

    def stack_height(self) -> int:
        """Current stack height in rows (0 = empty board)."""
        for r in range(self.rows):
            if self.board[r].any():
                return self.rows - r
        return 0

    def _lock(self):
        for r, c in self._cells(self.rot, self.r_off, self.c_off):
            self.board[r, c] = 1
        self.pieces_placed += 1
        # Peak stack height is captured right after locking, before any clear.
        self.max_stack_height = max(self.max_stack_height, self.stack_height())
        full = [r for r in range(self.rows) if self.board[r].all()]
        if full:
            self._clearing = full
            self._clear_timer = self.flash_s
            self.line_clear_events += 1
            self.rows_cleared += len(full)
        else:
            self._spawn()

    def _finish_clear(self):
        b = np.delete(self.board, self._clearing, axis=0)
        self.board = np.vstack(
            [np.zeros((len(self._clearing), self.cols), np.uint8), b])
        self._clearing = []
        self._spawn()

    # -- rendering --------------------------------------------------------
    def to_image(self, cell_px: int = 14, gap: int = 1) -> np.ndarray:
        """Grayscale frame (rows*cell x cols*cell) as float32 in [-1, 1]."""
        h, w = self.rows * cell_px, self.cols * cell_px
        arr = np.full((h, w), _BG, dtype=np.float32)

        def fill(r, c, val):
            y0, y1 = r * cell_px + gap, (r + 1) * cell_px - gap
            x0, x1 = c * cell_px + gap, (c + 1) * cell_px - gap
            arr[y0:y1, x0:x1] = val

        clearing = set(self._clearing)
        for r in range(self.rows):
            val = _FLASH if r in clearing else _SETTLED
            for c in range(self.cols):
                if self.board[r, c]:
                    fill(r, c, val)
        if not self._clearing:                 # draw the active piece
            for r, c in self._cells(self.rot, self.r_off, self.c_off):
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    fill(r, c, _ACTIVE)

        # Thin board frame.
        arr[0, :] = arr[-1, :] = _FRAME
        arr[:, 0] = arr[:, -1] = _FRAME
        return arr

    @property
    def aspect(self) -> float:
        """Board width / height ratio (for on-screen sizing)."""
        return self.cols / self.rows


# ---------------------------------------------------------------------------
# Headless self-test: simulate a block and report stats (+ optional PNG frames)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Simulate the self-playing Tetris.")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-frames", type=int, default=0,
                    help="Save this many evenly-spaced PNG frames to /tmp.")
    args = ap.parse_args()

    g = TetrisGame(seed=args.seed)
    dt = 1.0 / args.fps
    n = int(args.seconds * args.fps)
    save_at = (set(range(0, n, max(1, n // args.save_frames)))
               if args.save_frames else set())
    for i in range(n):
        g.update(dt)
        if i in save_at:
            try:
                from PIL import Image
                img = ((g.to_image() + 1) * 127.5).astype(np.uint8)
                Image.fromarray(img, "L").save(f"/tmp/tetris_{i:05d}.png")
            except Exception as exc:  # pragma: no cover
                print("frame save skipped:", exc)
    print(f"seed={args.seed} {args.seconds}s @ {args.fps}fps:")
    print(f"  pieces placed     : {g.pieces_placed}")
    print(f"  line-clear events : {g.line_clear_events}")
    print(f"  rows cleared      : {g.rows_cleared}")
    print(f"  max stack height  : {g.max_stack_height}  (the probe answer)")
    print(f"  board resets      : {g.resets}")
