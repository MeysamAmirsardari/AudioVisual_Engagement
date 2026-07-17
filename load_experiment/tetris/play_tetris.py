#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
play_tetris.py — play the real Tetris with the keyboard, and save the behaviour.

    python tetris/play_tetris.py --name alice

Controls
    Left / Right   move            (auto-repeat with DAS/ARR)
    Down           soft drop
    Up  or  X      rotate clockwise
    Z  or  Ctrl    rotate counter-clockwise
    Space          hard drop
    C  or  Shift   hold
    P              pause      R  restart      Esc / close  quit

On game over or quit the full game record (seed + every keystroke by simulation
tick + the final stats) is written to a JSON file, so the exact game can be
replayed and the behaviour analysed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tetris.tetris_game import COLORS, TetrisGame, PIECE_ID  # noqa: E402

CELL = 30
PAD = 24
SIDEBAR = 200
DAS_S, ARR_S = 0.14, 0.03          # delayed-auto-shift / auto-repeat-rate
SOFT_S = 0.03


def _colour(pid, ghost=False):
    r, g, b = COLORS.get(int(pid), (200, 200, 200))
    return (r // 4, g // 4, b // 4) if ghost else (r, g, b)


def save_record(game: TetrisGame, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"tetris_{name}_{ts}.json")
    rec = game.record()
    rec["player"] = name
    rec["saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description="Play Tetris and record the behaviour.")
    ap.add_argument("--name", default="player")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--level", type=int, default=0, help="starting speed (0 slow .. higher fast)")
    ap.add_argument("--gravity-scale", type=float, default=1.0,
                    help="fine speed knob (0.5 = 2x faster, 2.0 = half speed)")
    ap.add_argument("--auto", action="store_true", help="the AI plays (attract mode / demo)")
    ap.add_argument("--frames", type=int, default=0, help="auto-quit after N frames (testing)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "behavior"))
    args = ap.parse_args()

    import pygame
    pygame.init()
    W = args.cols * CELL + 2 * PAD + SIDEBAR
    H = args.rows * CELL + 2 * PAD
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()
    big = pygame.font.SysFont("Menlo,Consolas,monospace", 30, bold=True)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 18)

    def new_game():
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
        return TetrisGame(cols=args.cols, rows=args.rows, seed=seed,
                          mode="ai" if args.auto else "player",
                          start_level=args.level, gravity_scale=args.gravity_scale,
                          on_top_out="game_over"), seed

    game, seed = new_game()
    held = {}                       # key -> {"timer","phase"} for auto-repeat
    saved_paths, paused = [], False

    def draw_block(px, py, colour, size=CELL):
        pygame.draw.rect(screen, colour, (px, py, size - 1, size - 1))
        pygame.draw.rect(screen, tuple(min(255, c + 40) for c in colour),
                         (px, py, size - 1, 4))

    running, frame = True, 0
    while running:
        dt = clock.tick(60) / 1000.0
        frame += 1
        if args.frames and frame > args.frames:
            running = False

        # -- input -----------------------------------------------------------
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_p:
                    paused = not paused
                elif k == pygame.K_r:
                    saved_paths.append(save_record(game, args.name, args.out))
                    game, seed = new_game(); held.clear(); paused = False
                elif not paused and not game.game_over and not args.auto:
                    if k in (pygame.K_LEFT, pygame.K_RIGHT):
                        game.press("left" if k == pygame.K_LEFT else "right")
                        held[k] = 0.0
                    elif k == pygame.K_DOWN:
                        game.press("soft"); held[k] = 0.0
                    elif k in (pygame.K_UP, pygame.K_x):
                        game.press("cw")
                    elif k in (pygame.K_z, pygame.K_RCTRL, pygame.K_LCTRL):
                        game.press("ccw")
                    elif k == pygame.K_SPACE:
                        game.press("hard")
                    elif k in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT):
                        game.press("hold")
            elif e.type == pygame.KEYUP:
                held.pop(e.key, None)

        # -- advance: AI drives in auto mode; otherwise apply held-key auto-repeat --
        if not paused and not game.game_over:
            if not args.auto:
                for k, t in list(held.items()):
                    held[k] = t + dt
                    if k in (pygame.K_LEFT, pygame.K_RIGHT):
                        if held[k] >= DAS_S:
                            while held[k] - DAS_S >= 0:
                                game.press("left" if k == pygame.K_LEFT else "right")
                                held[k] -= ARR_S
                            held[k] += DAS_S
                    elif k == pygame.K_DOWN:
                        while held[k] >= SOFT_S:
                            game.press("soft"); held[k] -= SOFT_S
            game.update(dt)

        # -- draw ------------------------------------------------------------
        screen.fill((14, 16, 22))
        board, active, ghost, clearing = game.render_cells()
        ox, oy = PAD, PAD
        pygame.draw.rect(screen, (40, 44, 54),
                         (ox - 3, oy - 3, args.cols * CELL + 5, args.rows * CELL + 5), 3)
        for r in range(args.rows):                       # settled + clear flash
            for c in range(args.cols):
                if board[r, c]:
                    col = (255, 255, 255) if r in clearing else _colour(board[r, c])
                    draw_block(ox + c * CELL, oy + r * CELL, col)
        for r, c in ghost:                               # drop shadow
            if 0 <= r < args.rows:
                pygame.draw.rect(screen, _colour(PIECE_ID[game.active.name], ghost=True),
                                 (ox + c * CELL, oy + r * CELL, CELL - 1, CELL - 1), 2)
        for r, c in active:                              # falling piece
            if 0 <= r < args.rows:
                draw_block(ox + c * CELL, oy + r * CELL, _colour(PIECE_ID[game.active.name]))

        sx = ox + args.cols * CELL + 24                  # sidebar
        screen.blit(big.render("TETRIS", True, (235, 235, 235)), (sx, oy))
        for i, (label, val) in enumerate([("score", game.score), ("lines", game.lines),
                                          ("level", game.level), ("pieces", game.pieces_placed)]):
            screen.blit(small.render(f"{label:>6}: {val}", True, (200, 205, 215)),
                        (sx, oy + 46 + i * 24))
        screen.blit(small.render("next", True, (170, 175, 185)), (sx, oy + 170))
        _mini(screen, game.queue[0], sx, oy + 194, draw_block)
        screen.blit(small.render("hold", True, (170, 175, 185)), (sx, oy + 280))
        if game.hold:
            _mini(screen, game.hold, sx, oy + 304, draw_block)

        if paused:
            _overlay(screen, big, small, "PAUSED", "press P to resume", W, H)
        elif game.game_over:
            _overlay(screen, big, small, "GAME OVER",
                     f"score {game.score} · R to restart", W, H)
        pygame.display.flip()

    if game.pieces_placed and (game.game_over or not saved_paths):
        saved_paths.append(save_record(game, args.name, args.out))
    pygame.quit()
    for p in saved_paths:
        print("saved behaviour ->", p)


def _mini(screen, name, x, y, draw_block):
    from tetris.tetris_game import SHAPES, PIECE_ID
    for r, c in SHAPES[name][0]:
        draw_block(x + c * 20, y + r * 20, _colour(PIECE_ID[name]), size=20)


def _overlay(screen, big, small, title, sub, W, H):
    import pygame
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    s.fill((0, 0, 0, 170))
    screen.blit(s, (0, 0))
    t = big.render(title, True, (255, 255, 255))
    screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - 30))
    u = small.render(sub, True, (210, 210, 210))
    screen.blit(u, (W // 2 - u.get_width() // 2, H // 2 + 12))


if __name__ == "__main__":
    main()
