#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruct_tetris.py — render the Tetris visual stimuli to MP4, AFTER a session.

During the experiment run_block.py saves one reconstruction record per Tetris
trial in games/ (seed + parameters + block duration + the exact display frame
times). The game is fully deterministic, so this script replays each game and
encodes a faithful grayscale MP4 — no frames are stored during the experiment.

Two reconstruction modes:
  * default        — replay at a constant fps (the recorded refresh rate, or
                     --fps), giving a clean, content-faithful video.
  * --exact        — replay using the recorded per-frame display times, so the
                     video reproduces exactly what was on screen, frame for frame.

Usage:
    python reconstruct_tetris.py                       # every record in games/
    python reconstruct_tetris.py games/sub01_..._t001.json
    python reconstruct_tetris.py --subject sub01 --exact
    python reconstruct_tetris.py --fps 60 --out-dir games/mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import cma_common as cc
from tetris.tetris_game import TetrisGame


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """[-1,1] grayscale -> HxWx3 uint8 (row 0 = top; correct for video)."""
    a = ((np.clip(img, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2)


def reconstruct(record_path: str, out_dir: str, fps: float | None,
                exact: bool) -> str:
    with open(record_path, encoding="utf-8") as f:
        rec = json.load(f)

    if "inputs" in rec or str(rec.get("engine", "")).startswith("tetris_game"):
        game = TetrisGame.from_record(rec)             # replay the player's exact game
    else:                                              # legacy self-playing record
        game = TetrisGame(cols=int(rec["cols"]), rows=int(rec["rows"]),
                          seed=int(rec["seed"]), mode="ai", on_top_out="reset")
    cell = int(rec.get("render_cell_px", 14))
    duration = float(rec.get("block_duration_s", 0.0))

    frames = []
    if exact and rec.get("frame_times_s"):
        times = rec["frame_times_s"]
        out_fps = float(rec.get("refresh_hz") or (len(times) / duration
                                                  if duration else 60.0))
        prev = 0.0
        for t in times:
            game.update(float(t) - prev)
            prev = float(t)
            frames.append(_to_rgb_uint8(game.to_image(cell)))
    else:
        out_fps = float(fps or rec.get("refresh_hz") or 60.0)
        dt = 1.0 / out_fps
        for _ in range(int(round(duration * out_fps))):
            game.update(dt)
            frames.append(_to_rgb_uint8(game.to_image(cell)))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(record_path))[0]
    out_mp4 = os.path.join(out_dir, stem + ".mp4")

    import imageio  # imageio-ffmpeg backend (already installed)
    writer = imageio.get_writer(
        out_mp4, fps=out_fps, codec="libx264", quality=8,
        macro_block_size=None, ffmpeg_log_level="error")
    for fr in frames:
        writer.append_data(fr)
    writer.close()

    # Sanity: the deterministic replay must reproduce the recorded ground truth.
    ok = (game.max_stack_height == rec.get("max_stack_height")
          and game.line_clear_events == rec.get("line_clear_events"))
    cc.log(f"{stem}: {len(frames)} frames @ {out_fps:.1f}fps -> {out_mp4} "
           f"[max_height={game.max_stack_height}, clears={game.line_clear_events}, "
           f"{'MATCHES record' if ok else 'MISMATCH vs record!'}]")
    return out_mp4


def main() -> None:
    ap = argparse.ArgumentParser(description="Render saved Tetris games to mp4.")
    ap.add_argument("records", nargs="*", help="Specific record .json files.")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--subject", default=None, help="Only this subject's records.")
    ap.add_argument("--out-dir", default=None, help="Where to write mp4s.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Constant reconstruction fps (default: recorded refresh).")
    ap.add_argument("--exact", action="store_true",
                    help="Use recorded per-frame display times (frame-exact).")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    paths = cc.Paths.from_config(cfg)
    out_dir = args.out_dir or os.path.join(paths.games_dir, "mp4")

    if args.records:
        records = args.records
    else:
        pattern = f"{args.subject}_*.json" if args.subject else "*.json"
        records = sorted(glob.glob(os.path.join(paths.games_dir, pattern)))
    if not records:
        cc.log(f"No Tetris records found in {paths.games_dir}.")
        return

    cc.log(f"Reconstructing {len(records)} game(s) "
           f"({'frame-exact' if args.exact else 'constant fps'}) -> {out_dir}")
    for rp in records:
        reconstruct(rp, out_dir, args.fps, args.exact)


if __name__ == "__main__":
    main()
