#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_video_segments.py — find the usable BLACK-AND-WHITE game segments.

The tetris clip contains non-game (colour) sections — menus, intros, etc. — that
must NOT be used as stimuli. The actual gameplay is black-and-white (near-zero
colour saturation), so this script scans the video, classifies each sampled
frame as grayscale vs colour by its mean HSV saturation, and writes the
contiguous grayscale intervals that are long enough to hold a full block.

run_block.py then draws its random cuts ONLY from those intervals.

    python build_video_segments.py            # uses config.yaml
    python build_video_segments.py --preview  # also print every segment

Output: visual.video.segments_file  (JSON list of {start_s, end_s, duration_s}).
All parameters live under visual.video.filter in config.yaml.
"""

from __future__ import annotations

import argparse
import json
import os

import cma_common as cc


def _merge_and_filter(intervals, merge_gap_s, min_len_s):
    """Merge intervals separated by <= merge_gap_s, then keep those >= min_len_s."""
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= merge_gap_s:      # bridge a brief colour blip
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if (e - s) >= min_len_s]


def analyze_video(video_path, sample_interval_s, sat_threshold,
                  min_segment_s, merge_gap_s):
    """Return (duration_s, fps, [(start_s, end_s), ...]) for grayscale segments."""
    import cv2
    import numpy as np  # noqa: F401  (cv2 pulls it; kept explicit for clarity)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(sample_interval_s * fps)))
    sample_dt = step / fps
    duration = (total / fps) if total else 0.0
    cc.log(f"Scanning {os.path.basename(video_path)}: {fps:.2f} fps, {total} "
           f"frames, {duration:.1f}s; analysing 1 frame / {step} "
           f"(~{sample_interval_s:.2f}s), sat<{sat_threshold} => grayscale.")

    intervals = []
    run_start = None
    prev_gray_t = None
    idx = 0
    next_report = 0.1
    while True:
        if not cap.grab():                 # advance without full decode
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                sat = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                t = idx / fps
                is_gray = sat < sat_threshold
                if is_gray:
                    if run_start is None:
                        run_start = t
                    prev_gray_t = t
                elif run_start is not None:
                    intervals.append((run_start, prev_gray_t + sample_dt))
                    run_start = None
                if duration and t / duration >= next_report:
                    cc.log(f"  ... {int(100 * t / duration)}%")
                    next_report += 0.1
        idx += 1
    if run_start is not None:
        intervals.append((run_start, (prev_gray_t or run_start) + sample_dt))
    cap.release()

    segments = _merge_and_filter(intervals, merge_gap_s, min_segment_s)
    return duration, fps, segments


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect black-and-white game segments.")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--preview", action="store_true", help="Print each segment.")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    vcfg = cfg["visual"]["video"]
    flt = vcfg.get("filter", {}) or {}
    video_path = cc.abspath(vcfg["file"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = cc.abspath(vcfg.get("segments_file", "stims/video/segments.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    block_seconds = float(cfg["experiment"]["block_seconds"])
    min_segment_s = float(flt.get("min_segment_s", block_seconds))

    print("=" * 72)
    print("DETECTING BLACK-AND-WHITE GAME SEGMENTS")
    print("=" * 72)
    duration, fps, segments = analyze_video(
        video_path,
        float(flt.get("sample_interval_s", 0.2)),
        float(flt.get("saturation_threshold", 12)),
        min_segment_s,
        float(flt.get("merge_gap_s", 0.6)),
    )

    total_valid = sum(e - s for s, e in segments)
    payload = {
        "video": os.path.relpath(video_path, cc.BASE_DIR),
        "video_duration_s": round(duration, 3),
        "fps": round(fps, 3),
        "block_seconds": block_seconds,
        "params": {
            "sample_interval_s": float(flt.get("sample_interval_s", 0.2)),
            "saturation_threshold": float(flt.get("saturation_threshold", 12)),
            "min_segment_s": min_segment_s,
            "merge_gap_s": float(flt.get("merge_gap_s", 0.6)),
        },
        "n_segments": len(segments),
        "total_valid_s": round(total_valid, 3),
        "segments": [{"start_s": round(s, 3), "end_s": round(e, 3),
                      "duration_s": round(e - s, 3)} for s, e in segments],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 72)
    cc.log(f"Found {len(segments)} game segment(s), {total_valid:.1f}s usable "
           f"of {duration:.1f}s ({100 * total_valid / duration:.0f}%).")
    if args.preview:
        for i, seg in enumerate(payload["segments"], 1):
            cc.log(f"  seg {i:2d}: {seg['start_s']:7.1f} -> {seg['end_s']:7.1f} s "
                   f"({seg['duration_s']:.1f}s)")
    cc.log(f"Wrote {out_path}")
    if not segments:
        cc.log("WARNING: no usable segments found. Try raising "
               "saturation_threshold or lowering min_segment_s in config.yaml.")


if __name__ == "__main__":
    main()
