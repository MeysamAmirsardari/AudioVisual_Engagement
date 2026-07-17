"""Align photodiode edges to behaviour trials -> labelled decoding events.

The amplifier clock and the stimulus-PC clock differ by a single constant
offset. We find that offset by sliding it until the largest number of photodiode
edges fall onto a behaviour event (a gap-onset or an av-onset), then classify
every edge and keep the trials for which BOTH the gap-onset and the av-onset
edge were captured. Each surviving trial gives us:

    * gap_sample  — instruction offset / anticipatory-gap onset  (decode t = 0)
    * av_sample   — audiovisual stimulus onset
    * label       — attended modality (Audio / Visual)  [the decoding target]

Robustness: this never assumes strict edge alternation (the photodiode drops
edges), and it cross-checks each kept trial's EEG gap duration against the
logged ``delay_seconds`` so a mis-pairing cannot pass silently.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loading import Markers, _parse_new_segment_unix
from .utils import LOG


@dataclass
class AlignmentResult:
    events: pd.DataFrame          # one row per usable trial (see columns below)
    offset_s: float               # estimated amp-clock minus behaviour-clock
    n_edges: int
    match_error_ms: float         # median |edge - matched event| after offset
    behavior_path: str


# ---------------------------------------------------------------------------
# Behaviour loading / auto-discovery
# ---------------------------------------------------------------------------
def _read_behavior(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for t in d["trials"]:
        if t.get("aborted"):
            continue
        gap = t.get("gap_onset_wall_clock")
        av = t.get("webcam_av_onset_wallclock")
        if gap is None or av is None:
            continue
        rows.append(dict(
            trial=t["trial"], label=t["attended_modality"],
            gap_unix=float(gap), av_unix=float(av),
            delay_s=float(t["delay_seconds"]),
            audio_stim=t.get("audio_stim_id"),
        ))
    return pd.DataFrame(rows)


def discover_behavior(vmrk_path: str, search_roots=("data", "behavior")) -> str:
    """Pick the behaviour log whose session started just after this recording.

    Matches by (a) closest start time to the amplifier ``New Segment`` and
    (b) the most trials — the real EEG session is a full 60-trial block.
    """
    vmrk_text = open(vmrk_path, "r", encoding="utf-8", errors="replace").read()
    rec_start = _parse_new_segment_unix(vmrk_text)
    cands = []
    for root in search_roots:
        cands += glob.glob(os.path.join(root, "**", "*session*.json"),
                           recursive=True)
    best, best_key = None, None
    for p in cands:
        try:
            with open(p) as f:
                d = json.load(f)
            n = int(d.get("n_trials_run", 0))
            ts = d.get("session_timestamp")            # e.g. 20260701_163520
            if not ts or n < 1:
                continue
            from datetime import datetime
            start = datetime.strptime(ts, "%Y%m%d_%H%M%S").timestamp()
            dt = abs(start - rec_start)
            if dt > 900:                               # within 15 min of rec start
                continue
            key = (-n, dt)                             # most trials, then closest
            if best_key is None or key < best_key:
                best_key, best = key, p
        except Exception:
            continue
    if best is None:
        raise FileNotFoundError(
            "Could not auto-discover a behaviour log near the recording start; "
            "pass dataset.behavior explicitly.")
    LOG.info("Auto-discovered behaviour log: %s", best)
    return best


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align(markers: Markers, behavior_path: str,
          match_tol_s: float = 0.060, offset_search_s: float = 3.0,
          gap_lo_s: float = 1.4, gap_hi_s: float = 2.6) -> AlignmentResult:
    beh = _read_behavior(behavior_path)
    if beh.empty:
        raise RuntimeError(f"No usable trials in {behavior_path}")

    edge_unix = markers.rec_start_unix + (markers.edge_samples - 1) / markers.sfreq
    ev_time = np.concatenate([beh.gap_unix.values, beh.av_unix.values])
    ev_kind = np.array(["gap"] * len(beh) + ["av"] * len(beh))
    ev_trow = np.concatenate([np.arange(len(beh)), np.arange(len(beh))])

    # ---- 1. robust clock offset: maximise the number of matched edges --------
    coarse = float(np.median(edge_unix[:1]) - beh.gap_unix.values[0])
    grid = coarse + np.arange(-offset_search_s, offset_search_s, 0.001)
    inliers = np.empty(grid.size)
    for i, off in enumerate(grid):
        d = np.abs(edge_unix[:, None] - off - ev_time[None, :]).min(axis=1)
        inliers[i] = (d < match_tol_s).sum()
    offset = float(grid[int(np.argmax(inliers))])

    # refine: median residual of the matched edges
    corr = edge_unix - offset
    nn = np.abs(corr[:, None] - ev_time[None, :])
    jmin = nn.argmin(axis=1)
    resid = corr - ev_time[jmin]
    inlier = np.abs(resid) < match_tol_s
    if inlier.any():
        offset += float(np.median(resid[inlier]))
        corr = edge_unix - offset
        nn = np.abs(corr[:, None] - ev_time[None, :])
        jmin = nn.argmin(axis=1)
    match_err = float(np.median(np.abs(corr - ev_time[jmin])[inlier])) if inlier.any() else np.nan

    # ---- 2. classify each edge -> (trial, kind); keep closest per (trial,kind)
    per = {}   # (trial, kind) -> (edge_sample, error)
    for e_samp, e_t in zip(markers.edge_samples, corr):
        j = int(np.argmin(np.abs(ev_time - e_t)))
        err = abs(ev_time[j] - e_t)
        if err > match_tol_s:
            continue
        key = (int(ev_trow[j]), ev_kind[j])
        if key not in per or err < per[key][1]:
            per[key] = (int(e_samp), err)

    # ---- 3. build trials that have BOTH edges, and validate gap duration -----
    rows = []
    for ti in range(len(beh)):
        g, a = per.get((ti, "gap")), per.get((ti, "av"))
        if g is None or a is None:
            continue
        gap_dur = (a[0] - g[0]) / markers.sfreq
        if not (gap_lo_s <= gap_dur <= gap_hi_s):
            continue
        b = beh.iloc[ti]
        rows.append(dict(
            trial=int(b.trial), label=b.label,
            gap_sample=g[0], av_sample=a[0],
            gap_dur_eeg=round(gap_dur, 4), delay_log=round(float(b.delay_s), 4),
            dur_err_ms=round(abs(gap_dur - float(b.delay_s)) * 1000, 1),
            audio_stim=b.audio_stim,
        ))
    events = pd.DataFrame(rows).sort_values("gap_sample").reset_index(drop=True)

    LOG.info("Clock offset %.3f s | median edge-match error %.1f ms",
             offset, match_err * 1000 if not np.isnan(match_err) else -1)
    if not events.empty:
        LOG.info("Usable trials: %d  (Audio %d / Visual %d) | "
                 "gap-duration cross-check max err %.1f ms",
                 len(events), int((events.label == "Audio").sum()),
                 int((events.label == "Visual").sum()), events.dur_err_ms.max())
    return AlignmentResult(events=events, offset_s=offset,
                           n_edges=len(markers.edge_samples),
                           match_error_ms=(match_err * 1000 if not np.isnan(match_err) else float("nan")),
                           behavior_path=behavior_path)
