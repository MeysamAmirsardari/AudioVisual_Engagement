#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io.py — load the raw EEG and the photodiode-aligned trial table.

Reuses the hard-won loading + alignment from eeg_analysis/cma_eeg (the
BrainVision filename patch, photodiode edge pairing, and wall-clock alignment to
the behaviour log). Returns the continuous raw plus a per-trial table with the
av-onset sample and the attended-modality label.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

# Project root (this file is stim_decoding/sda/io.py).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EEG = os.path.join(ROOT, "eeg_analysis")
if _EEG not in sys.path:
    sys.path.insert(0, _EEG)

from cma_eeg import alignment, loading  # noqa: E402  (after sys.path insert)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def abspath(p: str, base: str = ROOT) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_raw_and_events(cfg: dict):
    """
    Return (raw, events_df, ali). events_df columns: trial, label, gap_sample,
    av_sample, gap_dur_eeg, delay_log, audio_stim, av_reconstructed.

    Optionally RECOVERS trials whose av-onset photodiode edge was dropped: the
    gap-onset edge is present for every trial (verified), so the av-onset is
    reconstructed as gap_edge + logged delay_seconds. This is validated against
    ground truth (on trials with both edges, gap_edge+delay hits the real av edge
    to <8 ms). Controlled by alignment.recover_missing_av.
    """
    ds = cfg["dataset"]
    vhdr = abspath(ds["vhdr"])
    vmrk = os.path.splitext(vhdr)[0] + ".vmrk"

    raw = loading.load_raw(vhdr)
    sf = raw.info["sfreq"]
    markers = loading.load_markers(
        vmrk, sf, edge_code=cfg["markers"]["edge_code"])

    beh = abspath(ds["behavior"]) if ds.get("behavior") else \
        alignment.discover_behavior(vmrk)
    ali = alignment.align(
        markers, beh,
        match_tol_s=cfg["alignment"]["match_tol_s"],
        offset_search_s=cfg["alignment"]["offset_search_s"],
        gap_lo_s=cfg["markers"]["gap_lo_s"],
        gap_hi_s=cfg["markers"]["gap_hi_s"])

    events: pd.DataFrame = ali.events.copy()
    events["av_reconstructed"] = False
    if cfg["alignment"].get("recover_missing_av", False):
        events = _recover_missing_av(events, markers, ali, cfg, beh, sf)
    return raw, events, ali


def _recover_missing_av(events, markers, ali, cfg, beh_path, sf):
    """Add gap-only trials with a reconstructed av-onset (gap_edge + delay)."""
    import json
    import numpy as np
    with open(beh_path, encoding="utf-8") as f:
        sess = json.load(f)
    tol = cfg["alignment"]["match_tol_s"]
    off, rec0 = ali.offset_s, markers.rec_start_unix
    edge_unix = rec0 + (np.asarray(markers.edge_samples) - 1) / sf
    have = {int(t) for t in events["trial"]}
    rows = events.to_dict("records")

    n_rec = 0
    for t in sess["trials"]:
        tr = int(t["trial"])
        if tr in have:
            continue
        g, dl = t.get("gap_onset_wall_clock"), t.get("delay_seconds")
        if g is None or dl is None:
            continue
        target = float(g) + off                      # gap-onset in edge-unix time
        i = int(np.argmin(np.abs(edge_unix - target)))
        if abs(edge_unix[i] - target) > tol:         # gap edge NOT captured -> skip
            continue
        gap_sample = int(markers.edge_samples[i])
        av_sample = gap_sample + int(round(float(dl) * sf))
        rows.append({
            "trial": tr, "label": t["attended_modality"],
            "gap_sample": gap_sample, "av_sample": av_sample,
            "gap_dur_eeg": round(float(dl), 4), "delay_log": round(float(dl), 4),
            "dur_err_ms": 0.0, "audio_stim": t.get("audio_stim_id"),
            "av_reconstructed": True,
        })
        n_rec += 1
    out = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    print(f"  recovered {n_rec} trials via av = gap_edge + delay "
          f"(now {len(out)} usable)")
    return out


def load_behavior_trials(cfg: dict) -> dict:
    """Map trial index -> behaviour trial dict (attended_modality, ids, etc.)."""
    import json
    with open(abspath(cfg["dataset"]["behavior"]), encoding="utf-8") as f:
        sess = json.load(f)
    return {int(t["trial"]): t for t in sess["trials"]}


def game_record_path(cfg: dict, trial: int) -> str | None:
    """Find the Tetris reconstruction record for a trial (by the _tNNN suffix)."""
    import glob
    gdir = abspath(cfg["dataset"]["games_dir"])
    hits = glob.glob(os.path.join(gdir, f"*_t{int(trial):03d}.json"))
    return hits[0] if hits else None
