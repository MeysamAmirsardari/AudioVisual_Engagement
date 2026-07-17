#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_triggers.py — visualise ALL photodiode triggers and the 60->N trial loss.

Panels:
  A. every trigger on the recording timeline (S 15 primary edges, S 14 twins),
     with the behaviour-predicted gap-onset / av-onset times overlaid.
  B. per-trial capture raster: for each behaviour trial, whether its gap edge and
     its av edge were captured (a trial is usable only if BOTH are).
  C. inter-edge interval sequence (reveals the gap/block structure + the drops).
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from sda import io  # noqa: E402
from cma_eeg import loading, alignment  # noqa: E402
import mne  # noqa: E402
mne.set_log_level("ERROR")


def main():
    cfg = io.load_config(os.path.join(HERE, "config_stimdec.yaml"))
    vhdr = io.abspath(cfg["dataset"]["vhdr"])
    vmrk = os.path.splitext(vhdr)[0] + ".vmrk"
    raw = loading.load_raw(vhdr)
    sf = raw.info["sfreq"]

    # all markers by code
    text = open(vmrk, encoding="utf-8", errors="replace").read()
    mks = []
    for line in text.splitlines():
        m = re.match(r"Mk\d+=Stimulus,([^,]*),(\d+),", line)
        if m:
            mks.append((m.group(1).strip(), int(m.group(2))))
    s15 = np.array(sorted(s for c, s in mks if c == "S 15"), float)
    s14 = np.array(sorted(s for c, s in mks if c == "S 14"), float)
    s15_t, s14_t = (s15 - 1) / sf, (s14 - 1) / sf

    markers = loading.load_markers(vmrk, sf, edge_code=cfg["markers"]["edge_code"])
    ali = alignment.align(markers, io.abspath(cfg["dataset"]["behavior"]),
                          match_tol_s=cfg["alignment"]["match_tol_s"],
                          offset_search_s=cfg["alignment"]["offset_search_s"],
                          gap_lo_s=cfg["markers"]["gap_lo_s"],
                          gap_hi_s=cfg["markers"]["gap_hi_s"])
    offset, rec0 = ali.offset_s, markers.rec_start_unix
    tol = cfg["alignment"]["match_tol_s"]

    # behaviour-predicted edge times (EEG seconds)
    sess = json.load(open(io.abspath(cfg["dataset"]["behavior"])))
    rows = []
    for t in sess["trials"]:
        g, a = t.get("gap_onset_wall_clock"), t.get("webcam_av_onset_wallclock")
        rows.append((int(t["trial"]),
                     (float(g) + offset - rec0) if g else None,
                     (float(a) + offset - rec0) if a else None,
                     t.get("attended_modality")))

    def captured(pred):
        if pred is None or len(s15_t) == 0:
            return False, np.inf
        d = np.min(np.abs(s15_t - pred))
        return d <= tol, d

    usable = set(int(x) for x in ali.events["trial"])
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 2.2, 1.0], hspace=0.35)

    # ---- A: timeline -------------------------------------------------------
    axA = fig.add_subplot(gs[0])
    axA.vlines(s15_t, 0.6, 1.0, color="#1f77b4", lw=0.8, label=f"S 15 edges (n={len(s15_t)})")
    axA.vlines(s14_t, 0.6, 1.0, color="#ff7f0e", lw=0.4, alpha=0.6, label=f"S 14 twins (n={len(s14_t)})")
    gpred = [g for _, g, _, _ in rows if g is not None]
    apred = [a for _, _, a, _ in rows if a is not None]
    axA.plot(gpred, [0.4] * len(gpred), "v", color="#2ca02c", ms=5, label="behav gap-onset (60)")
    axA.plot(apred, [0.25] * len(apred), "^", color="#d62728", ms=5, label="behav av-onset (60)")
    axA.set_yticks([]); axA.set_xlabel("time in recording (s)")
    axA.set_title("A. All photodiode triggers vs behaviour-predicted onsets")
    axA.legend(ncol=4, fontsize=8, loc="upper right")

    # ---- B: per-trial capture raster --------------------------------------
    axB = fig.add_subplot(gs[1])
    for tr, g, a, mod in rows:
        gc, _ = captured(g); ac, _ = captured(a)
        y = tr
        col_g = "#2ca02c" if gc else "#cccccc"
        col_a = "#d62728" if ac else "#cccccc"
        axB.barh(y, 0.45, left=0.0, color=col_g, edgecolor="k", lw=0.2)
        axB.barh(y, 0.45, left=0.5, color=col_a, edgecolor="k", lw=0.2)
        if tr in usable:
            axB.plot(1.1, y, "o", color="black", ms=3)
    axB.set_yticks(range(0, 61, 5)); axB.invert_yaxis()
    axB.set_xticks([0.22, 0.72, 1.1]); axB.set_xticklabels(["gap edge", "av edge", "usable"])
    axB.set_ylabel("trial")
    axB.set_title(f"B. Per-trial edge capture — green/red = captured, grey = MISSED "
                  f"(usable = both). {len(usable)}/60 usable "
                  f"({int(ali.events['label'].eq('Audio').sum())}A/"
                  f"{int(ali.events['label'].eq('Visual').sum())}V)")

    # ---- C: inter-edge intervals ------------------------------------------
    axC = fig.add_subplot(gs[2])
    di = np.diff(s15_t)
    axC.plot(range(1, len(di) + 1), di, ".-", color="#333", ms=4)
    axC.axhspan(cfg["markers"]["gap_lo_s"], cfg["markers"]["gap_hi_s"],
                color="#2ca02c", alpha=0.15, label="expected gap (~2 s)")
    axC.set_yscale("log"); axC.set_xlabel("S 15 edge index"); axC.set_ylabel("interval to next edge (s)")
    axC.set_title("C. Inter-edge intervals (short = within-trial gap; long = block+probe+ITI)")
    axC.legend(fontsize=8)

    out = io.abspath(os.path.join(cfg["output"]["root"], cfg["dataset"]["subject"],
                                  "figures", "triggers.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)

    # text summary
    both = sum(1 for _, g, a, _ in rows if captured(g)[0] and captured(a)[0])
    only_g = sum(1 for _, g, a, _ in rows if captured(g)[0] and not captured(a)[0])
    only_a = sum(1 for _, g, a, _ in rows if captured(a)[0] and not captured(g)[0])
    neither = sum(1 for _, g, a, _ in rows if not captured(g)[0] and not captured(a)[0])
    print(f"triggers.png -> {out}")
    print(f"S15 edges: {len(s15_t)} | S14 twins: {len(s14_t)} | offset {offset:.4f}s")
    print(f"of 60 trials: both edges={both}  only gap={only_g}  only av={only_a}  neither={neither}")
    print(f"usable (align) = {len(usable)}")


if __name__ == "__main__":
    main()
