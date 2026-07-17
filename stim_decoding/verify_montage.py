#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_montage.py — plot the assumed channel map and check it EMPIRICALLY.

Channels are recorded as bare physical numbers; positions come from an ASSUMED
actiCAP-64 map (montage_acticap64.json), which the decoding does NOT depend on
but every topography DOES. This produces:
  1. channel_map.png        — the 2D sensor layout with the assumed 10-20 labels.
  2. montage_check_alpha.png — resting ALPHA (8-13 Hz) power topography. Alpha is
     posterior-dominant in essentially everyone, so if the assumed montage is
     right the alpha topography must peak occipito-parietally. If it does not, the
     montage (and hence the spatial interpretation of the TRF/CCA maps) is suspect.

    python stim_decoding/verify_montage.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from sda import io  # noqa: E402

_EEG = os.path.join(io.ROOT, "eeg_analysis")
if _EEG not in sys.path:
    sys.path.insert(0, _EEG)
from cma_eeg import loading, preprocessing  # noqa: E402
import mne  # noqa: E402

mne.set_log_level("ERROR")

POST = ["O1", "Oz", "O2", "POz", "PO3", "PO4", "PO7", "PO8", "P3", "Pz", "P4"]
FRONT = ["Fp1", "Fp2", "Fz", "F3", "F4", "AFz", "AF3", "AF4"]


def main():
    from sda import preprocess
    cfg = io.load_config(os.path.join(HERE, "config_stimdec.yaml"))
    raw = loading.load_raw(io.abspath(cfg["dataset"]["vhdr"]))

    montage, num2name = preprocess._bvef(cfg)
    if montage is not None:
        raw.rename_channels({ch: num2name[ch] for ch in raw.ch_names if ch in num2name})
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
        src = "CACS-64.bvef (ground truth)"
    else:
        mm = cfg["preprocess"]["montage_map"]
        mm = mm if os.path.isabs(mm) else os.path.normpath(os.path.join(HERE, mm))
        preprocessing.apply_montage(raw, mm, cfg["preprocess"]["montage_name"])
        num2name = {}
        src = "assumed template"
    raw.pick("eeg").filter(1, 40, verbose="ERROR")

    outdir = io.abspath(os.path.join(cfg["output"]["root"],
                                     cfg["dataset"]["subject"], "figures"))
    os.makedirs(outdir, exist_ok=True)

    # 1) sensor layout — 10-20 labels AND the physical electrode numbers
    name2num = {v: k for k, v in num2name.items()}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    raw.plot_sensors(show_names=True, axes=axes[0], show=False)
    axes[0].set_title("10-20 labels")
    if name2num:
        raw_num = raw.copy().rename_channels(
            {c: name2num.get(c, c) for c in raw.ch_names})
        raw_num.plot_sensors(show_names=True, axes=axes[1], show=False)
        axes[1].set_title("physical electrode numbers")
    else:
        axes[1].axis("off")
    fig.suptitle(f"Sensor layout — source: {src}")
    fig.savefig(os.path.join(outdir, "channel_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) alpha power topography (empirical montage check)
    psd = raw.compute_psd(fmin=1, fmax=40, verbose="ERROR")
    freqs = psd.freqs
    data = psd.get_data()                               # (n_ch, n_freq)
    alpha = data[:, (freqs >= 8) & (freqs <= 13)].mean(1)
    alpha_db = 10 * np.log10(alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # robust colour range: one railing channel otherwise dominates the scale
    lo, hi = np.percentile(alpha_db, [5, 95])
    im, _ = mne.viz.plot_topomap(alpha_db, raw.info, axes=axes[0], show=False,
                                 cmap="viridis", contours=4, vlim=(lo, hi))
    axes[0].set_title("Resting alpha (8-13 Hz) power\n(must be POSTERIOR-max if "
                      "the montage is correct)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="power (dB, robust scale)")

    pidx = [raw.ch_names.index(c) for c in POST if c in raw.ch_names]
    fidx = [raw.ch_names.index(c) for c in FRONT if c in raw.ch_names]
    axes[1].plot(freqs, 10 * np.log10(data[pidx].mean(0)),
                 color="#c44e52", label=f"posterior ({len(pidx)} ch)")
    axes[1].plot(freqs, 10 * np.log10(data[fidx].mean(0)),
                 color="#4c72b0", label=f"frontal ({len(fidx)} ch)")
    axes[1].axvspan(8, 13, color="gray", alpha=0.15)
    axes[1].set_xlabel("frequency (Hz)"); axes[1].set_ylabel("power (dB)")
    axes[1].set_title("PSD: posterior vs frontal"); axes[1].legend(frameon=False)
    fig.savefig(os.path.join(outdir, "montage_check_alpha.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(alpha)[::-1]
    top = [raw.ch_names[i] for i in order[:8]]
    post_rank = np.mean([list(order).index(raw.ch_names.index(c))
                         for c in POST if c in raw.ch_names])
    print("channel_map.png + montage_check_alpha.png ->", outdir)
    print("top-8 alpha channels (expect occipito-parietal):", top)
    print(f"mean rank of posterior channels among alpha power: {post_rank:.1f} "
          f"of {len(raw.ch_names)} (lower = more posterior-dominant = montage OK)")


if __name__ == "__main__":
    main()
