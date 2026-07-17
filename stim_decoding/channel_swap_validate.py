#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
channel_swap_validate.py — decide EMPIRICALLY whether the two 32-channel cable
halves were swapped on this recording, and show it.

Background: channels are recorded as bare physical numbers; the .bvef gives the cap
geometry (electrode number -> 10-20 position). We assumed the two coloured 32-channel
cables were 1:32 / 33:64, but the first recordings may have had them reversed. This
compares the ASSUMED mapping against the HALF-SWAPPED one (recorded channel n placed
at electrode n±32) with two data-driven, name-independent tests:

  * SPATIAL SMOOTHNESS (decisive). EEG is volume-conducted, so neighbouring electrodes
    are strongly correlated. The CORRECT montage maximises the mean correlation between
    each channel and its nearest spatial neighbours; a wrong channel order scrambles
    positions and lowers it. (A random montage -> ~0.)
  * ALPHA POSTERIORITY (corroborating). Resting alpha (8-13 Hz) is posterior-dominant,
    so posterior-named channels should carry the most alpha under the correct mapping.

Outputs a side-by-side figure (montage labels + alpha topography for each mapping) and
prints the winner. Run:  python stim_decoding/channel_swap_validate.py
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
from sda import io, preprocess  # noqa: E402

_EEG = os.path.join(io.ROOT, "eeg_analysis")
if _EEG not in sys.path:
    sys.path.insert(0, _EEG)
from cma_eeg import loading  # noqa: E402
import mne  # noqa: E402

mne.set_log_level("ERROR")

POST = ["O1", "Oz", "O2", "POz", "PO3", "PO4", "PO7", "PO8", "P1", "P2", "Pz", "P3", "P4"]
FRONT = ["Fp1", "Fp2", "Fz", "F1", "F2", "F3", "F4", "AFz", "AF3", "AF4", "AF7", "AF8"]


def _prep(raw, num2name, montage):
    """Rename -> montage -> EEG picks -> average reference, on a copy."""
    r = raw.copy()
    r.rename_channels({ch: num2name[ch] for ch in r.ch_names if ch in num2name})
    r.set_montage(montage, on_missing="ignore")
    r.pick("eeg").set_eeg_reference("average", projection=False)
    return r


def neighbor_corr(raw, k=4):
    """Mean correlation of each channel with its k nearest spatial neighbours
    (broadband, 1-40 Hz). Higher = spatially smoother = montage more likely correct."""
    from scipy.spatial import cKDTree
    r = raw.copy().filter(1, 40)
    pos = r.get_montage().get_positions()["ch_pos"]
    names = [n for n in r.ch_names if n in pos and np.isfinite(pos[n]).all()]
    r.pick(names)
    P = np.array([pos[n] for n in names])
    X = r.get_data()
    C = np.corrcoef(X)
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k + 1)                      # col 0 is self
    return float(np.mean([C[i, idx[i, 1:]].mean() for i in range(len(names))]))


def alpha_scores(raw):
    """(alpha_db per channel, posterior-minus-frontal alpha in dB, mean posterior rank)."""
    psd = raw.compute_psd(fmin=1, fmax=40)
    f, d = psd.freqs, psd.get_data()
    alpha = d[:, (f >= 8) & (f <= 13)].mean(1)
    adb = 10 * np.log10(alpha)
    names = raw.ch_names
    pidx = [names.index(c) for c in POST if c in names]
    fidx = [names.index(c) for c in FRONT if c in names]
    order = list(np.argsort(alpha)[::-1])
    post_rank = float(np.mean([order.index(i) for i in pidx]))
    return adb, float(adb[pidx].mean() - adb[fidx].mean()), post_rank


def main():
    cfg = io.load_config(os.path.join(HERE, "config_stimdec.yaml"))
    raw0 = loading.load_raw(io.abspath(cfg["dataset"]["vhdr"]))
    montage, num2name = preprocess._bvef(cfg)             # base (swap off in this call)
    if montage is None:
        raise SystemExit("No .bvef montage available.")
    # ensure we have the UNSWAPPED base regardless of config, then derive the swap
    base = {}
    import xml.etree.ElementTree as ET
    p = io.abspath(cfg["preprocess"]["montage_bvef"])
    for e in ET.parse(p).getroot().findall("Electrode"):
        n = e.findtext("Number")
        if n is not None:
            base[str(int(n))] = e.findtext("Name")
    maps = {"assumed (1:32 / 33:64)": base,
            "SWAPPED (1:32 <-> 33:64)": preprocess._swap_cable_halves(base)}

    results = {}
    for label, n2n in maps.items():
        r = _prep(raw0, n2n, montage)
        sm = neighbor_corr(r)
        adb, pmf, prank = alpha_scores(r)
        results[label] = {"raw": r, "smooth": sm, "alpha_db": adb,
                          "post_minus_front_db": pmf, "post_rank": prank}
        print(f"{label:26s}: neighbour-corr {sm:.3f} | "
              f"posterior-frontal alpha {pmf:+.2f} dB | post rank {prank:.1f}")

    winner = max(results, key=lambda k: results[k]["smooth"])
    print(f"\nVERDICT (spatial smoothness): '{winner}' is the correct montage "
          f"(higher neighbour correlation = volume-conducted EEG placed sensibly).")

    # ---- figure: montage labels + alpha topo for each mapping ----
    outdir = io.abspath(os.path.join("records", "derivatives", "channel_swap",
                                     cfg["dataset"]["subject"]))
    os.makedirs(outdir, exist_ok=True)
    labels = list(maps)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    for j, lab in enumerate(labels):
        r = results[lab]["raw"]
        r.plot_sensors(show_names=True, axes=axes[0, j], show=False)
        star = "  ✓ WINNER" if lab == winner else ""
        axes[0, j].set_title(f"{lab}{star}\nneighbour-corr = {results[lab]['smooth']:.3f}",
                             fontsize=11)
        adb = results[lab]["alpha_db"]
        lo, hi = np.percentile(adb, [5, 95])
        im, _ = mne.viz.plot_topomap(adb, r.info, axes=axes[1, j], show=False,
                                     cmap="viridis", contours=4, vlim=(lo, hi))
        axes[1, j].set_title(f"alpha (8-13 Hz) power — posterior−frontal "
                             f"{results[lab]['post_minus_front_db']:+.1f} dB\n"
                             "(should peak occipito-parietal)", fontsize=10)
        fig.colorbar(im, ax=axes[1, j], fraction=0.046, label="dB")
    fig.suptitle(f"Cable-half swap check — {cfg['dataset']['subject']} "
                 f"(winner by spatial smoothness: {winner})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, "channel_swap_check.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    main()
