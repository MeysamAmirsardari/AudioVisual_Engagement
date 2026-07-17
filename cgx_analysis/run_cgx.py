#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cgx.py — full analysis of the CGX load-modulation recording (sub-Meysam ses-S002),
mirroring the av_integration diagnostics but with COGNITIVE LOAD (game difficulty) as
the contrast instead of Audio-vs-Visual.

Reuses the preprocessing + decoding helpers from av_integration/pipeline.py (STAR, ICA,
band-power, grouped CV, permutation, Haufe patterns) and adds:
  * behavioural manipulation checks (probe accuracy + RT vs difficulty),
  * a plot at every preprocessing step,
  * LOAD decoding (high vs low difficulty): band-power+LDA, Riemannian, CSP — with
    condition topographies, per-channel AUC, model comparison, ROC/confusion, Haufe,
  * the graded dose-response (frontal-theta / parietal-alpha vs the 6 difficulty tiers).

Prereq: run  python cgx_analysis/prep.py  once (caches raw.fif + trials.json).
Run:    python cgx_analysis/run_cgx.py  [--no-star --no-ica --n-perm N]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "av_integration"))
import pipeline as P  # noqa: E402  (reuse helpers; its main() is __main__-guarded)
import mne  # noqa: E402
from scipy.stats import spearmanr, sem  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix  # noqa: E402

mne.set_log_level("ERROR")
CACHE = os.path.join(HERE, "cache")
BANDS = P.BANDS
FRONTAL = ["Fz", "F3", "F4", "FC5", "FC6", "AF7", "AF8", "Fpz", "Fp1", "Fp2", "F7", "F8"]
PARIETAL = ["Pz", "P3", "P4", "P7", "P8", "PO7", "PO8", "Oz", "O1", "O2", "CP5", "CP6"]
PAD, WIN = 2.0, 25.0                                     # epoch: 2..27 s after audio onset


def run_ica_cgx(raw, figdir, n_comp=20):
    """CGX-robust ICA: a FIXED integer number of components (a variance threshold
    collapses to 1 component here — dry-electrode data has one dominant component),
    extended-infomax, using the real ExG EOG channels for ocular detection."""
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ncomp = int(min(n_comp, len(picks) - 1))
    ica = mne.preprocessing.ICA(n_components=ncomp, method="infomax",
                                fit_params=dict(extended=True), random_state=42, max_iter=500)
    ica.fit(raw.copy().filter(1, 45))
    drop = set()
    try:
        drop |= set(ica.find_bads_eog(raw)[0])              # uses ExG EOG channels
    except Exception:
        pass
    try:
        drop |= set(ica.find_bads_muscle(raw, threshold=0.9)[0])
    except Exception:
        pass
    drop = sorted(int(i) for i in drop)
    ica.exclude = drop
    P.STEP += 1
    try:
        cf = ica.plot_components(show=False)
        for i, f in enumerate(cf if isinstance(cf, list) else [cf]):
            f.suptitle(f"STEP {P.STEP}: ICA components (excluded {drop})", fontsize=10)
            f.savefig(os.path.join(figdir, f"step{P.STEP:02d}_ICA_components_{i}.png"),
                      dpi=130, bbox_inches="tight")
            plt.close(f)
    except Exception:
        pass
    before = raw.copy()
    ica.apply(raw)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    P._psd(before.copy().pick_types(eeg=True), ax, color="#c44", label="before ICA")
    P._psd(raw.copy().pick_types(eeg=True), ax, color="#282", label="after ICA")
    ax.legend(frameon=False); ax.set_title(f"ICA effect (removed {len(drop)}/{ncomp} ICs)")
    P._fig(figdir, f"step{P.STEP:02d}_ICA_effect.png", fig)
    return {"excluded": drop, "n_components": ncomp}


def behavioural_figs(meta, figdir):
    """Manipulation checks from the markers: probe accuracy + RT vs difficulty."""
    order = meta["difficulty_order"]
    tr = meta["trials"]
    acc, rt, beeprt = [], [], []
    for lv in order:
        d = [t for t in tr if t["difficulty"] == lv]
        cc = [bool(t["probe_correct"]) for t in d if t.get("probe_correct") is not None]
        acc.append(np.mean(cc) if cc else np.nan)
        rr = [t["probe_rt_s"] for t in d if t.get("probe_rt_s")]
        rt.append(np.mean(rr) if rr else np.nan)
        br = [t["beep_rt_s"] for t in d if t.get("beep_rt_s")]
        beeprt.append(np.mean(br) if br else np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(order))
    for ax, val, ttl, yl in [(axes[0], acc, "audio-probe accuracy", "P(correct)"),
                             (axes[1], rt, "audio-probe RT", "s"),
                             (axes[2], beeprt, "beep RT", "s")]:
        ax.plot(x, val, "o-", color="#b0651f")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(ttl); ax.set_ylabel(yl); ax.set_xlabel("difficulty (load)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhline(0.5, ls=":", color="grey")
    P._fig(figdir, "behaviour_by_difficulty.png", fig,
           "Behavioural manipulation check (does load affect audio comprehension / vigilance?)")
    return {"probe_acc_by_level": dict(zip(order, [None if np.isnan(a) else round(a, 3) for a in acc]))}


def roi_bandpower(bp, ch_names, roi, band):
    idx = [ch_names.index(c) for c in roi if c in ch_names]
    return bp[:, idx, list(BANDS).index(band)].mean(1)          # (n_trials,)


def graded_fig(bp, ch_names, load, order, figdir):
    """Dose-response: frontal theta + parietal alpha vs the 6 difficulty levels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    specs = [("frontal", FRONTAL, "theta", "#c0392b"),
             ("parietal", PARIETAL, "alpha", "#2471a3")]
    stats = {}
    for ax, (roiname, roi, band, col) in zip(axes, specs):
        v = roi_bandpower(bp, ch_names, roi, band)
        rho, p = spearmanr(load, v)
        m = [v[load == lv].mean() for lv in range(6)]
        e = [sem(v[load == lv]) for lv in range(6)]
        ax.errorbar(range(6), m, yerr=e, fmt="o-", color=col, capsize=3)
        ax.set_xticks(range(6)); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{roiname} {band} power vs load\nSpearman ρ={rho:+.2f} (p={p:.3f})")
        ax.set_ylabel("log band-power"); ax.set_xlabel("difficulty (load)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        stats[f"{roiname}_{band}"] = {"rho": float(rho), "p": float(p)}
    P._fig(figdir, "graded_load_doseresponse.png", fig,
           "Graded load effect (frontal theta ↑, parietal alpha ↓ with load?)")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--no-star", action="store_true")
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = os.path.join(HERE, "derivatives")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    meta = json.load(open(os.path.join(CACHE, "trials.json")))
    raw = mne.io.read_raw_fif(os.path.join(CACHE, "raw.fif"), preload=True)
    prov = {"subject": "Meysam_S002", "n_valid_trials": meta["n_valid"], "steps": []}
    print("=" * 74); print("CGX load-modulation analysis — sub-Meysam ses-S002"); print("=" * 74)

    # ---- behavioural checks -----------------------------------------------
    prov["behaviour"] = behavioural_figs(meta, figdir)

    # ---- STEP 0: montage + raw --------------------------------------------
    P.STEP = 0
    fig, ax = plt.subplots(figsize=(6.5, 6.5)); raw.copy().pick_types(eeg=True).plot_sensors(
        show_names=True, axes=ax, show=False)
    ax.set_title("CGX Quick-32r montage (29 scalp ch)"); P._fig(figdir, "step00_montage.png", fig)
    P.step_snapshot(raw.copy().pick_types(eeg=True), figdir, "raw", "CGX, 500 Hz (µV->V)")

    # ---- STEP 1: notch (50/60 Hz + harmonics) -----------------------------
    freqs = [f for f in (50, 60, 100, 120, 150, 180, 200, 240) if f < raw.info["sfreq"] / 2]
    before = raw.copy().pick_types(eeg=True)
    raw.notch_filter(freqs)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    P._psd(before, ax, color="#c44", label="before"); P._psd(raw.copy().pick_types(eeg=True), ax,
           color="#282", label="after notch")
    ax.legend(frameon=False); ax.set_title(f"notch {freqs}")
    P.STEP += 1; P._fig(figdir, f"step{P.STEP:02d}_notch.png", fig, f"STEP {P.STEP}: notch")

    # ---- STEP 2: band-pass + resample -------------------------------------
    raw.filter(1.0, 45.0, fir_design="firwin"); raw.resample(args.fs)
    P.step_snapshot(raw.copy().pick_types(eeg=True), figdir, "bandpass_resample",
                    f"1-45 Hz, {args.fs:.0f} Hz")

    # ---- STEP 3: robust bad channels (on avg-ref eeg copy) ----------------
    bads = P.robust_bads(raw.copy().pick_types(eeg=True))
    raw.info["bads"] = bads
    fig, ax = plt.subplots(figsize=(6, 5.5))
    eeg_names = raw.copy().pick_types(eeg=True).ch_names
    mask = np.array([c in bads for c in eeg_names], float)
    mne.viz.plot_topomap(mask, raw.copy().pick_types(eeg=True).info, axes=ax, show=False,
                         cmap="Reds", contours=0)
    ax.set_title(f"robust bad channels: {bads or 'none'}")
    P.STEP += 1; P._fig(figdir, f"step{P.STEP:02d}_bad_channels.png", fig, f"STEP {P.STEP}: bad channels")
    prov["bad_channels"] = bads; print(f"bad channels: {bads or 'none'}")

    # Average-reference BEFORE ICA: the CGX common-mode signal otherwise dominates
    # the PCA (one component ~99.97% of variance) and ICA collapses to 1 component.
    raw.set_eeg_reference("average", projection=False)

    # ---- STEP 4: ICA (ocular/muscle) --------------------------------------
    if not args.no_ica:
        prov["ica"] = run_ica_cgx(raw, figdir); print(f"ICA removed {prov['ica']['excluded']}")

    # ---- STEP 5: STAR (sparse artifacts, EEG only) ------------------------
    if not args.no_star:
        good = raw.copy().pick_types(eeg=True, exclude=bads)
        frac = P.run_star(good, figdir)
        raw._data[[raw.ch_names.index(c) for c in good.ch_names]] = good._data
        prov["star_fraction_repaired"] = frac; print(f"STAR repaired {frac*100:.1f}%")

    # ---- STEP 6: interpolate + average reference --------------------------
    if bads:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)
    P.step_snapshot(raw.copy().pick_types(eeg=True), figdir, "final_avgref",
                    f"interpolated {bads}, average ref")

    # ======================================================================
    # DECODING: cognitive load (high vs low difficulty)
    # ======================================================================
    fs = raw.info["sfreq"]
    eeg = raw.copy().pick_types(eeg=True)
    info, ch_names = eeg.info, eeg.ch_names
    D = eeg.get_data()
    Xe, load = [], []
    for d in meta["trials"]:
        s0 = int((d["onset_s"] + PAD) * fs)
        Xe.append(D[:, s0:s0 + int(WIN * fs)]); load.append(d["load"])
    X = np.stack(Xe); load = np.array(load)                 # (30, n_ch, win)
    y = (load >= 3).astype(int)                            # high load (hard+) vs low
    order = meta["difficulty_order"]
    print(f"\nDECODING load: {len(y)} trials, high {int(y.sum())} / low {int((y==0).sum())}, "
          f"{X.shape[1]} ch x {X.shape[2]} samp")

    bp = P.band_power_features(X, fs)                       # (tr, ch, band)
    bands = list(BANDS)
    # per-band condition topographies (high, low, difference)
    P.STEP = 7
    fig, axes = plt.subplots(3, len(bands), figsize=(4 * len(bands), 11))
    for j, bn in enumerate(bands):
        hi = bp[y == 1, :, j].mean(0); lo = bp[y == 0, :, j].mean(0)
        for i, (dat, ttl) in enumerate([(hi, "high load"), (lo, "low load"),
                                        (hi - lo, "high−low")]):
            cmap = "RdBu_r" if i == 2 else "viridis"
            vlim = ((-np.abs(dat).max(), np.abs(dat).max()) if i == 2
                    else tuple(np.percentile(dat, [5, 95])))
            mne.viz.plot_topomap(dat, info, axes=axes[i, j], show=False, cmap=cmap,
                                 vlim=vlim, contours=3)
            axes[i, j].set_title(f"{bn} {ttl}", fontsize=9)
    P._fig(figdir, "step07_bandpower_topographies.png", fig,
           "STEP 7: band-power by cognitive load")
    # per-channel discriminability
    auc = P.channel_band_auc(bp, y)
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))
    for j, bn in enumerate(bands):
        mne.viz.plot_topomap(auc[:, j] - 0.5, info, axes=axes[j], show=False, cmap="RdBu_r",
                             vlim=(-0.3, 0.3), contours=3); axes[j].set_title(f"{bn}\nAUC−0.5", fontsize=10)
    P._fig(figdir, "step07b_discriminability_AUC.png", fig,
           "STEP 7: per-channel high-vs-low-load discriminability")

    # graded dose-response
    prov["graded"] = graded_fig(bp, ch_names, load, order, figdir)

    # ---- models -----------------------------------------------------------
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from mne.decoding import CSP
    Xbp = bp.reshape(len(y), -1)
    models = {"bandpower+LDA": (Xbp, make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))),
              "Riemann TS+LR": (X, make_pipeline(Covariances("oas"), TangentSpace(),
                                LogisticRegression(C=1.0, max_iter=1000))),
              "CSP+LDA": (X, make_pipeline(CSP(6, reg="ledoit_wolf", log=True),
                          LDA(solver="lsqr", shrinkage="auto")))}
    results = {}
    for name, (Xf, est) in models.items():
        acc, auc_, p, null = P.perm_test(est, Xf, y, args.n_splits, args.seed, args.n_perm)
        results[name] = {"acc": acc, "auc": auc_, "perm_p": p, "null_p95": float(np.percentile(null, 95))}
        print(f"  {name:16s} acc {acc:.3f}  AUC {auc_:.3f}  perm-p {p:.4f}")
    prov["models"] = results

    fig, ax = plt.subplots(figsize=(7.5, 4.6)); names = list(results); xs = np.arange(len(names))
    ax.bar(xs, [results[n]["acc"] for n in names], 0.6, color=["#7b3fa0", "#2471a3", "#3b8f5a"])
    ax.hlines([results[n]["null_p95"] for n in names], xs - 0.3, xs + 0.3, ls="--", color="k",
              label="perm null p95")
    ax.axhline(0.5, ls=":", color="grey", label="chance")
    for xi, n in zip(xs, names):
        ax.text(xi, results[n]["acc"] + 0.02, f"{results[n]['acc']:.2f}\np={results[n]['perm_p']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 1.05)
    ax.set_ylabel("high-vs-low load accuracy (CV)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("STEP 8: load decoding — model comparison"); P._fig(figdir, "step08_model_comparison.png", fig)

    best = max(results, key=lambda n: results[n]["auc"])
    Xf, est = models[best]; proba, _, _ = P.grouped_cv(est, Xf, y, args.n_splits, args.seed)
    fpr, tpr, _ = roc_curve(y, proba); cm = confusion_matrix(y, (proba > 0.5).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, color="#2471a3"); axes[0].plot([0, 1], [0, 1], ls=":", color="grey")
    axes[0].set_title(f"ROC — {best} (AUC {results[best]['auc']:.2f})"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, str(v), ha="center", va="center")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["low", "high"]); axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["low", "high"]); axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    axes[1].set_title("confusion"); P._fig(figdir, "step08b_roc_confusion.png", fig)

    # Haufe patterns
    lda = models["bandpower+LDA"][1].fit(Xbp, y)
    w = lda.named_steps["lineardiscriminantanalysis"].coef_.ravel() / (
        lda.named_steps["standardscaler"].scale_ + 1e-12)
    A = P.haufe_pattern(Xbp, y, w).reshape(len(ch_names), len(bands))
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))
    for j, bn in enumerate(bands):
        mne.viz.plot_topomap(A[:, j], info, axes=axes[j], show=False, cmap="RdBu_r",
                             vlim=(-np.abs(A).max(), np.abs(A).max()), contours=3); axes[j].set_title(bn, fontsize=10)
    P._fig(figdir, "step08c_haufe_patterns.png", fig,
           "STEP 8: Haufe patterns of the load LDA (what the model uses)")
    try:
        csp = CSP(6, reg="ledoit_wolf", log=True).fit(X, y)
        fig = csp.plot_patterns(info, ch_type="eeg", show=False)
        fig.suptitle("STEP 8: CSP patterns (high vs low load)", fontsize=11)
        fig.savefig(os.path.join(figdir, "step08d_csp_patterns.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print("  (CSP patterns skipped:", e, ")")

    out = {"created": _dt.datetime.now().isoformat(timespec="seconds"),
           "pipeline": "cgx load-modulation", **prov, "best_model": best,
           "bands": {k: list(v) for k, v in BANDS.items()}, "n_permutations": args.n_perm}
    with open(os.path.join(outdir, "cgx_load_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    nf = len([f for f in os.listdir(figdir) if f.endswith(".png")])
    print("=" * 74); print(f"best load decoder: {best} (acc {results[best]['acc']:.3f}, p={results[best]['perm_p']:.4f})")
    print(f"saved {nf} figures + JSON -> {outdir}"); print("=" * 74)


if __name__ == "__main__":
    main()
