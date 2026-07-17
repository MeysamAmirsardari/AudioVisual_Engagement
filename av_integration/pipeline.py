#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
av_integration/pipeline.py — consolidated cross-modal (Audio-vs-Visual) attention
decoding, built end-to-end with a diagnostic PLOT AT EVERY STEP.

Pipeline
--------
  PRE  0. load + CORRECTED montage (cable-half swap; see channel_swap_validate.py)
       1. notch (line noise)                         -> PSD before/after
       2. band-pass + resample                       -> PSD
       3. robust bad-channel detection               -> flagged-channel topography
       4. STAR — sparse time-artifact removal        -> per-channel variance before/after
       5. ICA (ocular / muscle) removal              -> component topographies + overlay
       6. interpolate bads + recover reference + average reference -> final PSD + alpha topo
  DEC  7. epoch the audiovisual block (Audio vs Visual)
       8. FEATURES: band-power (theta/alpha/beta/gamma), spatial covariance, CSP
          -> per-band condition topographies + per-channel discriminability (AUC)
       9. MODELS: band-power+LDA, Riemannian tangent-space+LR, CSP+LDA
          -> trial-grouped CV accuracy + permutation null, ROC, confusion,
             Haufe activation patterns, CSP patterns

Every step writes a PNG to av_integration/derivatives/<subject>/figures/ and every
number to <subject>_av_integration.json. Run:
    python av_integration/pipeline.py            (--no-star / --no-ica / --n-perm N)
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
sys.path.insert(0, os.path.join(ROOT, "stim_decoding"))
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess as PP  # noqa: E402
from cma_eeg import loading  # noqa: E402
import mne  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.signal import welch  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix  # noqa: E402

mne.set_log_level("ERROR")

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
POS = "Audio"
STEP = 0                                              # running step counter for filenames


# ==========================================================================
# small plotting helpers
# ==========================================================================
def _fig(figdir, name, fig, title=None):
    global STEP
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    path = os.path.join(figdir, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _psd(raw, ax, fmax=60, color=None, label=None):
    p = raw.compute_psd(fmin=1, fmax=fmax)
    f, d = p.freqs, p.get_data()
    ax.plot(f, 10 * np.log10(d.mean(0)), color=color, label=label, lw=1.4)
    ax.fill_between(f, 10 * np.log10(np.percentile(d, 25, 0)),
                    10 * np.log10(np.percentile(d, 75, 0)), color=color, alpha=0.15)
    ax.set_xlabel("Hz"); ax.set_ylabel("power (dB)")


def step_snapshot(raw, figdir, label, note=""):
    """A consistent per-step diagnostic: PSD, broadband-variance topography, a 3 s
    trace of a few channels."""
    global STEP
    STEP += 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    _psd(raw, axes[0], color="#356", label="mean")
    axes[0].set_title("PSD (1-60 Hz)")
    try:
        var = np.log(np.var(raw.get_data(), axis=1) + 1e-30)
        mne.viz.plot_topomap(var, raw.info, axes=axes[1], show=False, cmap="magma",
                             contours=3)
        axes[1].set_title("log channel variance")
    except Exception as e:
        axes[1].text(0.5, 0.5, f"topo n/a\n{e}", ha="center"); axes[1].axis("off")
    picks = raw.ch_names[:6]
    seg = raw.copy().pick(picks).get_data(start=0, stop=int(3 * raw.info["sfreq"]))
    t = np.arange(seg.shape[1]) / raw.info["sfreq"]
    for i, ch in enumerate(picks):
        axes[2].plot(t, seg[i] * 1e6 + i * 60, lw=0.6)
    axes[2].set_yticks([i * 60 for i in range(len(picks))]); axes[2].set_yticklabels(picks)
    axes[2].set_xlabel("s"); axes[2].set_title("example traces (µV, offset)")
    return _fig(figdir, f"step{STEP:02d}_{label}.png", fig,
                f"STEP {STEP}: {label}   {note}")


# ==========================================================================
# preprocessing steps
# ==========================================================================
def robust_bads(raw, z=5.0):
    """Flag TRULY bad channels — specifically, low spatial coherence (a disconnected
    channel is uncorrelated with its neighbours) or a railing amplitude. We deliberately
    do NOT flag high variance alone, because frontal channels are legitimately high-
    variance from eye activity (handled by ICA), not broken.

    Detection runs on an AVERAGE-REFERENCED copy: the recording reference here is AF3
    (a frontal electrode, from the cable swap), which otherwise distorts the correlation
    structure of nearby frontal channels and makes them look falsely incoherent."""
    r = raw.copy().set_eeg_reference("average", projection=False, verbose="ERROR")
    X = r.get_data()
    C = np.corrcoef(X)
    np.fill_diagonal(C, np.nan)
    mc = np.nanmean(np.abs(C), 1)                         # mean |corr| with others
    lv = np.log(np.var(X, 1) + 1e-30)

    def zsc(v):
        med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-12
        return 0.6745 * (v - med) / mad

    bad = set()
    bad.update(raw.ch_names[i] for i in np.where(zsc(mc) < -z)[0])        # incoherent
    bad.update(raw.ch_names[i] for i in np.where(zsc(lv) > 6.0)[0])       # railing only
    return sorted(bad)


def run_star(raw, figdir, k=6):
    """meegkit STAR: sparse time-artifact removal, reconstructing outlier samples of a
    channel from its spatial neighbours. Plots per-channel variance before/after."""
    from meegkit import star
    pos = raw.get_montage().get_positions()["ch_pos"]
    names = raw.ch_names
    P = np.array([pos[n] for n in names])
    closest = cKDTree(P).query(P, k=k + 1)[1][:, 1:]     # (n_ch, k) neighbour indices
    X = raw.get_data().T                                 # (n_samples, n_ch)
    v0 = np.var(X, 0)
    y, w, _ = star.star(X, 3.0, closest=closest, verbose=False)
    v1 = np.var(y, 0)
    raw._data = y.T.astype(raw._data.dtype)
    frac = float(1 - (w.mean()))                         # fraction of samples flagged
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(np.log(v0 + 1e-30), label="before"); axes[0].plot(np.log(v1 + 1e-30),
                 label="after")
    axes[0].set_xlabel("channel"); axes[0].set_ylabel("log variance")
    axes[0].legend(frameon=False); axes[0].set_title("channel variance")
    mne.viz.plot_topomap(np.log(v0 + 1e-30), raw.info, axes=axes[1], show=False,
                         cmap="magma", contours=3); axes[1].set_title("variance before")
    mne.viz.plot_topomap(np.log(v1 + 1e-30), raw.info, axes=axes[2], show=False,
                         cmap="magma", contours=3); axes[2].set_title("variance after")
    global STEP; STEP += 1
    _fig(figdir, f"step{STEP:02d}_STAR.png", fig,
         f"STEP {STEP}: STAR sparse artifact removal — {frac*100:.1f}% of samples repaired")
    return frac


def run_ica(raw, figdir, frontal=("Fp1", "Fp2")):
    """Extended-infomax ICA; drop ocular (frontal-correlated) + muscle components.
    Plots component topographies with the removed ones marked, and a before/after PSD."""
    ica = mne.preprocessing.ICA(n_components=0.99, method="infomax",
                                fit_params=dict(extended=True), random_state=42, max_iter=500)
    ica.fit(raw.copy().filter(1, 45))
    bad_eog = []
    for ch in frontal:
        if ch in raw.ch_names:
            try:
                b, _ = ica.find_bads_eog(raw, ch_name=ch)
                bad_eog += b
            except Exception:
                pass
    try:
        bad_mus, _ = ica.find_bads_muscle(raw, threshold=0.8)
    except Exception:
        bad_mus = []
    drop = sorted(int(i) for i in (set(bad_eog) | set(bad_mus)))
    ica.exclude = drop
    try:
        figc = ica.plot_components(show=False)
        figs = figc if isinstance(figc, list) else [figc]
        global STEP; STEP += 1
        for i, f in enumerate(figs):
            f.suptitle(f"STEP {STEP}: ICA components (excluded {drop})", fontsize=11)
            f.savefig(os.path.join(figdir, f"step{STEP:02d}_ICA_components_{i}.png"),
                      dpi=130, bbox_inches="tight")
            plt.close(f)
    except Exception:
        pass
    before = raw.copy()
    ica.apply(raw)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _psd(before, ax, color="#c44", label="before ICA")
    _psd(raw, ax, color="#282", label="after ICA")
    ax.legend(frameon=False); ax.set_title(f"ICA effect on PSD (removed {len(drop)} ICs)")
    _fig(figdir, f"step{STEP:02d}_ICA_effect.png", fig)
    return {"excluded": drop, "n_eog": len(set(bad_eog)), "n_muscle": len(set(bad_mus))}


# ==========================================================================
# features
# ==========================================================================
def band_power_features(X, sfreq):
    """(n_trials, n_ch, n_bands) log band-power via Welch."""
    nper = min(X.shape[-1], int(sfreq * 2))
    f, P = welch(X, fs=sfreq, nperseg=nper, axis=-1)         # (tr, ch, freq)
    out = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        out.append(np.log(P[..., m].mean(-1) + 1e-30))
    return np.stack(out, -1)                                 # (tr, ch, band)


def channel_band_auc(bp, y):
    """AUC of each (channel, band) feature for Audio vs Visual -> (n_ch, n_bands)."""
    auc = np.zeros(bp.shape[1:])
    for c in range(bp.shape[1]):
        for b in range(bp.shape[2]):
            auc[c, b] = roc_auc_score(y, bp[:, c, b])
    return auc


# ==========================================================================
# decoding
# ==========================================================================
def grouped_cv(estimator, Xf, y, n_splits, seed):
    """Stratified k-fold trial CV; returns per-trial predicted P(Audio) and acc/auc."""
    proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xf, y):
        est = estimator
        est.fit(Xf[tr], y[tr])
        proba[te] = est.predict_proba(Xf[te])[:, 1]
    acc = float(np.mean((proba > 0.5).astype(int) == y))
    return proba, acc, float(roc_auc_score(y, proba))


def perm_test(estimator, Xf, y, n_splits, seed, n_perm):
    _, acc, auc = grouped_cv(estimator, Xf, y, n_splits, seed)
    rng = np.random.RandomState(seed)
    null = np.array([grouped_cv(estimator, Xf, rng.permutation(y), n_splits, seed)[1]
                     for _ in range(n_perm)])
    p = float((1 + np.sum(null >= acc)) / (1 + n_perm))
    return acc, auc, p, null


def haufe_pattern(Xf, y, w):
    """Haufe activation pattern A = Cov(X) w / (w' Cov(X) w) for a linear filter w."""
    Xc = Xf - Xf.mean(0)
    cov = Xc.T @ Xc / (len(Xf) - 1)
    a = cov @ w
    return a / (np.linalg.norm(a) + 1e-12)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "stim_decoding", "config_stimdec.yaml"))
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--no-star", action="store_true")
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global STEP
    cfg = io.load_config(args.config)
    subject = cfg["dataset"]["subject"]
    outdir = os.path.join(HERE, "derivatives", subject)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    prov = {"subject": subject, "steps": []}
    print("=" * 74)
    print(f"AV-integration consolidated pipeline — {subject}")
    print("=" * 74)

    # --- STEP 0: load + corrected montage ----------------------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    orig_sf = raw.info["sfreq"]
    montage, num2name = PP._bvef(cfg)                        # cable_halves_swapped from cfg
    raw.rename_channels({ch: num2name[ch] for ch in raw.ch_names if ch in num2name})
    raw.set_montage(montage, on_missing="ignore")
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    swapped = bool(cfg["preprocess"].get("cable_halves_swapped"))
    prov["montage"] = {"cable_halves_swapped": swapped, "n_ch": len(raw.ch_names)}
    print(f"STEP 0: montage set (cable_halves_swapped={swapped}), {len(raw.ch_names)} ch")
    # sensor layout
    fig, ax = plt.subplots(figsize=(6.5, 6.5)); raw.plot_sensors(show_names=True, axes=ax, show=False)
    ax.set_title(f"corrected montage (swapped={swapped})"); _fig(figdir, "step00_montage.png", fig)
    STEP = 0
    step_snapshot(raw, figdir, "raw", "as recorded (ref = AF3 under the swap)")

    # --- STEP 1: notch -----------------------------------------------------
    before = raw.copy()
    raw.notch_filter(np.arange(50, orig_sf / 2, 50))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _psd(before, ax, color="#c44", label="before"); _psd(raw, ax, color="#282", label="after notch")
    ax.legend(frameon=False); ax.set_title("notch 50 Hz + harmonics")
    STEP += 1; _fig(figdir, f"step{STEP:02d}_notch.png", fig, f"STEP {STEP}: notch")

    # --- STEP 2: band-pass + resample --------------------------------------
    raw.filter(1.0, 45.0, fir_design="firwin")
    raw.resample(args.fs)
    step_snapshot(raw, figdir, "bandpass_resample", f"1-45 Hz, {args.fs:.0f} Hz")

    # --- STEP 3: robust bad channels ---------------------------------------
    bads = robust_bads(raw)
    raw.info["bads"] = bads
    fig, ax = plt.subplots(figsize=(6, 5.5))
    mask = np.array([c in bads for c in raw.ch_names])
    mne.viz.plot_topomap(mask.astype(float), raw.info, axes=ax, show=False, cmap="Reds",
                         contours=0)
    ax.set_title(f"robust bad channels: {bads or 'none'}")
    STEP += 1; _fig(figdir, f"step{STEP:02d}_bad_channels.png", fig, f"STEP {STEP}: bad channels")
    prov["bad_channels"] = bads
    print(f"STEP {STEP}: bad channels = {bads or 'none'}")

    # --- STEP 4: ICA (remove ocular/muscle FIRST) --------------------------
    # ICA before STAR: eye blinks/muscle are physiological, best handled by ICA. If
    # STAR ran first it would waste effort trying to "repair" every blink transient
    # (slow, and not what sparse-artifact removal is for).
    if not args.no_ica:
        ica_info = run_ica(raw, figdir)
        prov["ica"] = ica_info
        print(f"STEP {STEP}: ICA removed {ica_info['excluded']}", flush=True)

    # --- STEP 5: STAR (sparse sensor artifacts on the now-clean data) -------
    if not args.no_star:
        good = raw.copy().pick([c for c in raw.ch_names if c not in bads])
        frac = run_star(good, figdir)
        raw._data[[raw.ch_names.index(c) for c in good.ch_names]] = good._data
        prov["star_fraction_repaired"] = frac
        print(f"STEP {STEP}: STAR repaired {frac*100:.1f}% of samples", flush=True)

    # --- STEP 6: interpolate + recover reference + average reference -------
    if bads:
        raw.interpolate_bads(reset_bads=True)
    ref_num = str(cfg["preprocess"].get("reference_electrode", "")).strip()
    ref_name = num2name.get(ref_num)
    if ref_name and ref_name not in raw.ch_names:
        raw.add_reference_channels([ref_name]); raw.set_montage(montage, on_missing="ignore")
        print(f"STEP 6: recovered reference channel '{ref_name}' (AF3 under the swap)")
    raw.set_eeg_reference("average", projection=False)
    step_snapshot(raw, figdir, "final_avgref", f"interpolated {bads}, ref={ref_name}, average")
    # alpha topography (should now be spatially smooth)
    psd = raw.compute_psd(fmin=1, fmax=45); f = psd.freqs; d = psd.get_data()
    adb = 10 * np.log10(d[:, (f >= 8) & (f <= 13)].mean(1))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    lo, hi = np.percentile(adb, [5, 95])
    im, _ = mne.viz.plot_topomap(adb, raw.info, axes=ax, show=False, cmap="viridis",
                                 vlim=(lo, hi), contours=4)
    fig.colorbar(im, ax=ax, fraction=0.046, label="dB"); ax.set_title("final alpha (8-13 Hz)")
    _fig(figdir, "step06b_alpha_topo.png", fig)

    # ======================================================================
    # DECODING
    # ======================================================================
    prov["orig_sfreq"] = orig_sf
    trials, info, ch_names = PP.extract_trials(raw, events, cfg, prov)
    order = sorted(trials)
    X = np.stack([trials[t]["eeg"] for t in order])          # (n_trials, n_ch, n_times)
    y = np.array([1 if trials[t]["label"] == POS else 0 for t in order])
    fs = raw.info["sfreq"]
    print(f"\nDECODING: {len(y)} trials (Audio {int(y.sum())} / Visual {int((y==0).sum())}), "
          f"{X.shape[1]} ch x {X.shape[2]} samples")

    # ---- STEP 7: features -------------------------------------------------
    bp = band_power_features(X, fs)                          # (tr, ch, band)
    bands = list(BANDS)
    # per-band condition topographies + difference
    fig, axes = plt.subplots(3, len(bands), figsize=(4 * len(bands), 11))
    for j, bn in enumerate(bands):
        a = bp[y == 1, :, j].mean(0); v = bp[y == 0, :, j].mean(0)
        for i, (dat, ttl) in enumerate([(a, "Audio"), (v, "Visual"), (a - v, "Audio−Visual")]):
            cmap = "RdBu_r" if i == 2 else "viridis"
            lo, hi = np.percentile(dat, [5, 95]) if i < 2 else (-np.abs(dat).max(), np.abs(dat).max())
            mne.viz.plot_topomap(dat, info, axes=axes[i, j], show=False, cmap=cmap,
                                 vlim=(lo, hi), contours=3)
            axes[i, j].set_title(f"{bn} {ttl}", fontsize=9)
    STEP = 7
    _fig(figdir, "step07_bandpower_topographies.png", fig,
         "STEP 7: band-power features by attended stream")
    # per-channel-band discriminability (AUC)
    auc = channel_band_auc(bp, y)
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))
    for j, bn in enumerate(bands):
        mne.viz.plot_topomap(auc[:, j] - 0.5, info, axes=axes[j], show=False, cmap="RdBu_r",
                             vlim=(-0.3, 0.3), contours=3)
        axes[j].set_title(f"{bn}\nAUC−0.5", fontsize=10)
    _fig(figdir, "step07b_discriminability_AUC.png", fig,
         "STEP 7: per-channel Audio-vs-Visual discriminability (AUC of band-power)")

    # ---- STEP 8: models ---------------------------------------------------
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from mne.decoding import CSP
    Xbp = bp.reshape(len(y), -1)                             # (tr, ch*band)
    models = {
        "bandpower+LDA": (Xbp, make_pipeline(StandardScaler(),
                          LDA(solver="lsqr", shrinkage="auto"))),
        "Riemann TS+LR": (X, make_pipeline(Covariances("oas"), TangentSpace(),
                          LogisticRegression(C=1.0, max_iter=1000))),
        "CSP+LDA": (X, make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True),
                    LDA(solver="lsqr", shrinkage="auto"))),
    }
    results, nulls = {}, {}
    for name, (Xf, est) in models.items():
        acc, auc_, p, null = perm_test(est, Xf, y, args.n_splits, args.seed, args.n_perm)
        results[name] = {"acc": acc, "auc": auc_, "perm_p": p,
                         "null_p95": float(np.percentile(null, 95))}
        nulls[name] = null
        print(f"  {name:16s} acc {acc:.3f}  AUC {auc_:.3f}  perm-p {p:.4f}")
    prov["models"] = results

    # model comparison bar
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    names = list(results); xs = np.arange(len(names))
    ax.bar(xs, [results[n]["acc"] for n in names], 0.6,
           color=["#7b3fa0", "#2471a3", "#3b8f5a"])
    ax.hlines([results[n]["null_p95"] for n in names], xs - 0.3, xs + 0.3, ls="--",
              color="k", label="perm null p95")
    ax.axhline(0.5, ls=":", color="grey", label="chance")
    for xi, n in zip(xs, names):
        ax.text(xi, results[n]["acc"] + 0.02, f"{results[n]['acc']:.2f}\np={results[n]['perm_p']:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 1.05)
    ax.set_ylabel("attended-modality accuracy (CV)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("STEP 8: model comparison (Audio vs Visual)")
    _fig(figdir, "step08_model_comparison.png", fig)

    # ROC + confusion for the best model
    best = max(results, key=lambda n: results[n]["auc"])
    Xf, est = models[best]
    proba, _, _ = grouped_cv(est, Xf, y, args.n_splits, args.seed)
    fpr, tpr, _ = roc_curve(y, proba); cm = confusion_matrix(y, (proba > 0.5).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, color="#2471a3"); axes[0].plot([0, 1], [0, 1], ls=":", color="grey")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC — {best} (AUC {results[best]['auc']:.2f})")
    im = axes[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, str(v), ha="center", va="center")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Visual", "Audio"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Visual", "Audio"])
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true"); axes[1].set_title("confusion")
    _fig(figdir, "step08b_roc_confusion.png", fig)

    # Haufe patterns for the band-power LDA (interpretable topographies)
    lda_pipe = models["bandpower+LDA"][1].fit(Xbp, y)
    w = lda_pipe.named_steps["lineardiscriminantanalysis"].coef_.ravel()
    w = w / (lda_pipe.named_steps["standardscaler"].scale_ + 1e-12)   # back to feature space
    A = haufe_pattern(Xbp, y, w).reshape(len(ch_names), len(bands))
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))
    for j, bn in enumerate(bands):
        mne.viz.plot_topomap(A[:, j], info, axes=axes[j], show=False, cmap="RdBu_r",
                             vlim=(-np.abs(A).max(), np.abs(A).max()), contours=3)
        axes[j].set_title(bn, fontsize=10)
    _fig(figdir, "step08c_haufe_patterns.png", fig,
         "STEP 8: Haufe activation patterns of the band-power LDA (what the model uses)")

    # CSP patterns
    try:
        csp = CSP(n_components=6, reg="ledoit_wolf", log=True).fit(X, y)
        fig = csp.plot_patterns(info, ch_type="eeg", show=False)
        fig.suptitle("STEP 8: CSP spatial patterns (Audio vs Visual)", fontsize=11)
        fig.savefig(os.path.join(figdir, "step08d_csp_patterns.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("  (CSP pattern plot skipped:", e, ")")

    # --- save --------------------------------------------------------------
    out = {"created": _dt.datetime.now().isoformat(timespec="seconds"),
           "pipeline": "av_integration consolidated", **prov,
           "n_trials": int(len(y)), "n_audio": int(y.sum()),
           "bands": {k: list(v) for k, v in BANDS.items()},
           "best_model": best, "n_permutations": args.n_perm}
    def _jd(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(os.path.join(outdir, f"{subject}_av_integration.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=_jd)
    n_figs = len([f for f in os.listdir(figdir) if f.endswith(".png")])
    print("=" * 74)
    print(f"best: {best} (acc {results[best]['acc']:.3f}, p={results[best]['perm_p']:.4f})")
    print(f"saved {n_figs} step figures + JSON -> {outdir}")
    print("=" * 74)


if __name__ == "__main__":
    main()
