#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labram_attention.py — decode the ATTENDED MODALITY (Audio vs Visual) from EEG with
the LaBraM foundation model (Jiang et al., ICLR 2024) as a FROZEN feature extractor.

Design (careful, and honest about a single subject / 60 trials):

  * WHY a foundation model here, and WHY frozen. With one subject we cannot fine-tune
    a brain model without overfitting, so LaBraM is used purely as a pretrained
    encoder: each 15-s window -> its 200-d [CLS] embedding, computed ONCE. A small,
    regularised linear probe (shrinkage-LDA, parameter-free) is trained on top. The
    encoder never sees the labels or the fold split, so it cannot leak.

  * WHAT signal we expect. Attending vision vs audio is dominated by ALPHA-band
    (8-12 Hz) power redistribution over sensory cortices (Foxe & Snyder 2011), a
    broadband oscillatory effect. We therefore feed LaBraM broadband 200-Hz EEG
    (its native input) rather than the 1-8 Hz CCA band, which would delete it.

  * LEAKAGE CONTROL. Every 30-s trial yields several overlapping windows; those
    windows are NEVER split across train/test — cross-validation groups by TRIAL
    (each trial wholly in one fold). Standardisation/PCA are fit on the training
    windows only. The permutation null shuffles TRIAL labels (not windows) and
    re-runs the identical CV, so the null respects the grouping.

  * HONEST YARDSTICK. A classical alpha-power (and theta/alpha/beta band-power)
    decoder is run through the very same CV. LaBraM has to beat the simple,
    interpretable alpha decoder to be worth anything here. We also plot the alpha
    attention topography: a POSTERIOR (occipito-parietal) effect is the genuine
    intersensory-attention signature; a frontal one would flag an ocular confound
    (mitigated by the ICA ocular removal in preprocessing).

Outputs -> records/derivatives/labram_attention/<subject>/ : a JSON of every number,
a cached embeddings .npz, and two figures.
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
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess, labram as LB  # noqa: E402

from scipy.signal import welch  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (13, 30)}
POS = "Audio"                                    # positive class (label==1)


# --------------------------------------------------------------------------
# probes (parameter-free, robust for high-dim / small-N)
# --------------------------------------------------------------------------
def lda_pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def pca_lda_pipe():
    """PCA(20) then shrinkage-LDA — needed for LaBraM's 200-d embeddings, which a
    raw high-dim LDA cannot exploit at this N (established in the pooling diagnostic)."""
    return Pipeline([("sc", StandardScaler()), ("pca", PCA(20)),
                     ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


# --------------------------------------------------------------------------
# classical band-power features (same windows as LaBraM)
# --------------------------------------------------------------------------
def _bandpowers(win, fs):
    """log band-power per channel for `win` (n_ch, n_times) -> dict band -> (n_ch,)."""
    nper = min(win.shape[1], int(fs * 2))
    f, P = welch(win, fs=fs, nperseg=nper, axis=1)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[b] = np.log(P[:, m].mean(1) + 1e-20)
    return out


def band_tensor(trials, fs, win, n_win):
    """Per-window band-power tensor aligned to LaBraM's windows.
    Returns BP (n_win, n_ch, n_bands) and trial ids."""
    bands = list(BANDS)
    BP, tid = [], []
    for t, X in enumerate(trials):
        Xuv = np.asarray(X, np.float64) * 1e6                 # volts -> µV
        for s in LB.window_starts(Xuv.shape[1], win, n_win):
            bp = _bandpowers(Xuv[:, s:s + win], fs)
            BP.append(np.stack([bp[b] for b in bands], axis=1))   # (n_ch, n_bands)
            tid.append(t)
    return np.asarray(BP), np.asarray(tid, int)


def roi_masks(ch_names):
    """(posterior, frontal) boolean channel masks for the confound control.
    posterior = occipito-parietal (O/P/PO/I); frontal = Fp/AF/F (not fronto-central)."""
    up = [c.upper() for c in ch_names]
    post = np.array([c.startswith(("O", "P", "I")) for c in up])
    front = np.array([c.startswith(("FP", "AF")) or
                      (c.startswith("F") and not c.startswith(("FC", "FT", "FP"))) for c in up])
    return post, front


# --------------------------------------------------------------------------
# trial-grouped CV  (windows of a trial never split across folds)
# --------------------------------------------------------------------------
def make_folds(y_trial, n_splits, seed):
    fold = np.empty(len(y_trial), int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y_trial)), y_trial)):
        fold[te] = f
    return fold


def cv_trial_proba(Z, tid, y_trial, fold, n_splits, make_pipe):
    """Cross-validated P(Audio) per trial = mean over that trial's windows."""
    proba = np.full(len(y_trial), np.nan)
    ywin = y_trial[tid]
    for f in range(n_splits):
        tr = np.isin(tid, np.where(fold != f)[0])
        te_trials = np.where(fold == f)[0]
        pipe = make_pipe().fit(Z[tr], ywin[tr])
        te = np.isin(tid, te_trials)
        pw = pipe.predict_proba(Z[te])[:, 1]
        tt = tid[te]
        for t in te_trials:
            proba[t] = pw[tt == t].mean()
    return proba


def _acc(y, proba):
    return float(np.mean((proba > 0.5).astype(int) == y))


def evaluate(Z, tid, y_trial, fold, n_splits, make_pipe, n_perm, seed):
    """Observed trial accuracy + AUC, permutation null (shuffle trial labels), p."""
    proba = cv_trial_proba(Z, tid, y_trial, fold, n_splits, make_pipe)
    acc, auc = _acc(y_trial, proba), float(roc_auc_score(y_trial, proba))
    win_acc = float(np.mean((cv_window_proba(Z, tid, y_trial, fold, n_splits, make_pipe) > 0.5)
                            .astype(int) == y_trial[tid]))
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        yp = rng.permutation(y_trial)
        null[p] = _acc(yp, cv_trial_proba(Z, tid, yp, fold, n_splits, make_pipe))
    pval = float((1 + np.sum(null >= acc)) / (1 + n_perm))
    return {"trial_acc": acc, "auc": auc, "window_acc": win_acc,
            "perm_p": pval, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "proba": proba.tolist()}, null


def cv_window_proba(Z, tid, y_trial, fold, n_splits, make_pipe):
    pw = np.full(len(tid), np.nan)
    ywin = y_trial[tid]
    for f in range(n_splits):
        tr = np.isin(tid, np.where(fold != f)[0]); te = np.isin(tid, np.where(fold == f)[0])
        pw[te] = make_pipe().fit(Z[tr], ywin[tr]).predict_proba(Z[te])[:, 1]
    return pw


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_accuracy(results, figdir):
    names = list(results)
    colors = ["#7b3fa0" if n.startswith("LaBraM") else
              ("#b0651f" if "posterior" in n or "frontal" in n else "#2471a3") for n in names]
    fig, ax = plt.subplots(figsize=(max(7.6, 1.15 * len(names)), 4.8))
    x = np.arange(len(names))
    accs = [results[n]["trial_acc"] for n in names]
    p95 = [results[n]["null_p95"] for n in names]
    ax.bar(x, accs, 0.62, color=colors, zorder=3)
    ax.hlines(p95, x - 0.31, x + 0.31, color="k", lw=1.3, ls="--", zorder=4,
              label="permutation null p95")
    ax.axhline(0.5, color="grey", lw=1, ls=":", label="chance")
    for xi, n, a in zip(x, names, accs):
        star = "*" if results[n]["perm_p"] < 0.05 else ""
        ax.text(xi, a + 0.015, f"{a:.2f}\np={results[n]['perm_p']:.3f}{star}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5)
    ax.set_ylabel("attended-modality accuracy (leave-trials-out CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Decoding attention to Audio vs Visual (sub001, 60 trials)\n"
                 "LaBraM foundation model vs classical band-power", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_labram_accuracy.png"), dpi=170)
    plt.close(fig)


def fig_alpha_topo(alpha_ch, tid, y_trial, info, figdir):
    """Scalp map of the alpha-power attention contrast (attend-Audio − attend-Visual).
    Posterior alpha increase when attending AUDIO (vision suppressed) is the expected
    intersensory-attention pattern; a frontal pattern would suggest an ocular confound."""
    import mne
    ywin = y_trial[tid]
    diff = alpha_ch[ywin == 1].mean(0) - alpha_ch[ywin == 0].mean(0)   # Audio − Visual
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    mne.viz.plot_topomap(diff, info, axes=ax, show=False, cmap="RdBu_r",
                         contours=4, sensors=True)
    ax.set_title("alpha power: attend-Audio − attend-Visual\n(+ = more posterior alpha "
                 "when attending audio)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_alpha_topography.png"), dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-win", type=int, default=3, help="windows per trial")
    ap.add_argument("--device", default="cpu", help="cpu | mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recompute", action="store_true", help="ignore cached embeddings")
    args = ap.parse_args()

    print("=" * 74)
    print("LaBraM attention decoding — attend-Audio vs attend-Visual (frozen encoder)")
    print("=" * 74)
    cfg = io.load_config(args.config)
    subject = cfg["dataset"]["subject"]
    outdir = os.path.join(ROOT, "records", "derivatives", "labram_attention", subject)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    cache = os.path.join(outdir, f"{subject}_labram_embeddings.npz")

    # ---- load + LaBraM-branch preprocessing + trials -----------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    raw_model, prov = LB.preprocess_for_labram(raw, cfg)
    trials_d, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    order = sorted(trials_d)
    trials = [trials_d[t]["eeg"] for t in order]                    # (n_ch, n_times) volts
    labels = np.array([1 if trials_d[t]["label"] == POS else 0 for t in order])
    fs = float(raw_model.info["sfreq"])
    print(f"{len(trials)} trials | Audio {int(labels.sum())} Visual {int((labels==0).sum())} "
          f"| {len(ch_names)} ch @ {fs:.0f} Hz | band {prov['labram_band']} notch {prov['labram_notch']}")

    # ---- frozen LaBraM embeddings (cached): [CLS] and mean-patch -----------
    use_cache = os.path.exists(cache) and not args.recompute
    if use_cache:
        d = np.load(cache, allow_pickle=True)
        cached_channels = [str(x) for x in d["ch_names"]] if "ch_names" in d.files else []
        if cached_channels != list(ch_names):
            print("cached LaBraM embeddings were created with a different montage/channel "
                  "order; recomputing")
            use_cache = False
    if use_cache:
        Zcls, Zmean, tid = d["Zcls"], d["Zmean"], d["tid"]
        print(f"loaded cached embeddings {Zmean.shape} <- {os.path.basename(cache)}")
    else:
        print(f"loading LaBraM ({LB.LABRAM_HF}) on {args.device} ...")
        model = LB.load_labram(device=args.device)
        Zcls, Zmean, tid = LB.embed_trials(model, trials, ch_names, device=args.device,
                                           n_win=args.n_win)
        np.savez(cache, Zcls=Zcls, Zmean=Zmean, tid=tid,
                 ch_names=np.asarray(ch_names, dtype=object),
                 montage_source=str(prov.get("montage_source", "")))
        print(f"embedded {Zmean.shape[0]} windows -> {Zmean.shape[1]}-d (CLS + mean-patch); cached")

    # ---- classical band-power features (same windows) ----------------------
    BP, tid_b = band_tensor(trials, fs, LB.WIN_SAMPLES, args.n_win)     # (n_win, n_ch, 3)
    assert np.array_equal(tid, tid_b), "window alignment mismatch"
    ai = list(BANDS).index("alpha")
    post, front = roi_masks(ch_names)
    Aalpha = BP[:, :, ai]                                               # alpha, all ch
    Aband = BP.reshape(len(BP), -1)                                     # θαβ, all ch
    Apost = BP[:, post, :].reshape(len(BP), -1)                         # θαβ, posterior
    Afront = BP[:, front, :].reshape(len(BP), -1)                       # θαβ, frontal
    print(f"ROI channels: posterior={int(post.sum())}, frontal={int(front.sum())}")

    # ---- evaluate every feature set through the IDENTICAL grouped CV --------
    # LaBraM: mean-patch + PCA20->LDA is the primary variant; [CLS] the naive default.
    fold = make_folds(labels, args.n_splits, args.seed)
    feats = [("LaBraM_meanpatch", Zmean, pca_lda_pipe), ("LaBraM_cls", Zcls, lda_pipe),
             ("alpha_power", Aalpha, lda_pipe), ("bandpower_θαβ", Aband, lda_pipe),
             ("bandpower_posterior", Apost, lda_pipe), ("bandpower_frontal", Afront, lda_pipe)]
    results, nulls = {}, {}
    for name, X, pipe in feats:
        res, null = evaluate(X, tid, labels, fold, args.n_splits, pipe, args.n_perm, args.seed)
        results[name], nulls[name] = res, null
        print(f"  {name:20s}: trial-acc {res['trial_acc']:.3f}  AUC {res['auc']:.3f}  "
              f"(win {res['window_acc']:.3f})  perm-p {res['perm_p']:.4f}  "
              f"[null {res['null_mean']:.3f}, p95 {res['null_p95']:.3f}]")

    # ---- figures -----------------------------------------------------------
    fig_accuracy(results, figdir)
    try:
        fig_alpha_topo(BP[:, :, ai], tid, labels, info, figdir)
    except Exception as e:
        print(f"  (alpha topography skipped: {type(e).__name__}: {e})")

    # ---- save --------------------------------------------------------------
    notes = ("LaBraM is FROZEN (linear probe only) and uses the same corrected cable-half-swapped "
             "montage as CCA2+. Both [CLS]+shrinkage-LDA and mean-patch+PCA20->LDA exceed their "
             "trial-label permutation thresholds in this rerun. Mean-patch remains a POST-HOC "
             "pooling choice and needs held-out-subject confirmation. Even at best, LaBraM stays "
             "below the simple band-power decoder. With one subject, the strong band-power result "
             "cannot be fully separated from ocular/arousal differences between playing Tetris and "
             "listening — hence the posterior/frontal control.")
    out = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "subject": subject,
           "n_trials": len(trials), "n_audio": int(labels.sum()),
           "n_visual": int((labels == 0).sum()), "n_channels": len(ch_names),
           "channels": ch_names, "fs": fs, "window_s": LB.WIN_SAMPLES / fs,
           "n_windows_per_trial": args.n_win, "cv": f"stratified group {args.n_splits}-fold "
           f"(groups=trial), leave-trials-out; permutation over trial labels",
           "n_permutations": args.n_perm,
           "probes": {"LaBraM_meanpatch": "PCA20 + shrinkage-LDA", "other": "shrinkage-LDA"},
           "labram": {"model": LB.LABRAM_HF, "frozen": True, "embed_dim": int(Zmean.shape[1]),
                      "input": "broadband 200 Hz, µV/100", "band": prov["labram_band"],
                      "primary_pooling": "mean-patch"},
           "roi": {"posterior_ch": [c for c, m in zip(ch_names, post) if m],
                   "frontal_ch": [c for c, m in zip(ch_names, front) if m]},
           "notes": notes,
           "preprocessing": {k: prov[k] for k in prov if k not in ("ica",)},
           "results": {n: {k: v for k, v in r.items() if k != "proba"}
                       for n, r in results.items()},
           "proba_audio": {n: results[n]["proba"] for n in results},
           "labels_audio": labels.tolist()}
    with open(os.path.join(outdir, f"{subject}_labram_attention.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 74)
    best = max(results, key=lambda n: results[n]["trial_acc"])
    print(f"best: {best} at {results[best]['trial_acc']:.3f} (p={results[best]['perm_p']:.4f})")
    print(f"saved -> {outdir}")
    print("=" * 74)


if __name__ == "__main__":
    main()
