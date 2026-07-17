#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_auditory_attention.py - can we DETECT whether the subject attended to the
AUDIO stream, using auditory speech-envelope tracking alone?

Builds on decode_auditory_engagement.py. Same auditory-only rCCA machinery, but now
the question is DISCRIMINATIVE: attend-Audio vs attend-Visual. It produces

  1. FEATURE ENGINEERING - a leakage-free per-trial feature vector describing how
     the EEG tracks the speech (overall + top canonical correlations + early/late +
     temporal profile of tracking across the trial).
  2. ATTENTION DETECTION - an LDA classifier on those features, leave-one-trial-out,
     with a label-shuffle permutation null -> accuracy, ROC-AUC, p-values.
  3. CONDITION SPATIOTEMPORAL MAPS - the common canonical filter expressed through
     each condition's own covariance: Haufe forward maps (channels x lags) for
     attend-Audio, attend-Visual, and their difference, plus per-condition TRFs.
  4. FEATURE COMPARISON - per-trial features and the tracking time-course contrasted
     between conditions, with univariate effect sizes / AUCs.
  5. CCA ANATOMY - a detailed diagnostic of the CCA itself: covariance eigenspectrum
     and the PCA cut, the canonical-correlation spectrum, the leading canonical
     variates (EEG vs envelope), and the envelope-side temporal weights.

All figures + a classification/feature JSON + a feature .npz are saved next to the
engagement outputs:  records/derivatives/auditory_engagement/<subject>/

Usage:
    python stim_decoding/compare_auditory_attention.py
    python stim_decoding/compare_auditory_attention.py --perms 2000 --n-win 8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess, stimuli, models  # noqa: E402

FEATS = ["engagement", "track_mean", "track_top1",
         "track_early", "track_late", "track_slope", "track_std"]


# ==========================================================================
# small helpers
# ==========================================================================
def _uv(model, X, Y):
    """Project one trial's pre-lagged (X, Y) onto the fitted canonical directions."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        U = np.asarray((X - model["mx"]) @ model["Ax"])
        V = np.asarray((Y - model["my"]) @ model["Ay"])
    return U, V


def _corrs(U, V, k):
    cs = [np.corrcoef(U[:, c], V[:, c])[0, 1] for c in range(min(k, U.shape[1]))]
    return np.array([c if np.isfinite(c) else 0.0 for c in cs])


def _window_track(U, V, n_win, k):
    """Mean top-K canonical correlation in each of n_win equal time windows."""
    T = U.shape[0]
    e = np.linspace(0, T, n_win + 1).astype(int)
    return np.array([_corrs(U[a:b], V[a:b], k).mean() for a, b in zip(e[:-1], e[1:])])


def _sign_fix(P):
    return P * np.sign(P.flat[np.argmax(np.abs(P))])


def _pattern(Cxx, Ax, n_lags, n_ch, comp=0):
    return (Cxx @ Ax).reshape(n_lags, n_ch, -1)[:, :, comp]


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Detect attend-Audio vs attend-Visual.")
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--n-win", type=int, default=6, help="tracking time windows")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    pc, m = cfg["preprocess"], cfg["model"]
    fs = float(pc["resample_hz"])
    band = (pc["l_freq"], pc["h_freq"])
    ktrack = int(m["cca_components"])
    eeg_lags = list(range(int(round(m["cca_eeg_lags_s"][0] * fs)),
                          int(round(m["cca_eeg_lags_s"][1] * fs)) + 1))
    env_lags = list(range(int(round(m["cca_env_lags_s"][0] * fs)),
                          int(round(m["cca_env_lags_s"][1] * fs)) + 1))
    # Hyper-parameters fixed a priori to the values nested CV selected in the
    # engagement analysis (fixing them avoids nested selection here and, being
    # label-free, cannot leak the attend-Audio/Visual labels we are classifying).
    NPCA, SH = 220, 0.005
    subject = cfg["dataset"]["subject"]
    outdir = args.out or os.path.join(ROOT, "records", "derivatives",
                                      "auditory_engagement", subject)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 74)
    print("AUDITORY ATTENTION DETECTION - attend-Audio vs attend-Visual (rCCA)")
    print("=" * 74)

    # ---- load + clean + envelopes ------------------------------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    n_ch, n_lags = len(ch_names), len(eeg_lags)
    audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    tids, EEG, ENV, LAB = [], [], [], []
    for _, row in events.iterrows():
        t = int(row["trial"])
        wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
        if t not in trials or not os.path.exists(wav):
            continue
        eeg = trials[t]["eeg"].T
        EEG.append(eeg)
        ENV.append(stimuli.audio_envelope(wav, fs, eeg.shape[0], cfg, band))
        LAB.append(str(row["label"]))
        tids.append(t)
    LAB = np.array(LAB)
    N = len(tids)
    Xlist, Ylist = models.lag_trials(EEG, ENV, eeg_lags, env_lags)
    y = (LAB == "Audio").astype(int)
    print(f"{N} trials | attend-Audio {int(y.sum())} / attend-Visual {int((1-y).sum())}")

    # ---- 1. leakage-free per-trial features (LOO rCCA) ---------------------
    print("feature engineering (leave-one-trial-out rCCA projections)...")
    nw = int(args.n_win)
    F = np.zeros((N, len(FEATS)))
    wins = np.zeros((N, nw))
    nrng = np.random.RandomState(1)
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        model = models.cca_fit_lagged([Xlist[j] for j in tr], [Ylist[j] for j in tr],
                                      ktrack, NPCA, SH, eeg_lags, env_lags)
        U, V = _uv(model, Xlist[i], Ylist[i])
        rk = _corrs(U, V, ktrack)
        js = nrng.choice(tr, size=min(8, len(tr)), replace=False)
        null = np.mean([models.cca_score_lagged(model, Xlist[i], Ylist[j], k=ktrack)
                        for j in js])
        w = _window_track(U, V, nw, ktrack)
        wins[i] = w
        F[i] = [rk.mean() - null, rk.mean(), rk[0],
                w[:nw // 2].mean(), w[nw // 2:].mean(),
                np.polyfit(np.arange(nw), w, 1)[0], w.std()]

    # ---- 2. attention detection (LDA, LOO, permutation) --------------------
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    loo = LeaveOneOut()

    def _cv(Xf, yy):
        pred = cross_val_predict(pipe, Xf, yy, cv=loo)
        dec = cross_val_predict(pipe, Xf, yy, cv=loo, method="decision_function")
        return accuracy_score(yy, pred), roc_auc_score(yy, dec), pred, dec

    acc, auc, pred, dec = _cv(F, y)
    prng = np.random.RandomState(0)
    pacc = np.empty(args.perms)
    pauc = np.empty(args.perms)
    for b in range(args.perms):
        yp = prng.permutation(y)
        a_, u_, _, _ = _cv(F, yp)
        pacc[b], pauc[b] = a_, u_
    p_acc = float((pacc >= acc).mean())
    p_auc = float((pauc >= auc).mean())
    cm = confusion_matrix(y, pred).tolist()
    feat_auc = {f: float(roc_auc_score(y, F[:, j])) for j, f in enumerate(FEATS)}
    print(f"  LDA leave-one-out: accuracy {acc:.3f} (perm p={p_acc:.3f}) | "
          f"AUC {auc:.3f} (perm p={p_auc:.3f})")
    print(f"  chance {max(y.mean(), 1-y.mean()):.3f} | "
          f"best single feature: " +
          max(feat_auc, key=lambda f: abs(feat_auc[f] - 0.5)) +
          f" AUC={max(feat_auc.values(), key=lambda v: abs(v-0.5)):.3f}")

    # ---- 3. condition spatiotemporal maps (common filter, per-cond cov) ----
    ai = [i for i in range(N) if LAB[i] == "Audio"]
    vi = [i for i in range(N) if LAB[i] == "Visual"]
    model_all = models.cca_fit_lagged(Xlist, Ylist, max(ktrack, len(env_lags)),
                                      NPCA, SH, eeg_lags, env_lags)
    Ax = model_all["Ax"]
    Cxx_all, Cyy_all, Cxy_all, mx_all, my_all = models._pool(Xlist, Ylist)
    Cxx_a = models._pool([Xlist[i] for i in ai], [Ylist[i] for i in ai])[0]
    Cxx_v = models._pool([Xlist[i] for i in vi], [Ylist[i] for i in vi])[0]
    Pa = _pattern(Cxx_a, Ax, n_lags, n_ch)
    Pv = _pattern(Cxx_v, Ax, n_lags, n_ch)
    s = np.sign((Pa + Pv).flat[np.argmax(np.abs(Pa + Pv))])   # shared sign
    Pa, Pv = s * Pa, s * Pv
    Pd = Pa - Pv
    gfp_a = np.sqrt((Pa ** 2).mean(1))
    gfp_v = np.sqrt((Pv ** 2).mean(1))
    lag_ms = np.array(eeg_lags) / fs * 1e3

    # ---- figures ------------------------------------------------------------
    import mne  # noqa: F401
    _fig_detection(y, dec, pred, cm, acc, auc, pacc, p_acc, figdir)
    _fig_condition_maps(Pa, Pv, Pd, lag_ms, info, figdir)
    _fig_condition_trf(gfp_a, gfp_v, lag_ms, F, y, wins, figdir)
    _fig_features(F, y, feat_auc, wins, figdir)
    _fig_cca_anatomy(Cxx_all, model_all, Xlist, Ylist, mx_all, my_all,
                     eeg_lags, env_lags, fs, NPCA, figdir)

    # ---- save ---------------------------------------------------------------
    result = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "question": "detect attend-Audio vs attend-Visual from auditory tracking",
        "n_trials": N, "n_audio": int(y.sum()), "n_visual": int((1 - y).sum()),
        "classifier": "LDA on engineered tracking features, leave-one-trial-out",
        "features": FEATS, "n_windows": nw,
        "hyperparams_fixed": {"n_pca": NPCA, "shrink": SH, "k_track": ktrack,
                              "note": "CV-selected in engagement analysis; fixed here"},
        "accuracy": float(acc), "accuracy_perm_p": p_acc,
        "auc": float(auc), "auc_perm_p": p_auc,
        "chance": float(max(y.mean(), 1 - y.mean())),
        "confusion_matrix": cm, "confusion_labels": ["Visual", "Audio"],
        "univariate_feature_auc": feat_auc,
        "attend_audio_track_mean": float(F[y == 1, 1].mean()),
        "attend_visual_track_mean": float(F[y == 0, 1].mean()),
        "canonical_corrs_all": [float(c) for c in model_all["corrs"]],
    }
    with open(os.path.join(outdir, f"{subject}_attention.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        os.path.join(outdir, f"{subject}_features.npz"),
        features=F, feature_names=np.array(FEATS, object), labels=LAB, y=y,
        windows=wins, decision=dec, pattern_audio=Pa, pattern_visual=Pv,
        pattern_diff=Pd, trf_gfp_audio=gfp_a, trf_gfp_visual=gfp_v, lag_ms=lag_ms)

    print("=" * 74)
    print(f"saved -> {outdir}")
    for fn in sorted(os.listdir(outdir)):
        print("   ", fn)
    for fn in sorted(os.listdir(figdir)):
        print("    figures/", fn)
    print("=" * 74)


# ==========================================================================
# figures
# ==========================================================================
def _fig_detection(y, dec, pred, cm, acc, auc, pacc, p_acc, figdir):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, dec)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(fpr, tpr, lw=2, color="#1a6"); ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title(f"ROC for attend-Audio detection\nAUC={auc:.3f}")
    im = ax[1].imshow(cm, cmap="Blues"); ax[1].set_xticks([0, 1]); ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(["Visual", "Audio"]); ax[1].set_yticklabels(["Visual", "Audio"])
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true")
    ax[1].set_title(f"confusion (acc={acc:.3f})")
    for (r, c), val in np.ndenumerate(np.array(cm)):
        ax[1].text(c, r, int(val), ha="center", va="center",
                   color="white" if val > np.max(cm) / 2 else "black")
    ax[2].hist(pacc, bins=25, color="#bbb", label="label-shuffle null")
    ax[2].axvline(acc, color="crimson", lw=2, label=f"observed {acc:.3f}\np={p_acc:.3f}")
    ax[2].set_xlabel("LOO accuracy"); ax[2].set_ylabel("permutations")
    ax[2].set_title("permutation null"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "attention_detection.png"), dpi=150)
    plt.close(fig)


def _fig_condition_maps(Pa, Pv, Pd, lag_ms, info, figdir):
    import mne
    show = [0, 60, 100, 150, 200, 250]
    idx = [int(np.argmin(np.abs(lag_ms - L))) for L in show]
    rows = [("attend-Audio", Pa), ("attend-Visual", Pv), ("Audio - Visual", Pd)]
    fig, axes = plt.subplots(3, len(idx), figsize=(2.0 * len(idx), 6.4))
    for r, (name, P) in enumerate(rows):
        vmax = np.percentile(np.abs(P), 99)
        for c, li in enumerate(idx):
            mne.viz.plot_topomap(P[li], info, axes=axes[r, c], show=False,
                                 cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
            if r == 0:
                axes[r, c].set_title(f"{lag_ms[li]:.0f} ms", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("Spatiotemporal envelope-tracking pattern by attended stream "
                 "(common canonical filter x per-condition covariance)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(figdir, "condition_spatiotemporal.png"), dpi=150)
    plt.close(fig)


def _fig_condition_trf(gfp_a, gfp_v, lag_ms, F, y, wins, figdir):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(lag_ms, gfp_a, lw=2, color="#c33", label="attend-Audio")
    ax[0].plot(lag_ms, gfp_v, lw=2, color="#36c", label="attend-Visual")
    ax[0].set_xlabel("lag (ms)"); ax[0].set_ylabel("pattern GFP")
    ax[0].set_title("TRF by attended stream"); ax[0].legend()
    # tracking strength per condition
    ta, tv = F[y == 1, 1], F[y == 0, 1]
    ax[1].bar([0, 1], [ta.mean(), tv.mean()],
              yerr=[ta.std(ddof=1)/np.sqrt(len(ta)), tv.std(ddof=1)/np.sqrt(len(tv))],
              color=["#c33", "#36c"], capsize=5)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Audio", "Visual"])
    ax[1].set_ylabel("mean tracking (top-K r)"); ax[1].set_title("tracking strength")
    # tracking time-course per condition
    nw = wins.shape[1]
    wa, wv = wins[y == 1], wins[y == 0]
    xw = np.arange(nw)
    ax[2].errorbar(xw, wa.mean(0), wa.std(0)/np.sqrt(len(wa)), color="#c33",
                   label="Audio", capsize=3)
    ax[2].errorbar(xw, wv.mean(0), wv.std(0)/np.sqrt(len(wv)), color="#36c",
                   label="Visual", capsize=3)
    ax[2].set_xlabel("trial window"); ax[2].set_ylabel("tracking (top-K r)")
    ax[2].set_title("tracking over the trial"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "condition_trf.png"), dpi=150)
    plt.close(fig)


def _fig_features(F, y, feat_auc, wins, figdir):
    nF = F.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes = axes.ravel()
    for j in range(nF):
        a, v = F[y == 1, j], F[y == 0, j]
        axes[j].boxplot([a, v], tick_labels=["Aud", "Vis"])
        axes[j].set_title(f"{FEATS[j]}\nAUC={feat_auc[FEATS[j]]:.2f}", fontsize=9)
    # last panel: univariate discriminability
    ax = axes[-1]
    order = sorted(range(nF), key=lambda j: abs(feat_auc[FEATS[j]] - 0.5))
    ax.barh([FEATS[j] for j in order],
            [feat_auc[FEATS[j]] - 0.5 for j in order], color="#568")
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("AUC - 0.5")
    ax.set_title("feature discriminability")
    fig.suptitle("Per-trial auditory-tracking features: attend-Audio vs attend-Visual",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(figdir, "feature_comparison.png"), dpi=150)
    plt.close(fig)


def _fig_cca_anatomy(Cxx, model, Xlist, Ylist, mx, my, eeg_lags, env_lags, fs,
                     n_pca, figdir):
    ev = np.linalg.eigvalsh(Cxx)[::-1]
    ev = np.clip(ev, 1e-30, None)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) covariance eigenspectrum + PCA cut
    ax[0, 0].semilogy(ev / ev[0], ".-", ms=3, color="#333")
    ax[0, 0].axvline(min(n_pca, len(ev)) - 1, color="crimson", ls="--",
                     label=f"PCA cut n_pca={n_pca}")
    ax[0, 0].set_xlabel("component"); ax[0, 0].set_ylabel("eigenvalue (norm.)")
    ax[0, 0].set_title("EEG covariance spectrum (whitening)"); ax[0, 0].legend()
    # (b) canonical correlation spectrum
    cc = np.asarray(model["corrs"])
    ax[0, 1].bar(range(1, len(cc) + 1), cc, color="#1a6")
    for i, c in enumerate(cc):
        ax[0, 1].text(i + 1, c + 0.005, f"{c:.2f}", ha="center", fontsize=9)
    ax[0, 1].set_xlabel("canonical component"); ax[0, 1].set_ylabel("canonical r (train)")
    ax[0, 1].set_title("canonical correlation spectrum")
    # (c) leading canonical variates: EEG vs envelope projection
    U = np.concatenate([np.asarray((X - mx) @ model["Ax"])[:, 0] for X in Xlist])
    V = np.concatenate([np.asarray((Y - my) @ model["Ay"])[:, 0] for Y in Ylist])
    sub = np.random.RandomState(0).choice(len(U), size=min(4000, len(U)), replace=False)
    r1 = np.corrcoef(U, V)[0, 1]
    ax[1, 0].scatter(U[sub], V[sub], s=3, alpha=0.25, color="#a2560f")
    ax[1, 0].set_xlabel("EEG canonical variate U1"); ax[1, 0].set_ylabel("envelope V1")
    ax[1, 0].set_title(f"leading canonical pair (pooled r={r1:.2f})")
    # (d) envelope-side temporal weights
    env_ms = np.array(env_lags) / fs * 1e3
    Ay = np.asarray(model["Ay"])
    for c in range(min(3, Ay.shape[1])):
        ax[1, 1].plot(env_ms, Ay[:, c] * np.sign(Ay[np.argmax(np.abs(Ay[:, c])), c]),
                      ".-", label=f"comp {c+1}")
    ax[1, 1].set_xlabel("envelope lag (ms)"); ax[1, 1].set_ylabel("weight")
    ax[1, 1].set_title("envelope-side canonical weights"); ax[1, 1].legend(fontsize=8)
    fig.suptitle("CCA anatomy: regularized spatio-temporal rCCA", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(figdir, "cca_anatomy.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
