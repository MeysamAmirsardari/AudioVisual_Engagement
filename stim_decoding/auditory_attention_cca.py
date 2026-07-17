#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auditory_attention_cca.py - does the brain track the speech envelope MORE when the
subject attends the AUDIO stream than when they attend the VISUAL (Tetris) stream?

This is the sensitive, field-standard test of attentional modulation of auditory
tracking, built on the CCA model (de Cheveigne et al. 2018) and its recommended
match-vs-mismatch evaluation (their Fig 7). Earlier crude tests (mean top-K
canonical correlation over the whole 19 s trial) diluted any effect; here we use:

  1. ONE common CCA stimulus-tracking filter, fit leakage-free by leave-one-trial-out
     (never per condition -> the filter models tracking, not attention).
  2. A JOINT match-vs-mismatch LDA on the multivariate canonical-correlation vector
     (trained on the training trials only) -> a calibrated tracking-strength score,
     far more sensitive than a single correlation.
  3. A per-trial AUDITORY-TRACKING INDEX (that LDA score for the true envelope),
     compared attend-Audio vs attend-Visual (Welch t + permutation, Cohen's d).
  4. AAD ACCURACY vs window length per condition (match-vs-mismatch over 1-16 s
     segments): attentional enhancement should make attend-Audio > attend-Visual.
  5. A single-trial ATTENDED-MODALITY decoder (LDA over the tracking features,
     leave-one-trial-out + label permutation) -> can we detect the attended stream?

All figures + JSON -> records/derivatives/cca_decoding/<subject>/.

Usage:
    python stim_decoding/auditory_attention_cca.py
    python stim_decoding/auditory_attention_cca.py --perms 5000
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess, stimuli, models  # noqa: E402

COND = {"Audio": "#c0392b", "Visual": "#2471a3"}
NPCA, SHRINK, KTRACK = 120, 0.1, 6
WIN_SECS = [1, 2, 4, 8, 16]


def _project(model, X, Y):
    with np.errstate(all="ignore"):
        U = np.asarray((X - model["mx"]) @ model["Ax"])
        V = np.asarray((Y - model["my"]) @ model["Ay"])
    return U, V


def _corrvec(U, V, k):
    out = np.empty(k)
    for c in range(k):
        a, b = U[:, c], V[:, c]
        if a.std() == 0 or b.std() == 0:
            out[c] = 0.0
        else:
            r = np.corrcoef(a, b)[0, 1]
            out[c] = r if np.isfinite(r) else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    pc = cfg["preprocess"]
    fs = float(pc["resample_hz"])
    band = (pc["l_freq"], pc["h_freq"])
    eeg_lags = list(range(0, int(round(0.25 * fs)) + 1))
    feat_lags = list(range(0, int(round(0.25 * fs)) + 1))
    k = KTRACK
    subject = cfg["dataset"]["subject"]
    outdir = args.out or os.path.join(ROOT, "records", "derivatives",
                                      "cca_decoding", subject)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 74)
    print("AUDITORY ATTENTION via CCA - attend-Audio vs attend-Visual (match/mismatch)")
    print("=" * 74)

    # ---- load + clean + envelopes ------------------------------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    EEG, ENV, LAB = [], [], []
    for _, row in events.iterrows():
        t = int(row["trial"])
        wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
        if t not in trials or not os.path.exists(wav):
            continue
        e = trials[t]["eeg"].T
        EEG.append(e)
        ENV.append(stimuli.audio_envelope(wav, fs, e.shape[0], cfg, band))
        LAB.append(str(row["label"]))
    LAB = np.array(LAB)
    N = len(EEG)
    y = (LAB == "Audio").astype(int)
    Xlag, Ylag = models.lag_trials(EEG, ENV, eeg_lags, feat_lags)
    T = Xlag[0].shape[0]
    edge = max(eeg_lags + feat_lags)
    print(f"{N} trials | Audio {int(y.sum())} Visual {int((1-y).sum())} | "
          f"CCA model 2 (n_pca={NPCA}, shrink={SHRINK}, K={k})")

    # ---- leakage-free LOO: tracking index + AAD-vs-window -------------------
    print("leave-one-trial-out CCA + joint match/mismatch LDA ...")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rng = np.random.RandomState(0)
    track_idx = np.zeros(N)
    aad = np.zeros((N, len(WIN_SECS)))
    matched_r = np.zeros(N)                                    # raw mean top-K corr
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        model = models.cca_fit_lagged([Xlag[j] for j in tr], [Ylag[j] for j in tr],
                                      k, NPCA, SHRINK, eeg_lags, feat_lags)
        U = [None] * N; V = [None] * N
        for j in range(N):
            U[j], V[j] = _project(model, Xlag[j], Ylag[j])
        # joint match/mismatch LDA on TRAIN trials only
        pos = [_corrvec(U[j], V[j], k) for j in tr]
        neg = []
        for j in tr:
            for l in rng.choice([x for x in tr if x != j], size=4, replace=False):
                neg.append(_corrvec(U[j], V[l], k))
        Xd = np.vstack(pos + neg)
        yd = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        lda.fit(Xd, yd)
        # test trial: continuous tracking index (leakage-free)
        mvec = _corrvec(U[i], V[i], k)
        track_idx[i] = float(lda.decision_function([mvec])[0])
        matched_r[i] = float(mvec.mean())
        # AAD accuracy vs window length (match vs all other envelopes)
        comps = [l for l in tr]
        for wi, ws in enumerate(WIN_SECS):
            wlen = int(ws * fs)
            starts = list(range(edge, T - edge - wlen + 1, wlen)) or [edge]
            acc = []
            for a in starts:
                b = a + wlen
                sm = _corrvec(U[i][a:b], V[i][a:b], k).mean()
                sc = np.array([_corrvec(U[i][a:b], V[l][a:b], k).mean() for l in comps])
                acc.append(float(np.mean(sm > sc)))
            aad[i, wi] = float(np.mean(acc))
        if (i + 1) % 15 == 0:
            print(f"    {i+1}/{N}")

    # ---- 1. attention contrast on the tracking index -----------------------
    from scipy import stats
    a_idx, v_idx = track_idx[y == 1], track_idx[y == 0]
    tt = stats.ttest_ind(a_idx, v_idx, equal_var=False)
    pooled_sd = np.sqrt(((len(a_idx)-1)*a_idx.var(ddof=1) +
                         (len(v_idx)-1)*v_idx.var(ddof=1)) / (len(a_idx)+len(v_idx)-2))
    d = float((a_idx.mean() - v_idx.mean()) / (pooled_sd + 1e-12))
    obs = float(a_idx.mean() - v_idx.mean())
    prng = np.random.RandomState(1); lab = y.copy(); perm = np.empty(args.perms)
    for bidx in range(args.perms):
        prng.shuffle(lab)
        perm[bidx] = track_idx[lab == 1].mean() - track_idx[lab == 0].mean()
    p_perm = float((perm >= obs).mean())
    print(f"  tracking index: Audio {a_idx.mean():+.3f} vs Visual {v_idx.mean():+.3f} | "
          f"d={d:+.2f} | t p={tt.pvalue:.3f} | perm p(1-sided)={p_perm:.4f}")

    # ---- 2. AAD accuracy vs window, per condition ---------------------------
    aad_a, aad_v = aad[y == 1], aad[y == 0]
    print("  AAD accuracy (match vs mismatch), Audio | Visual:")
    for wi, ws in enumerate(WIN_SECS):
        print(f"    {ws:2d}s : {aad_a[:,wi].mean():.3f} | {aad_v[:,wi].mean():.3f}")

    # ---- 3. single-trial attended-modality decoder --------------------------
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
    Feat = np.column_stack([track_idx, matched_r, aad])
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    loo = LeaveOneOut()
    pred = cross_val_predict(pipe, Feat, y, cv=loo)
    dec = cross_val_predict(pipe, Feat, y, cv=loo, method="decision_function")
    acc = accuracy_score(y, pred); auc = roc_auc_score(y, dec)
    prng2 = np.random.RandomState(2); pa = np.empty(args.perms // 5)
    for bidx in range(len(pa)):
        yp = prng2.permutation(y)
        pa[bidx] = accuracy_score(yp, cross_val_predict(pipe, Feat, yp, cv=loo))
    p_dec = float((pa >= acc).mean())
    print(f"  attended-modality decoder: acc {acc:.3f} (perm p={p_dec:.3f}) | AUC {auc:.3f}")

    # ---- figures ------------------------------------------------------------
    _fig_index(track_idx, y, a_idx, v_idx, d, p_perm, perm, obs, figdir)
    _fig_aad(aad_a, aad_v, figdir)
    _fig_decode(y, dec, pred, acc, auc, pa, p_dec, figdir)

    # ---- save ---------------------------------------------------------------
    out = {"created": _dt.datetime.now().isoformat(timespec="seconds"),
           "method": "CCA (de Cheveigne 2018) + joint match/mismatch LDA, LOO",
           "n_trials": N, "n_audio": int(y.sum()), "n_visual": int((1-y).sum()),
           "cca": {"n_pca": NPCA, "shrink": SHRINK, "k": k,
                   "eeg_lags_s": [L/fs for L in eeg_lags],
                   "feat_lags_s": [L/fs for L in feat_lags]},
           "tracking_index": {"audio_mean": float(a_idx.mean()),
                              "visual_mean": float(v_idx.mean()),
                              "cohens_d": d, "welch_t": float(tt.statistic),
                              "welch_p": float(tt.pvalue), "perm_p_1sided": p_perm},
           "aad_accuracy": {f"{ws}s": {"audio": float(aad_a[:, wi].mean()),
                                       "visual": float(aad_v[:, wi].mean())}
                            for wi, ws in enumerate(WIN_SECS)},
           "attended_modality_decoder": {"accuracy": float(acc), "auc": float(auc),
                                         "perm_p": p_dec, "chance": 0.5},
           "per_trial": [{"trial": int(i), "label": str(LAB[i]),
                          "tracking_index": float(track_idx[i]),
                          "matched_r": float(matched_r[i])} for i in range(N)]}
    with open(os.path.join(outdir, f"{subject}_attention_cca.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 74)
    print(f"saved -> {outdir}")
    for fn in ("attn_tracking_index.png", "attn_aad_vs_window.png",
               "attn_attended_decode.png", f"{subject}_attention_cca.json"):
        print("   ", fn)
    print("=" * 74)


# ==========================================================================
def _fig_index(track_idx, y, a_idx, v_idx, d, p_perm, perm, obs, figdir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    parts = ax[0].violinplot([a_idx, v_idx], showmeans=True)
    for pc, c in zip(parts["bodies"], COND.values()):
        pc.set_facecolor(c); pc.set_alpha(0.4)
    for xj, arr, c in [(1, a_idx, COND["Audio"]), (2, v_idx, COND["Visual"])]:
        ax[0].scatter(np.random.RandomState(xj).normal(xj, 0.05, len(arr)), arr,
                      s=18, color=c, alpha=0.7, zorder=3)
    ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["attend-Audio", "attend-Visual"])
    ax[0].set_ylabel("auditory-tracking index (LDA score)")
    ax[0].set_title(f"per-trial tracking\nCohen d={d:+.2f}, perm p={p_perm:.3f}")
    ax[1].hist(perm, bins=40, color="#bbb", label="label-shuffle null")
    ax[1].axvline(obs, color="crimson", lw=2, label=f"observed {obs:+.3f}")
    ax[1].set_xlabel("Audio - Visual tracking index"); ax[1].set_ylabel("permutations")
    ax[1].set_title("attention-effect permutation null"); ax[1].legend(fontsize=8)
    fig.suptitle("Auditory attention: envelope-tracking strength by attended stream",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(figdir, "attn_tracking_index.png"), dpi=150); plt.close(fig)


def _fig_aad(aad_a, aad_v, figdir):
    x = np.arange(len(WIN_SECS))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.errorbar(x, aad_a.mean(0), aad_a.std(0)/np.sqrt(len(aad_a)), color=COND["Audio"],
                lw=2, marker="o", capsize=4, label="attend-Audio")
    ax.errorbar(x, aad_v.mean(0), aad_v.std(0)/np.sqrt(len(aad_v)), color=COND["Visual"],
                lw=2, marker="o", capsize=4, label="attend-Visual")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([f"{w}s" for w in WIN_SECS])
    ax.set_xlabel("decision window length"); ax.set_ylabel("match-vs-mismatch accuracy")
    ax.set_title("Auditory attention decoding accuracy vs window length")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "attn_aad_vs_window.png"), dpi=150); plt.close(fig)


def _fig_decode(y, dec, pred, acc, auc, pa, p_dec, figdir):
    from sklearn.metrics import roc_curve, confusion_matrix
    fpr, tpr, _ = roc_curve(y, dec); cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(fpr, tpr, lw=2, color="#1a6"); ax[0].plot([0, 1], [0, 1], "k:")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title(f"attended-modality ROC\nAUC={auc:.3f}")
    ax[1].imshow(cm, cmap="Blues"); ax[1].set_xticks([0, 1]); ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(["Visual", "Audio"]); ax[1].set_yticklabels(["Visual", "Audio"])
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true")
    ax[1].set_title(f"confusion (acc={acc:.3f})")
    for (r, c), v in np.ndenumerate(cm):
        ax[1].text(c, r, int(v), ha="center", va="center",
                   color="white" if v > cm.max()/2 else "black")
    ax[2].hist(pa, bins=20, color="#bbb", label="label-shuffle null")
    ax[2].axvline(acc, color="crimson", lw=2, label=f"observed {acc:.3f}\np={p_dec:.3f}")
    ax[2].set_xlabel("LOO accuracy"); ax[2].set_ylabel("permutations")
    ax[2].set_title("decoder permutation null"); ax[2].legend(fontsize=8)
    fig.suptitle("Can we detect the attended stream from auditory tracking?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(figdir, "attn_attended_decode.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
