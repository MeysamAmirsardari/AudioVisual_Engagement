#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_envelope_tracking.py — isolated, leakage-free AUDITORY envelope tracking
via spatio-temporal regularized CCA (de Cheveigne rCCA).

Auditory only — the visual encoding is stripped. Strict nested leave-one-trial-out:

    for each held-out trial t:
        train = all other trials
        (n_pca, shrink) <- K-fold CV over TRAIN ONLY   (models.select_cca_params)
        fit rCCA on TRAIN with the selected hyper-parameters
        project trial t into the learned canonical space
        tracking[t] = mean Pearson correlation of the top-K canonical components
                      on trial t alone  (NO cross-trial normalisation)

Nothing about the held-out trial ever informs the covariance, the whitening, the
hyper-parameters, or the mean-centering. The spatial covariance is stable because
`preprocess.py` now does all bad-channel interpolation + average referencing ONCE
at the continuous level, so every trial shares a fixed channel geometry.

Usage:
    python stim_decoding/evaluate_envelope_tracking.py
    python stim_decoding/evaluate_envelope_tracking.py --k 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

# Silence the spurious numpy-2.x SIMD "divide by zero / overflow / invalid in
# matmul" RuntimeWarnings (padding-lane artifacts; results are unaffected — the
# fast CCA path is bit-exact to the reference). Real errors still surface.
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eeg_analysis"))
from sda import io, preprocess, stimuli, models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Nested-CV rCCA auditory envelope tracking.")
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--k", type=int, default=None, help="top-K canonical components")
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    pc, m = cfg["preprocess"], cfg["model"]
    fs = float(pc["resample_hz"])
    band = (pc["l_freq"], pc["h_freq"])
    k = args.k or int(m["cca_components"])
    eeg_lags = list(range(int(round(m["cca_eeg_lags_s"][0] * fs)),
                          int(round(m["cca_eeg_lags_s"][1] * fs)) + 1))
    env_lags = list(range(int(round(m["cca_env_lags_s"][0] * fs)),
                          int(round(m["cca_env_lags_s"][1] * fs)) + 1))
    n_pca_grid = list(m["cca_n_pca_grid"])
    shrink_grid = list(m["cca_shrink_grid"])
    folds = int(m["cca_cv_folds"])

    print("=" * 72)
    print("AUDITORY ENVELOPE TRACKING — leakage-free nested-CV regularized CCA")
    print("=" * 72)

    # 1) load + align (photodiode, with av-onset recovery) -------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)

    # 2) clean ONCE at the continuous level, then crop constant-geometry blocks
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    print(f"montage : {prov['montage_source']} | {prov['n_channels']} channels")
    print(f"session bad-channels interpolated once: {prov['session_bads_interpolated']}")
    print(f"reference: {prov['reference']}")

    # 3) envelope target per trial + assemble --------------------------------
    audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    tids, EEG, ENV, LAB = [], [], [], []
    for _, row in events.iterrows():
        t = int(row["trial"])
        if t not in trials:
            continue
        wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
        if not os.path.exists(wav):
            continue
        eeg = trials[t]["eeg"].T                        # (n_times, n_ch)
        EEG.append(eeg)
        ENV.append(stimuli.audio_envelope(wav, fs, eeg.shape[0], cfg, band))
        LAB.append(str(row["label"]))
        tids.append(t)
    LAB = np.array(LAB)
    N = len(tids)
    print(f"{N} usable trials | Audio {int((LAB == 'Audio').sum())} / "
          f"Visual {int((LAB == 'Visual').sum())}")
    print(f"EEG lags 0..{eeg_lags[-1]} samp ({len(eeg_lags)}, ~{eeg_lags[-1]/fs*1e3:.0f} ms) | "
          f"env lags ({len(env_lags)}) | K={k} | grid n_pca {n_pca_grid} x shrink {shrink_grid}")

    # 4) pre-lag every trial ONCE (cached; keeps the nested search fast) ------
    Xlist, Ylist = models.lag_trials(EEG, ENV, eeg_lags, env_lags)

    # 5) strict nested LOO ----------------------------------------------------
    # For each held-out trial: tune (n_pca, shrink) on TRAIN by null-corrected CV,
    # fit, then score MATCHED (this trial's envelope) and a NULL (this EEG vs other
    # trials' envelopes -- different speech clips). Genuine tracking = matched-null.
    n_null = 8
    null_rng = np.random.RandomState(1)
    rows = []
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        (npca, sh), cv = models.select_cca_params(
            [Xlist[j] for j in tr], [Ylist[j] for j in tr], k,
            n_pca_grid, shrink_grid, eeg_lags, env_lags, folds=folds)
        model = models.cca_fit_lagged([Xlist[j] for j in tr], [Ylist[j] for j in tr],
                                      k, npca, sh, eeg_lags, env_lags)
        js = null_rng.choice(tr, size=min(n_null, len(tr)), replace=False)
        corr, matched, null = models.null_corrected_score(
            model, Xlist[i], Ylist[i], [Ylist[j] for j in js], k=k)
        rows.append({"trial": int(tids[i]), "label": str(LAB[i]),
                     "matched": float(matched), "null": float(null),
                     "tracking": float(corr), "n_pca": int(npca),
                     "shrink": float(sh), "cv": float(cv)})
        print(f"  trial {tids[i]:2d} [{LAB[i]:6s}] matched={matched:+.3f} "
              f"null={null:+.3f} -> tracking={corr:+.3f}  (n_pca={npca}, shrink={sh})")

    # 6) report (primary = NULL-CORRECTED tracking) ---------------------------
    from scipy import stats
    R = np.array([x["tracking"] for x in rows])      # matched - null (genuine)
    M = np.array([x["matched"] for x in rows])
    Z = np.array([x["null"] for x in rows])
    a, v = R[LAB == "Audio"], R[LAB == "Visual"]
    obs = float(a.mean() - v.mean())
    rng = np.random.RandomState(0)
    lab = LAB.copy()
    perm = np.empty(5000)
    for b in range(5000):
        rng.shuffle(lab)
        perm[b] = R[lab == "Audio"].mean() - R[lab == "Visual"].mean()
    p_perm = float((perm >= obs).mean())
    t1 = stats.ttest_1samp(R, 0.0)

    print("=" * 72)
    print(f"ENVELOPE TRACKING  (top-{k} canonical r; leakage-free nested LOO)")
    print(f"  raw matched  : {M.mean():+.4f}   mismatched null: {Z.mean():+.4f}")
    print(f"  NULL-CORRECTED tracking (matched - null): mean {R.mean():+.4f}  "
          f"SEM {R.std(ddof=1)/np.sqrt(N):.4f}")
    print(f"  genuine tracking > 0 ? one-sample t={t1.statistic:.2f}  p={t1.pvalue:.2g}")
    print(f"  attend-AUDIO  : {a.mean():+.4f}  (n={len(a)})")
    print(f"  attend-VISUAL : {v.mean():+.4f}  (n={len(v)})")
    print(f"  attention effect (Audio - Visual): {obs:+.4f}  "
          f"perm p(1-sided)={p_perm:.3f}")
    print("=" * 72)

    # 7) save -----------------------------------------------------------------
    subject = cfg["dataset"]["subject"]
    out = io.abspath(os.path.join(cfg["output"]["root"], subject,
                                  f"{subject}_envtrack.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "k": k, "eeg_lags_s": [L / fs for L in eeg_lags],
            "env_lags_s": [L / fs for L in env_lags], "n_trials": N,
            "metric": "null_corrected_tracking (matched - mismatched_envelope)",
            "matched_mean": float(M.mean()), "null_mean": float(Z.mean()),
            "overall_mean": float(R.mean()),
            "overall_sem": float(R.std(ddof=1) / np.sqrt(N)),
            "tracking_gt0_p": float(t1.pvalue),
            "audio_mean": float(a.mean()), "visual_mean": float(v.mean()),
            "attention_effect": obs, "attention_p_perm": p_perm,
            "montage_source": prov["montage_source"],
            "session_bads": prov["session_bads_interpolated"],
            "trials": rows,
        }, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
