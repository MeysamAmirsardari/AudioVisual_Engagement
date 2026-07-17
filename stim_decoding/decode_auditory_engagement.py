#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_auditory_engagement.py - quantify AUDITORY ENGAGEMENT (neural tracking of
the speech envelope) from EEG, and extract its SPATIOTEMPORAL PATTERN.

Auditory only (visual information ignored). The method is deliberately rigorous:

  1. ENGAGEMENT per trial - strict, leakage-free nested leave-one-trial-out
     regularized spatio-temporal CCA (de Cheveigne rCCA): hyper-parameters tuned
     by K-fold CV on the training trials only, fit on train, scored on the single
     held-out trial. The index is NULL-CORRECTED (matched envelope minus this
     EEG's correlation with OTHER trials' envelopes), so it measures tracking of
     *this* speech, not any slow signal.

  2. SIGNIFICANCE - one-sample test on the out-of-sample null-corrected index, and
     an envelope-shuffle permutation null (re-pair EEG with random envelopes) for
     a group z-score.

  3. SPATIOTEMPORAL PATTERN - the rCCA canonical directions are turned into a
     Haufe forward pattern (channels x time-lags): the interpretable neural map of
     where and when the envelope is tracked. Its global field power over lags is
     the temporal response function (auditory N1/P2-like latencies).

  4. RELIABILITY - the pattern is re-estimated on B bootstrap resamples of trials;
     the mean bootstrap correlation with the full pattern is the reliability, and
     the element-wise mean/std gives a z-map of the trustworthy pattern regions.

Everything (method + provenance, per-trial engagement, patterns, and figures) is
saved to a NEW directory:  records/derivatives/auditory_engagement/<subject>/

Usage:
    python stim_decoding/decode_auditory_engagement.py
    python stim_decoding/decode_auditory_engagement.py --bootstrap 300 --perms 5000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess, stimuli, models  # noqa: E402


# ==========================================================================
# helpers
# ==========================================================================
def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=ROOT)
        return out.stdout.strip() or None
    except Exception:
        return None


def _fit_pattern(Xlist, Ylist, eeg_lags, env_lags, k, n_pca, shrink,
                 n_ch, n_lags):
    """rCCA on the given (pre-lagged) trials -> leading-component Haufe pattern
    (n_lags, n_ch), the full model, and the canonical correlations."""
    Cxx, Cyy, Cxy, mx, my = models._pool(Xlist, Ylist)
    evx, Vx = models._eig(Cxx)
    evy, Vy = models._eig(Cyy)
    model = models._model_from_eig(evx, Vx, evy, Vy, Cxx, Cyy, Cxy, mx, my,
                                   k, n_pca, shrink, eeg_lags, env_lags)
    P = np.asarray(model["Px"]).reshape(n_lags, n_ch, -1)      # (lag, ch, comp)
    return P[:, :, 0], model


def _sign_fix(P):
    """Fix the arbitrary CCA sign so the largest-magnitude weight is positive."""
    return P * np.sign(P.flat[np.argmax(np.abs(P))])


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Auditory engagement decoder (rCCA).")
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--k", type=int, default=None, help="top-K canonical comps")
    ap.add_argument("--bootstrap", type=int, default=300, help="pattern bootstraps")
    ap.add_argument("--perms", type=int, default=5000, help="permutation nulls")
    ap.add_argument("--out", default=None, help="output directory")
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
    subject = cfg["dataset"]["subject"]
    outdir = args.out or os.path.join(ROOT, "records", "derivatives",
                                      "auditory_engagement", subject)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 74)
    print("AUDITORY ENGAGEMENT - neural speech-envelope tracking (rCCA)")
    print("=" * 74)

    # ---- 1. load + clean (session-level) + envelopes -----------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    n_ch, n_lags = len(ch_names), len(eeg_lags)
    print(f"montage: {prov['montage_source']} | {n_ch} ch | ref: {prov['reference']}")
    print(f"session bads interpolated once: {prov['session_bads_interpolated']}")

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
    print(f"{N} trials | EEG lags 0-{eeg_lags[-1]/fs*1e3:.0f} ms ({n_lags}) | K={k}")

    # ---- 2. engagement per trial: nested LOO, null-corrected ---------------
    print("nested leave-one-trial-out (null-corrected)...")
    n_null, null_rng = 8, np.random.RandomState(1)
    rows, sel = [], []
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
                     "engagement": float(corr), "matched": float(matched),
                     "null": float(null)})
        sel.append((npca, sh))
    R = np.array([r["engagement"] for r in rows])
    M = np.array([r["matched"] for r in rows])
    Z = np.array([r["null"] for r in rows])
    npca_star = int(np.bincount([s[0] for s in sel]).argmax())
    sh_star = float(max(set(s[1] for s in sel), key=[s[1] for s in sel].count))

    # ---- 3. significance ----------------------------------------------------
    from scipy import stats
    t1 = stats.ttest_1samp(R, 0.0)
    # envelope-shuffle permutation: project every trial through the pooled model,
    # build the N x N canonical-correlation matrix ONCE (matched = diagonal), then
    # permuting the EEG<->envelope pairing is just re-indexing it -> a cheap,
    # exact null for "does the model track THIS trial's speech above random pairing".
    full_pat, full_model = _fit_pattern(Xlist, Ylist, eeg_lags, env_lags, k,
                                        npca_star, sh_star, n_ch, n_lags)
    U = [np.asarray((X - full_model["mx"]) @ full_model["Ax"]) for X in Xlist]
    V = [np.asarray((Y - full_model["my"]) @ full_model["Ay"]) for Y in Ylist]
    kk = U[0].shape[1]
    Cmat = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            cs = [np.corrcoef(U[i][:, c], V[j][:, c])[0, 1] for c in range(kk)]
            Cmat[i, j] = np.mean([c if np.isfinite(c) else 0.0 for c in cs])
    obs = float(np.mean(np.diag(Cmat)))
    prng, ar = np.random.RandomState(7), np.arange(N)
    perm = np.array([Cmat[ar, prng.permutation(N)].mean() for _ in range(args.perms)])
    z_perm = float((obs - perm.mean()) / (perm.std() + 1e-12))
    p_perm = float((perm >= obs).mean())
    aud, vis = R[LAB == "Audio"], R[LAB == "Visual"]

    print(f"  engagement (null-corrected): mean {R.mean():+.4f}  SEM {R.std(ddof=1)/np.sqrt(N):.4f}")
    print(f"  matched {M.mean():+.4f} vs mismatched-null {Z.mean():+.4f}")
    print(f"  significant > 0 : t={t1.statistic:.1f} p={t1.pvalue:.2g} | "
          f"perm z={z_perm:.1f} p={p_perm:.4f}")
    print(f"  attend-Audio {aud.mean():+.4f} vs attend-Visual {vis.mean():+.4f} "
          f"(n {len(aud)}/{len(vis)})")

    # ---- 4. spatiotemporal pattern (Haufe) ---------------------------------
    full_pat = _sign_fix(full_pat)                             # (n_lags, n_ch)
    gfp = np.sqrt((full_pat ** 2).mean(axis=1))                # TRF time-course
    peak_lag = int(np.argmax(gfp))
    lag_ms = np.array(eeg_lags) / fs * 1e3
    print(f"  canonical corrs: {np.round(full_model['corrs'], 3)} | "
          f"n_pca* {npca_star} (kept {full_model['n_pca_kept']}), shrink* {sh_star}")
    print(f"  pattern TRF peak at {lag_ms[peak_lag]:.0f} ms")

    # ---- 5. bootstrap reliability ------------------------------------------
    print(f"bootstrap pattern reliability (B={args.bootstrap})...")
    brng = np.random.RandomState(11)
    boot = np.empty((args.bootstrap, n_lags, n_ch), np.float64)
    rel = np.empty(args.bootstrap)
    fv = full_pat.ravel()
    for b in range(args.bootstrap):
        idx = brng.choice(N, N, replace=True)
        Pb, _ = _fit_pattern([Xlist[j] for j in idx], [Ylist[j] for j in idx],
                             eeg_lags, env_lags, k, npca_star, sh_star, n_ch, n_lags)
        if np.corrcoef(Pb.ravel(), fv)[0, 1] < 0:             # sign-align to full
            Pb = -Pb
        boot[b] = Pb
        rel[b] = np.corrcoef(Pb.ravel(), fv)[0, 1]
    boot_mean, boot_std = boot.mean(0), boot.std(0)
    zmap = boot_mean / (boot_std + 1e-12)                      # element reliability
    reliability = float(rel.mean())
    print(f"  pattern reliability (mean bootstrap r vs full): {reliability:.3f}")

    # ---- 6. save method + patterns FIRST (the core deliverable) -------------
    import mne
    method = {
        "analysis": "auditory engagement = neural speech-envelope tracking (rCCA)",
        "subject": subject, "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__, "mne": mne.__version__},
        "dataset": {k2: cfg["dataset"][k2] for k2 in cfg["dataset"]},
        "preprocessing": {"bandpass_hz": [pc["l_freq"], pc["h_freq"]],
                          "resample_hz": fs, "montage": prov["montage_source"],
                          "reference": prov["reference"],
                          "session_bads_interpolated": prov["session_bads_interpolated"],
                          "block_seconds": cfg["block"]["seconds"]},
        "model": {"method": "regularized spatio-temporal CCA (de Cheveigne rCCA)",
                  "eeg_lags_s": [L / fs for L in eeg_lags],
                  "env_lags_s": [L / fs for L in env_lags], "k_components": k,
                  "n_pca_grid": n_pca_grid, "shrink_grid": shrink_grid,
                  "cv_folds": folds, "selected_n_pca": npca_star,
                  "selected_shrink": sh_star, "n_pca_kept": full_model["n_pca_kept"]},
        "evaluation": {"scheme": "nested leave-one-trial-out",
                       "index": "null-corrected top-K canonical correlation",
                       "null": f"mismatched envelope ({n_null} other trials)",
                       "permutations": args.perms, "bootstraps": args.bootstrap},
        "leakage_controls": [
            "all bad-channel interpolation + average reference done ONCE at the "
            "continuous level -> constant Cxx geometry across trials",
            "hyper-parameters tuned by inner CV on training trials only",
            "covariance / whitening / means estimated on training trials only",
            "test trial scored alone; no cross-trial normalisation",
            "mismatched-envelope null removes generic slow-signal correlation"],
    }
    results = {
        "engagement_mean": float(R.mean()),
        "engagement_sem": float(R.std(ddof=1) / np.sqrt(N)),
        "matched_mean": float(M.mean()), "null_mean": float(Z.mean()),
        "t_gt0": float(t1.statistic), "p_gt0": float(t1.pvalue),
        "perm_z": z_perm, "perm_p": p_perm,
        "attend_audio_mean": float(aud.mean()), "attend_visual_mean": float(vis.mean()),
        "canonical_corrs": [float(c) for c in full_model["corrs"]],
        "trf_peak_ms": float(lag_ms[peak_lag]),
        "pattern_reliability": reliability, "n_trials": N,
        "trials": rows,
    }
    with open(os.path.join(outdir, f"{subject}_method.json"), "w") as f:
        json.dump(method, f, indent=2)
    with open(os.path.join(outdir, f"{subject}_engagement.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        os.path.join(outdir, f"{subject}_patterns.npz"),
        pattern=full_pat, boot_mean=boot_mean, boot_std=boot_std, zmap=zmap,
        trf_gfp=gfp, lag_ms=lag_ms, ch_names=np.array(ch_names, object),
        canonical_corrs=np.asarray(full_model["corrs"]),
        engagement=R, matched=M, null=Z, labels=LAB)

    # ---- 7. figures (data already saved; never lose the computation) --------
    try:
        _fig_spatiotemporal(full_pat, lag_ms, info, peak_lag, figdir)
        _fig_trf(gfp, lag_ms, peak_lag, figdir)
        _fig_engagement(rows, M, Z, R, LAB, figdir)
        _fig_reliability(zmap, lag_ms, info, peak_lag, rel, figdir)
    except Exception as e:
        import traceback
        print(f"  !! figure generation failed ({type(e).__name__}: {e})")
        traceback.print_exc()

    print("=" * 74)
    print(f"saved -> {outdir}")
    for fn in sorted(os.listdir(outdir)):
        print("   ", fn)
    print("=" * 74)


# ==========================================================================
# figures
# ==========================================================================
def _topo_lags(pattern, lag_ms, info, lags_ms_show):
    import mne
    idxs = [int(np.argmin(np.abs(lag_ms - L))) for L in lags_ms_show]
    vmax = np.percentile(np.abs(pattern), 99)
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.0 * len(idxs), 2.4))
    for ax, li in zip(axes, idxs):
        mne.viz.plot_topomap(pattern[li], info, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
        ax.set_title(f"{lag_ms[li]:.0f} ms", fontsize=9)
    return fig, axes, vmax


def _fig_spatiotemporal(pattern, lag_ms, info, peak_lag, figdir):
    show = [0, 50, 100, 150, 200, 250, 300]
    fig, axes, vmax = _topo_lags(pattern, lag_ms, info, show)
    fig.suptitle("Auditory-engagement spatiotemporal pattern "
                 "(rCCA Haufe forward map)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    cax = fig.add_axes([0.25, 0.02, 0.5, 0.03])
    import matplotlib as mpl
    fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(-vmax, vmax),
                 cmap="RdBu_r"), cax=cax, orientation="horizontal", label="a.u.")
    fig.savefig(os.path.join(figdir, "spatiotemporal_pattern.png"), dpi=150)
    plt.close(fig)
    # channels x lags heatmap
    fig, ax = plt.subplots(figsize=(7, 4))
    v = np.percentile(np.abs(pattern), 99)
    im = ax.imshow(pattern.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                   extent=[lag_ms[0], lag_ms[-1], pattern.shape[1], 0])
    ax.set_xlabel("lag (ms)"); ax.set_ylabel("channel index")
    ax.set_title("Spatiotemporal pattern (channels x lags)")
    fig.colorbar(im, ax=ax, label="a.u.")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "pattern_heatmap.png"), dpi=150)
    plt.close(fig)


def _fig_trf(gfp, lag_ms, peak_lag, figdir):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(lag_ms, gfp, lw=2, color="#204060")
    ax.axvline(lag_ms[peak_lag], ls="--", color="crimson",
               label=f"peak {lag_ms[peak_lag]:.0f} ms")
    ax.set_xlabel("lag (ms)"); ax.set_ylabel("pattern GFP")
    ax.set_title("Temporal response function of auditory engagement")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "trf_timecourse.png"), dpi=150)
    plt.close(fig)


def _fig_engagement(rows, M, Z, R, LAB, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(M, bins=15, alpha=0.7, label="matched", color="#2a7")
    axes[0].hist(Z, bins=15, alpha=0.7, label="mismatched null", color="#999")
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_xlabel("canonical correlation"); axes[0].set_ylabel("trials")
    axes[0].set_title("matched vs null"); axes[0].legend()
    a, v = R[LAB == "Audio"], R[LAB == "Visual"]
    axes[1].boxplot([a, v], tick_labels=["attend\nAudio", "attend\nVisual"])
    axes[1].axhline(0, color="k", lw=0.6, ls=":")
    axes[1].set_ylabel("engagement (null-corrected)")
    axes[1].set_title("engagement by attended stream")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "engagement.png"), dpi=150)
    plt.close(fig)


def _fig_reliability(zmap, lag_ms, info, peak_lag, rel, figdir):
    import mne
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    vmax = np.percentile(np.abs(zmap), 99)
    mne.viz.plot_topomap(zmap[peak_lag], info, axes=axes[0], show=False,
                         cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
    axes[0].set_title(f"reliability z-map @ {lag_ms[peak_lag]:.0f} ms")
    axes[1].hist(rel, bins=20, color="#468")
    axes[1].axvline(rel.mean(), color="crimson", ls="--",
                    label=f"mean r={rel.mean():.2f}")
    axes[1].set_xlabel("bootstrap pattern r vs full"); axes[1].set_ylabel("count")
    axes[1].set_title("pattern reliability"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "reliability.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
