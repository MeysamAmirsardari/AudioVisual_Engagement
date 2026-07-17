#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cca_auditory_decoding.py - reproduce de Cheveigne et al. (2018, NeuroImage) Figs 2-6
for our cross-modal dataset, comparing ATTEND-AUDIO vs ATTEND-VISUAL trials and two
acoustic descriptors (amplitude ENVELOPE and F0 pitch contour).

Our single-subject adaptation of the paper: where the paper plots one line per
SUBJECT (Fig 5), we plot one line per ATTENTION CONDITION (attend-Audio vs
attend-Visual) - the contrast of interest. Every model is evaluated exactly as in
the paper: cross-validated correlation as a function of the global stimulus-EEG
shift L.

Figures (-> records/derivatives/cca_decoding/<subject>/figures/):
  fig2_backward.png      backward model: corr vs shift (+train) + scalp topography
  fig3_forward.png       forward model: corr vs shift + impulse response/transfer fn
  fig4_cca.png           CCA models 1/2/3: canonical corr of every CC pair vs shift
  fig5_models.png        best score per model, attend-Audio vs attend-Visual (env,F0)
  fig6_components_*.png   best CCA's first 12 CCs: transfer functions + topographies

Usage:
    python stim_decoding/cca_auditory_decoding.py
    python stim_decoding/cca_auditory_decoding.py --folds 5 --k 6
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
from sda import io, preprocess, stimuli, cca_paper as CP  # noqa: E402

COND = {"Audio": "#c0392b", "Visual": "#2471a3"}
MODELS5 = ["backward", "forward", "CCA1", "CCA2", "CCA3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--k", type=int, default=6, help="CC pairs to track")
    ap.add_argument("--shift-lo", type=float, default=-0.75)
    ap.add_argument("--shift-hi", type=float, default=1.25)
    ap.add_argument("--shift-step", type=float, default=0.125)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    pc, m = cfg["preprocess"], cfg["model"]
    fs = float(pc["resample_hz"])
    band = (pc["l_freq"], pc["h_freq"])
    k = args.k
    eeg_lags = list(range(0, int(round(0.25 * fs)) + 1))
    feat_lags = list(range(0, int(round(0.25 * fs)) + 1))
    shifts = list(np.round(np.arange(args.shift_lo, args.shift_hi + 1e-9,
                                     args.shift_step), 4))
    subject = cfg["dataset"]["subject"]
    outdir = args.out or os.path.join(ROOT, "records", "derivatives",
                                      "cca_decoding", subject)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 74)
    print("CCA AUDITORY DECODING (de Cheveigne et al. 2018) - Audio vs Visual, env vs F0")
    print("=" * 74)

    # ---- load + clean + descriptors ----------------------------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    EEG, ENV, F0, LAB = [], [], [], []
    for _, row in events.iterrows():
        t = int(row["trial"])
        wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
        if t not in trials or not os.path.exists(wav):
            continue
        e = trials[t]["eeg"].T
        EEG.append(e)
        ENV.append(stimuli.audio_envelope(wav, fs, e.shape[0], cfg, band))
        F0.append(stimuli.audio_f0_contour(wav, fs, e.shape[0], cfg, band))
        LAB.append(str(row["label"]))
    LAB = np.array(LAB)
    N = len(EEG)
    Xlag = [CP.lag(e, eeg_lags).astype(np.float32) for e in EEG]
    Xspa = [e.astype(np.float32) for e in EEG]
    feats = {"envelope": ENV, "F0": F0}
    conds = {c: [i for i in range(N) if LAB[i] == c] for c in ("Audio", "Visual")}
    print(f"{N} trials | Audio {len(conds['Audio'])} Visual {len(conds['Visual'])} | "
          f"shifts {shifts[0]:.2f}..{shifts[-1]:.2f}s | EEG/feat lags 0-250 ms | K={k}")

    # ---- EEG fold caches (feature independent -> once per condition) --------
    cache = {c: CP.eeg_fold_cache([Xlag[i] for i in idx], [Xspa[i] for i in idx],
                                  folds=args.folds, seed=0)
             for c, idx in conds.items()}

    # ---- run every model for each descriptor x condition -------------------
    res = {}
    for fname, F in feats.items():
        res[fname] = {}
        for c, idx in conds.items():
            print(f"  [{fname:8s} | attend-{c}] backward/forward/CCA1-3 ...")
            Xl = [Xlag[i] for i in idx]; Xs = [Xspa[i] for i in idx]
            Fc = [F[i] for i in idx]; Ec = [EEG[i] for i in idx]
            ca = cache[c]
            tr, te = CP.backward_curve(ca, Xl, Fc, shifts, fs)
            b_best = int(np.argmax(te))
            topo = CP.backward_topography(Ec, Fc, shifts[b_best], fs)
            f_test, f_ch, f_irf, f_tf, f_fr = CP.forward_curve(ca, Xs, Fc, feat_lags,
                                                               shifts, fs)
            cca = {mdl: CP.cca_curve(ca, Xl, Xs, Fc, feat_lags, shifts, fs,
                                     k=k, model=mdl) for mdl in (1, 2, 3)}
            res[fname][c] = {
                "backward_train": tr.tolist(), "backward_test": te.tolist(),
                "backward_best_shift": shifts[b_best], "topo": topo.tolist(),
                "forward_test": f_test.tolist(), "forward_best_ch": ch_names[f_ch],
                "forward_irf": f_irf.tolist(), "forward_tf": f_tf.tolist(),
                "forward_freqs": f_fr.tolist(),
                "cca1": cca[1].tolist(), "cca2": cca[2].tolist(),
                "cca3": cca[3].tolist(),
                "best": {"backward": float(te.max()), "forward": float(f_test.max()),
                         "CCA1": float(cca[1][:, 0].max()),
                         "CCA2": float(cca[2][:, 0].max()),
                         "CCA3": float(cca[3][:, 0].max())}}

    # ---- figures ------------------------------------------------------------
    _fig2(res, shifts, info, figdir)
    _fig3(res, shifts, figdir)
    _fig4(res, shifts, k, figdir)
    _fig5(res, figdir)
    for fname in feats:                                    # Fig 6 per descriptor
        best_shift = res[fname]["Audio"]["backward_best_shift"]
        idx = conds["Audio"]
        comp = CP.cca_components_model3([EEG[i] for i in idx],
                                        [feats[fname][i] for i in idx],
                                        best_shift, fs, k=12, n_bands=20)
        _fig6(comp, info, fname, figdir)

    # ---- save summary -------------------------------------------------------
    summary = {"created": _dt.datetime.now().isoformat(timespec="seconds"),
               "paper": "de Cheveigne et al. 2018 NeuroImage 172:206-216",
               "subject": subject, "n_trials": N, "shifts_s": shifts,
               "eeg_lags_s": [L / fs for L in eeg_lags],
               "feat_lags_s": [L / fs for L in feat_lags], "k": k,
               "best_scores": {f: {c: res[f][c]["best"] for c in conds} for f in feats},
               "backward_best_shift_s": {f: {c: res[f][c]["backward_best_shift"]
                                             for c in conds} for f in feats},
               "forward_best_channel": {f: {c: res[f][c]["forward_best_ch"]
                                            for c in conds} for f in feats}}
    with open(os.path.join(outdir, f"{subject}_cca_decoding.json"), "w") as fjs:
        json.dump(summary, fjs, indent=2)

    print("=" * 74)
    print(f"saved -> {outdir}")
    for fn in sorted(os.listdir(figdir)):
        print("    figures/", fn)
    _print_scores(res, feats, conds)
    print("=" * 74)


# ==========================================================================
def _fig2(res, shifts, info, figdir):
    import mne
    feats = list(res); x = shifts
    fig = plt.figure(figsize=(13, 3.4 * len(feats)))
    for r, fn in enumerate(feats):
        axc = fig.add_subplot(len(feats), 3, 3 * r + 1)
        for c, col in COND.items():
            axc.plot(x, res[fn][c]["backward_test"], color=col, lw=2, label=f"{c} test")
            axc.plot(x, res[fn][c]["backward_train"], color=col, lw=1, ls=":",
                     alpha=0.7)
        axc.axvline(0, color="k", lw=0.5); axc.axhline(0, color="k", lw=0.5)
        axc.set_xlabel("shift (s)"); axc.set_ylabel("correlation")
        axc.set_title(f"backward: {fn}\n(dotted=train, solid=test)")
        axc.legend(fontsize=8)
        for j, c in enumerate(COND):
            ax = fig.add_subplot(len(feats), 3, 3 * r + 2 + j)
            topo = np.array(res[fn][c]["topo"])
            vmax = np.percentile(np.abs(topo), 98)
            mne.viz.plot_topomap(topo, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vmax, vmax), contours=4)
            ax.set_title(f"{fn}, attend-{c}\nEEG-stimulus corr @ "
                         f"{res[fn][c]['backward_best_shift']:+.2f}s", fontsize=9)
    fig.suptitle("Fig 2. Backward model (stimulus reconstruction)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(figdir, "fig2_backward.png"), dpi=150); plt.close(fig)


def _fig3(res, shifts, figdir):
    feats = list(res); x = shifts
    fig, axes = plt.subplots(len(feats), 3, figsize=(13, 3.4 * len(feats)))
    axes = np.atleast_2d(axes)
    for r, fn in enumerate(feats):
        for c, col in COND.items():
            axes[r, 0].plot(x, res[fn][c]["forward_test"], color=col, lw=2, label=c)
        axes[r, 0].axvline(0, color="k", lw=0.5); axes[r, 0].axhline(0, color="k", lw=0.5)
        axes[r, 0].set_xlabel("shift (s)"); axes[r, 0].set_ylabel("correlation")
        axes[r, 0].set_title(f"forward: {fn} (best channel)"); axes[r, 0].legend(fontsize=8)
        lags_ms = np.arange(len(res[fn][next(iter(COND))]["forward_irf"])) / 64.0 * 1e3
        for c, col in COND.items():
            axes[r, 1].plot(lags_ms, res[fn][c]["forward_irf"], color=col, lw=1.6,
                            label=f"{c} ({res[fn][c]['forward_best_ch']})")
            axes[r, 2].plot(res[fn][c]["forward_freqs"], res[fn][c]["forward_tf"],
                            color=col, lw=1.6)
        axes[r, 1].set_xlabel("lag (ms)"); axes[r, 1].set_ylabel("weight")
        axes[r, 1].set_title(f"{fn} impulse response"); axes[r, 1].legend(fontsize=7)
        axes[r, 2].set_xlabel("frequency (Hz)"); axes[r, 2].set_ylabel("|H(f)|")
        axes[r, 2].set_xlim(0, 15); axes[r, 2].set_title(f"{fn} transfer function")
    fig.suptitle("Fig 3. Forward model (encoding)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(figdir, "fig3_forward.png"), dpi=150); plt.close(fig)


def _fig4(res, shifts, k, figdir):
    feats = list(res); x = shifts
    fig, axes = plt.subplots(len(feats), 3, figsize=(13, 3.4 * len(feats)))
    axes = np.atleast_2d(axes)
    for r, fn in enumerate(feats):
        for col_i, mdl in enumerate(("cca1", "cca2", "cca3")):
            ax = axes[r, col_i]
            for c, col in COND.items():
                cc = np.array(res[fn][c][mdl])              # (n_shift, k)
                ax.plot(x, cc[:, 0], color=col, lw=2.2, label=f"{c} CC1")
                for j in range(1, cc.shape[1]):
                    ax.plot(x, cc[:, j], color=col, lw=0.7, alpha=0.35)
            ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("shift (s)")
            if col_i == 0:
                ax.set_ylabel(f"{fn}\ncanonical corr")
            ax.set_title(f"CCA model {col_i + 1}")
            if r == 0 and col_i == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Fig 4. CCA models (bold=CC1, thin=CC2..K)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(figdir, "fig4_cca.png"), dpi=150); plt.close(fig)


def _fig5(res, figdir):
    feats = list(res)
    fig, axes = plt.subplots(1, len(feats), figsize=(6.2 * len(feats), 4.2))
    axes = np.atleast_1d(axes)
    xi = np.arange(len(MODELS5))
    for a, fn in enumerate(feats):
        for c, col in COND.items():
            ys = [res[fn][c]["best"][mdl] for mdl in MODELS5]
            axes[a].plot(xi, ys, "-o", color=col, lw=2, ms=7, label=f"attend-{c}")
        axes[a].set_xticks(xi); axes[a].set_xticklabels(MODELS5, rotation=20)
        axes[a].set_ylabel("best correlation"); axes[a].set_title(f"{fn} decoding")
        axes[a].grid(alpha=0.3); axes[a].legend()
    fig.suptitle("Fig 5. Best score per model", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(figdir, "fig5_models.png"), dpi=150); plt.close(fig)


def _fig6(comp, info, fname, figdir):
    # Laid out exactly like the paper (de Cheveigne 2018, Fig 6): 4 columns,
    # top = normalised amplitude transfer functions of the CCA-derived FIR filters
    # (log frequency), bottom = EEG topographies with an INDIVIDUAL colour scale per
    # component (jet), reflecting the lower SNR of higher-order CCs.
    import mne
    kk = comp["tf"].shape[0]
    ncol = 4
    nrow = int(np.ceil(kk / ncol))
    fig = plt.figure(figsize=(11, 2.05 * nrow + 0.6))
    for c in range(kk):                                    # top: transfer functions
        ax = fig.add_subplot(2 * nrow, ncol, c + 1)
        ax.semilogx(comp["freqs"], comp["tf"][c] / (comp["tf"][c].max() + 1e-12),
                    color="#1f6f8b", lw=1.6)
        ax.set_xlim(0.4, 12); ax.set_ylim(0, 1.05)
        ax.set_title(f"CC {c+1}", fontsize=8)
        ax.set_xticks([1, 10]); ax.set_xticklabels(["1", "10"])
        ax.set_yticks([0, 1]); ax.tick_params(labelsize=6)
        if c % ncol == 0:
            ax.set_ylabel("normalized\namplitude", fontsize=7)
        if c >= kk - ncol:
            ax.set_xlabel("frequency (Hz)", fontsize=7)
    for c in range(kk):                                    # bottom: topographies
        ax = fig.add_subplot(2 * nrow, ncol, nrow * ncol + c + 1)
        t = comp["topo"][c]; vmax = np.percentile(np.abs(t), 97) + 1e-9
        mne.viz.plot_topomap(t, info, axes=ax, show=False, cmap="RdBu_r",
                             vlim=(-vmax, vmax), contours=4)
        ax.set_title(f"CC {c+1}   r={comp['corrs'][c]:.2f}", fontsize=8)
    fig.suptitle(f"Fig 6. CCA model 3 components ({fname}): "
                 f"transfer functions (top) + topographies (bottom)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(figdir, f"fig6_components_{fname}.png"), dpi=150)
    plt.close(fig)


def _print_scores(res, feats, conds):
    print("best correlation (max over shift):")
    print(f"    {'feature':9s} {'cond':7s} " + " ".join(f"{m:>9s}" for m in MODELS5))
    for fn in feats:
        for c in conds:
            b = res[fn][c]["best"]
            print(f"    {fn:9s} {c:7s} " + " ".join(f"{b[m]:9.3f}" for m in MODELS5))


if __name__ == "__main__":
    main()
