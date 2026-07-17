#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cca_decoding.py — de Cheveigné-style stimulus-response decoding for the FOUR speech-
hierarchy features (envelope, word onset, lexical surprise, contextual surprise),
attend-Audio (focused) vs attend-Visual (distracted), using the project's cca_models.py.

Figures (av_distraction/figures/):
  cca_backward.png   backward reconstruction — corr vs shift (train dotted / test solid)
                     + EEG↔stimulus correlation topography, per feature × condition
  cca_models.png     CCA models 1/2/3 — canonical corr vs shift (CC1 bold, CC2..K thin)
  cca_best.png       best test score per model {backward, forward, CCA1, CCA2, CCA3},
                     attend-Audio vs attend-Visual
  cca_decoding_topo.png  clean per-feature decoding (Audio vs Visual, best model) + topography

Reuses the cleaned data saved by pipeline.py (preprocessed/) and rebuilds the 4 features.
Cross-validation scores the POOLED held-out trials per fold (the de Cheveigné convention).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)                                    # cca_models.py (project root)
sys.path.insert(0, os.path.join(ROOT, "stim_decoding"))
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
import cca_models as M  # noqa: E402  (the model the user added)
from sda import io, hierarchy  # noqa: E402
import mne  # noqa: E402
mne.set_log_level("ERROR")

FEATURES = ["envelope", "word_onset", "word_frequency", "gpt2_surprisal"]
PRETTY = {"envelope": "Envelope", "word_onset": "Word onset",
          "word_frequency": "Lexical surprise", "gpt2_surprisal": "Contextual surprise"}
CA, CV = "#c0392b", "#2471a3"                               # attend-Audio / attend-Visual
FIG = os.path.join(HERE, "figures"); PRE = os.path.join(HERE, "preprocessed")
MODELS5 = ["backward", "forward", "cca1", "cca2", "cca3"]
KTHIN = 6                                                   # CC components to draw


# ------------------------------------------------------------------ helpers
def _shift(f, L):
    if L == 0:
        return f
    out = np.zeros_like(f)
    if L > 0:
        out[L:] = f[:-L]
    else:
        out[:L] = f[-L:]
    return out


def _folds(n, k=4, seed=0):
    idx = np.arange(n); np.random.RandomState(seed).shuffle(idx)
    return [(np.setdiff1d(idx, f), np.sort(f)) for f in np.array_split(idx, k)]


def _red(name):
    if name == "forward":
        return lambda r: float(np.nanmax(r)) if r.size else 0.0        # best EEG channel
    return lambda r: float(r[0]) if r.size else 0.0                    # backward / CC1


def curve(name, EEG, FEAT, shifts, folds, want_train=False):
    """Pooled CV correlation vs shift (scalar). Returns (train, test) arrays."""
    red = _red(name); tr = np.full(len(shifts), np.nan); te = np.zeros(len(shifts))
    for j, L in enumerate(shifts):
        F = [_shift(f, L) for f in FEAT]; a_te, a_tr = [], []
        for a, b in folds:
            m = M.model(name).fit([EEG[i] for i in a], [F[i] for i in a])
            a_te.append(red(np.asarray(m.score([EEG[i] for i in b], [F[i] for i in b]), float)))
            if want_train:
                a_tr.append(red(np.asarray(m.score([EEG[i] for i in a], [F[i] for i in a]), float)))
        te[j] = np.mean(a_te)
        if want_train:
            tr[j] = np.mean(a_tr)
    return tr, te


def cca_curve(name, EEG, FEAT, shifts, folds, K):
    """Pooled CV canonical correlations per component vs shift -> (n_shifts, K)."""
    out = np.zeros((len(shifts), K))
    for j, L in enumerate(shifts):
        F = [_shift(f, L) for f in FEAT]; acc = []
        for a, b in folds:
            m = M.model(name).fit([EEG[i] for i in a], [F[i] for i in a])
            r = np.asarray(m.score([EEG[i] for i in b], [F[i] for i in b]), float)
            acc.append(np.pad(r[:K], (0, max(0, K - r.size))))
        out[j] = np.nanmean(acc, 0)
    return out


def topo_corr(EEG, FEAT, L, n_ch):
    """Per-channel EEG↔stimulus correlation at shift L (averaged over trials)."""
    F = [_shift(f, L) for f in FEAT]; cc = np.zeros(n_ch)
    for e, f in zip(EEG, F):
        s = f[:, 0] - f[:, 0].mean(); Ec = e - e.mean(0)
        d = np.linalg.norm(Ec, axis=0) * np.linalg.norm(s)
        cc += np.divide((Ec * s[:, None]).sum(0), d, out=np.zeros(n_ch), where=d > 0)
    return cc / len(EEG)


# ------------------------------------------------------------------ data
def load():
    d = np.load(os.path.join(PRE, "sub001_Kavin_trials.npz"), allow_pickle=True)
    EEG = [x.T.astype(np.float64) for x in d["eeg"]]        # list of (n_times, n_ch)
    lab = d["labels"]; order = list(d["trials"])
    info = mne.io.read_raw_fif(os.path.join(PRE, "sub001_Kavin_cleaned_raw.fif")).pick("eeg").info
    cfg = io.load_config(os.path.join(ROOT, "stim_decoding", "config_stimdec.yaml"))
    fs = float(cfg["preprocess"]["resample_hz"]); band = (cfg["preprocess"]["l_freq"], cfg["preprocess"]["h_freq"])
    n_times = EEG[0].shape[0]; audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    _, events, _ = io.load_raw_and_events(cfg)
    stim = {int(r["trial"]): r["audio_stim"] for _, r in events.iterrows() if r["audio_stim"] is not None}
    gpt2 = hierarchy.load_gpt2()
    FEATS = {f: [] for f in FEATURES}
    for t in order:
        wav = os.path.join(audio_dir, f"{stim[int(t)]}.wav")
        wj = os.path.join(audio_dir, f"{stim[int(t)]}.words.json")
        words = json.load(open(wj))["words"]
        FEATS["envelope"].append(hierarchy.envelope(wav, fs, n_times, cfg, band).astype(np.float64))
        FEATS["word_onset"].append(hierarchy.word_onset(words, fs, n_times).astype(np.float64))
        FEATS["word_frequency"].append(hierarchy.word_frequency(words, fs, n_times).astype(np.float64))
        FEATS["gpt2_surprisal"].append(hierarchy.gpt2_surprisal(words, fs, n_times, *gpt2).astype(np.float64))
    aud = np.array([str(x) for x in lab]) == "Audio"
    print(f"{len(EEG)} trials | focused {int(aud.sum())} distracted {int((~aud).sum())} | {len(info['ch_names'])} ch @ {fs:.0f} Hz")
    return EEG, FEATS, aud, info, fs


# ------------------------------------------------------------------ main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=-0.75); ap.add_argument("--hi", type=float, default=1.25)
    ap.add_argument("--step", type=float, default=0.125); ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    EEG, FEATS, aud, info, fs = load()
    shifts = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 4)
    shift_samp = [int(round(s * fs)) for s in shifts]
    conds = {"Audio": np.where(aud)[0], "Visual": np.where(~aud)[0]}
    print("=" * 70); print("CCA DECODING (cca_models.py) — 4 features × attend-Audio/Visual"); print("=" * 70)

    R = {f: {c: {} for c in conds} for f in FEATURES}
    for f in FEATURES:
        for c, idx in conds.items():
            E = [EEG[i] for i in idx]; folds = _folds(len(E), args.folds)
            Fl = {k: [FEATS[f][i] for i in idx] for k in [f]}[f]
            btr, bte = curve("backward", E, Fl, shift_samp, folds, want_train=True)
            _, fte = curve("forward", E, Fl, shift_samp, folds)
            cc = {m: cca_curve(m, E, Fl, shift_samp, folds, KTHIN) for m in ("cca1", "cca2", "cca3")}
            bshift = int(np.argmax(bte))
            R[f][c] = {"btr": btr, "bte": bte, "fte": fte, "cc": cc, "bshift": bshift,
                       "topo": topo_corr(E, Fl, shift_samp[bshift], len(info["ch_names"])),
                       "best": {"backward": float(bte.max()), "forward": float(fte.max()),
                                "cca1": float(cc["cca1"][:, 0].max()), "cca2": float(cc["cca2"][:, 0].max()),
                                "cca3": float(cc["cca3"][:, 0].max())}}
            print(f"  {f:15s} attend-{c:6s}: backward {bte.max():.3f} @ {shifts[bshift]:+.2f}s | "
                  f"cca2 {cc['cca2'][:,0].max():.3f}")

    fig_backward(R, shifts, info); fig_models(R, shifts); fig_best(R); fig_decoding_topo(R, shifts, EEG, FEATS, aud, info, fs)
    json.dump({f: {c: R[f][c]["best"] for c in conds} for f in FEATURES},
              open(os.path.join(HERE, "cca_decoding.json"), "w"), indent=2)
    print(f"saved 4 CCA figures + cca_decoding.json -> {HERE}")


# ------------------------------------------------------------------ figures
def _topo(ax, vals, info, title):
    v = np.abs(vals).max() or 1
    mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-v, v), contours=4)
    ax.set_title(title, fontsize=8)


def fig_backward(R, shifts, info):
    fig, ax = plt.subplots(len(FEATURES), 3, figsize=(13, 3.1 * len(FEATURES)))
    for i, f in enumerate(FEATURES):
        A, V = R[f]["Audio"], R[f]["Visual"]
        ax[i, 0].plot(shifts, A["btr"], ":", color=CA, lw=1); ax[i, 0].plot(shifts, A["bte"], color=CA, lw=2, label="Audio test")
        ax[i, 0].plot(shifts, V["btr"], ":", color=CV, lw=1); ax[i, 0].plot(shifts, V["bte"], color=CV, lw=2, label="Visual test")
        ax[i, 0].axvline(0, color="k", lw=.5); ax[i, 0].axhline(0, color="k", lw=.5)
        ax[i, 0].set_ylabel("correlation"); ax[i, 0].set_title(f"backward: {PRETTY[f]}\n(dotted=train, solid=test)", fontsize=9)
        if i == 0:
            ax[i, 0].legend(fontsize=8, frameon=False)
        if i == len(FEATURES) - 1:
            ax[i, 0].set_xlabel("shift (s)")
        _topo(ax[i, 1], A["topo"], info, f"{f}, attend-Audio\nEEG↔stim corr @ {shifts[A['bshift']]:+.2f}s")
        _topo(ax[i, 2], V["topo"], info, f"{f}, attend-Visual\nEEG↔stim corr @ {shifts[V['bshift']]:+.2f}s")
    fig.suptitle("Backward model (stimulus reconstruction) — 4 speech-hierarchy features", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(os.path.join(FIG, "cca_backward.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_models(R, shifts):
    mods = ["cca1", "cca2", "cca3"]
    fig, ax = plt.subplots(len(FEATURES), 3, figsize=(13, 3.0 * len(FEATURES)), sharex=True)
    for i, f in enumerate(FEATURES):
        for j, m in enumerate(mods):
            for c, col in (("Audio", CA), ("Visual", CV)):
                cc = R[f][c]["cc"][m]
                for k in range(1, cc.shape[1]):
                    ax[i, j].plot(shifts, cc[:, k], color=col, lw=.6, alpha=.3)
                ax[i, j].plot(shifts, cc[:, 0], color=col, lw=2.2, label=f"{c} CC1" if i == 0 and j == 0 else None)
            ax[i, j].axvline(0, color="k", lw=.5); ax[i, j].axhline(0, color="k", lw=.4)
            if i == 0:
                ax[i, j].set_title(f"CCA model {j+1}", fontsize=10)
            if j == 0:
                ax[i, j].set_ylabel(f"{PRETTY[f]}\ncanonical corr", fontsize=9)
            if i == len(FEATURES) - 1:
                ax[i, j].set_xlabel("shift (s)")
    ax[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("CCA models (bold = CC1, thin = CC2..K) — Audio vs Visual", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(os.path.join(FIG, "cca_models.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_best(R):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, f in enumerate(FEATURES):
        a = ax[i // 2, i % 2]; x = np.arange(len(MODELS5))
        a.plot(x, [R[f]["Audio"]["best"][m] for m in MODELS5], "o-", color=CA, label="attend-Audio (focused)")
        a.plot(x, [R[f]["Visual"]["best"][m] for m in MODELS5], "o-", color=CV, label="attend-Visual (distracted)")
        a.set_xticks(x); a.set_xticklabels(["backward", "forward", "CCA1", "CCA2", "CCA3"], fontsize=9)
        a.set_ylabel("best correlation"); a.set_title(PRETTY[f]); a.grid(alpha=.2)
        if i == 0:
            a.legend(frameon=False, fontsize=9)
    fig.suptitle("Best score per model: attend-Audio vs attend-Visual", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(os.path.join(FIG, "cca_best.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_decoding_topo(R, shifts, EEG, FEATS, aud, info, fs):
    """Clean per-feature decoding (per-trial backward reconstruction, Audio vs Visual) + topography."""
    fig, ax = plt.subplots(len(FEATURES), 2, figsize=(11, 3.0 * len(FEATURES)),
                           gridspec_kw={"width_ratios": [1.4, 1]})
    for i, f in enumerate(FEATURES):
        # per-trial backward score at each condition's best shift (LOO-ish: fit others, score trial)
        pt = {}
        for c, idx in (("Audio", np.where(aud)[0]), ("Visual", np.where(~aud)[0])):
            L = [int(round(shifts[R[f][c]["bshift"]] * fs))][0]
            E = [EEG[j] for j in idx]; Fl = [_shift(FEATS[f][j], L) for j in idx]
            sc = np.zeros(len(E))
            for a, b in _folds(len(E), 5):
                m = M.model("backward").fit([E[j] for j in a], [Fl[j] for j in a])
                for j in b:
                    r = np.asarray(m.score([E[j]], [Fl[j]]), float); sc[j] = r[0] if r.size else 0
            pt[c] = sc
        bp = ax[i, 0].boxplot([pt["Audio"], pt["Visual"]], widths=.55, patch_artist=True, showfliers=False)
        for patch, cc in zip(bp["boxes"], (CA, CV)):
            patch.set_facecolor(cc); patch.set_alpha(.25)
        rng = np.random.RandomState(0)
        ax[i, 0].scatter(rng.normal(1, .05, len(pt["Audio"])), pt["Audio"], color=CA, s=16, zorder=3)
        ax[i, 0].scatter(rng.normal(2, .05, len(pt["Visual"])), pt["Visual"], color=CV, s=16, zorder=3)
        ax[i, 0].set_xticks([1, 2]); ax[i, 0].set_xticklabels(["focused\n(Audio)", "distracted\n(Visual)"])
        ax[i, 0].set_ylabel(f"{PRETTY[f]}\nreconstruction r")
        for s in ("top", "right"):
            ax[i, 0].spines[s].set_visible(False)
        _topo(ax[i, 1], R[f]["Audio"]["topo"], info, f"{PRETTY[f]} — EEG↔stim topography")
    fig.suptitle("Per-trial decoding by feature (focused vs distracted) + topography", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(os.path.join(FIG, "cca_decoding_topo.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
