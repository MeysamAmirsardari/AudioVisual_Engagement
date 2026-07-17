#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — does VISUAL DISTRACTION hit the HIGHER levels of the speech hierarchy
more than the lower levels?  (Kavin = sub001, cross-modal attention session 1.)

Attend-Audio = listener focused on speech; attend-Visual (Tetris) = visual distraction.
For a hierarchy of speech representations from acoustic to linguistic we measure how
well the EEG tracks each level in each condition; the FOCUS-minus-DISTRACTION difference
(attend-Audio − attend-Visual) is the distraction effect, and the hypothesis is that it
GROWS up the hierarchy.

  envelope → spectrogram → word onset → lexical surprise → GPT-2 contextual surprise
  (acoustic ─────────────────────────────────────────────→ linguistic)

Clean, self-contained: preprocesses session-1 Kavin with the CORRECTED montage (cable
half-swap), SAVES the preprocessed data, then reuses our hierarchical-tracking code
(aud_cca regularised CCA) to score each level, and writes a plot for EVERY level plus
the comparative trend + a report.

Run:  python av_distraction/pipeline.py   [--perms 2000 --boots 2000 --no-gpt2]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SD = os.path.join(ROOT, "stim_decoding")
sys.path.insert(0, SD)
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess, hierarchy  # noqa: E402
from hierarchical_tracking import feature_tracking  # noqa: E402  (aud_cca CCA, reused)

PRETTY = {"envelope": "Envelope", "spectrogram": "Spectrogram", "word_onset": "Word onset",
          "word_frequency": "Lexical surprise", "gpt2_surprisal": "Contextual surprise (GPT-2)"}
KCOL = {"acoustic": "#3b6ea5", "lexical": "#c46a1b", "linguistic": "#8e2f8e"}
CA, CV = "#c0392b", "#2471a3"                            # attend-Audio / attend-Visual colours
FIG = os.path.join(HERE, "figures"); PRE = os.path.join(HERE, "preprocessed")


# ---------------------------------------------------------------- preprocessing
def preprocess_and_save(cfg):
    import mne
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    os.makedirs(PRE, exist_ok=True)
    raw_model.save(os.path.join(PRE, "sub001_Kavin_cleaned_raw.fif"), overwrite=True)
    order = sorted(trials)
    EEG = np.stack([trials[t]["eeg"] for t in order])            # (n_trials, n_ch, n_times)
    lab = np.array([trials[t]["label"] for t in order])
    np.savez_compressed(os.path.join(PRE, "sub001_Kavin_trials.npz"),
                        eeg=EEG, labels=lab, ch_names=ch_names, trials=order)
    # preprocessing overview figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    raw_model.plot_sensors(show_names=True, axes=ax[0], show=False)
    ax[0].set_title(f"corrected montage ({len(ch_names)} ch, cable-swap fixed)")
    psd = raw_model.compute_psd(fmin=1, fmax=raw_model.info["sfreq"] / 2 - 1)
    ax[1].plot(psd.freqs, 10 * np.log10(psd.get_data().mean(0)), color="#356")
    ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB"); ax[1].set_title("cleaned PSD")
    mne.viz.plot_topomap(np.log(np.var(raw_model.get_data(), 1) + 1e-30), raw_model.info,
                         axes=ax[2], show=False, cmap="magma", contours=3)
    ax[2].set_title("log channel variance")
    fig.suptitle("Preprocessing: session-1 Kavin (sub001); 1-8 Hz, 64 Hz, ICA, avg-ref, "
                 "corrected channels", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(FIG, "00_preprocessing.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    return trials, events, ch_names, prov


# ---------------------------------------------------------------- per-level plot
def plot_level(i, level, feat0, track_l, aud, kind, st, fs):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    t = np.arange(feat0.shape[0]) / fs
    if level == "spectrogram":
        ax[0].imshow(feat0.T, aspect="auto", origin="lower", cmap="magma",
                     extent=[0, t[-1], 0, feat0.shape[1]])
        ax[0].set_ylabel("mel band")
    elif level in ("word_onset", "word_frequency", "gpt2_surprisal"):
        v = feat0[:, 0]; idx = np.nonzero(v)[0]
        ax[0].vlines(t[idx], 0, v[idx], color=KCOL[kind], lw=1.4)
        ax[0].axhline(0, color="k", lw=.5)
    else:
        ax[0].plot(t, feat0[:, 0], color=KCOL[kind], lw=1)
    ax[0].set_xlabel("time (s)"); ax[0].set_title(f"{PRETTY[level]}: stimulus feature (trial 1)")
    # per-trial tracking, focus vs distraction
    a, v = track_l[aud], track_l[~aud]
    bp = ax[1].boxplot([a, v], widths=.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], (CA, CV)):
        patch.set_facecolor(c); patch.set_alpha(.25)
    rng = np.random.RandomState(0)
    ax[1].scatter(rng.normal(1, .06, len(a)), a, color=CA, s=22, zorder=3)
    ax[1].scatter(rng.normal(2, .06, len(v)), v, color=CV, s=22, zorder=3)
    ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(["attend Audio\n(focused)", "attend Visual\n(distracted)"])
    ax[1].set_ylabel("EEG→speech tracking (top-k canonical r)")
    ax[1].set_title(f"distraction effect  d = {st['d']:+.2f}   (p = {st['p']:.3f})")
    for s in ("top", "right"):
        ax[1].spines[s].set_visible(False)
    fig.suptitle(f"LEVEL {i}: {PRETTY[level]}  ({kind})", fontsize=13, color=KCOL[kind])
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(os.path.join(FIG, f"{i:02d}_{level}.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- comparative
def plot_compare(levels, track, aud, d, d_ci, perm_p, rho, rho_p, kinds):
    lev = np.arange(len(levels))
    # (1) tracking Audio vs Visual across levels
    fig, ax = plt.subplots(figsize=(9.5, 5))
    a = [track[l][aud].mean() for l in levels]; v = [track[l][~aud].mean() for l in levels]
    ae = [track[l][aud].std() / np.sqrt(aud.sum()) for l in levels]
    ve = [track[l][~aud].std() / np.sqrt((~aud).sum()) for l in levels]
    w = .38
    ax.bar(lev - w/2, a, w, yerr=ae, capsize=3, color=CA, label="attend Audio (focused)")
    ax.bar(lev + w/2, v, w, yerr=ve, capsize=3, color=CV, label="attend Visual (distracted)")
    ax.set_xticks(lev); ax.set_xticklabels([PRETTY[l].replace(" ", "\n") for l in levels], fontsize=9)
    ax.set_ylabel("EEG→speech tracking (top-k canonical r)")
    ax.set_title("Tracking of each speech level, focused vs distracted"); ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "10_tracking_by_level.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # (2) distraction effect (d) per hierarchy level (levels only; no trend, no stat)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for j, l in enumerate(levels):
        ax.plot([j, j], d_ci[:, j], color=KCOL[kinds[j]], lw=2.2, zorder=1)
        ax.scatter(j, d[j], color=KCOL[kinds[j]], s=90, zorder=3,
                   edgecolor="w", linewidth=1)
        if perm_p[j] < 0.05:
            ax.text(j, d_ci[1, j] + .03, "*", ha="center", fontsize=15)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(lev); ax.set_xticklabels([PRETTY[l].replace(" ", "\n") for l in levels], fontsize=9)
    ax.set_ylabel("distraction effect  (Cohen's d, focused − distracted)")
    ax.set_title("Visual-distraction effect across the speech hierarchy", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for k, c in KCOL.items():                                # kind legend
        ax.scatter([], [], color=c, label=k)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "11_distraction_vs_hierarchy.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(SD, "config_stimdec.yaml"))
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--boots", type=int, default=2000); ap.add_argument("--no-gpt2", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    cfg = io.load_config(args.config)
    fs = float(cfg["preprocess"]["resample_hz"]); band = (cfg["preprocess"]["l_freq"], cfg["preprocess"]["h_freq"])
    n_times = int(round(cfg["block"]["seconds"] * fs)); audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    print("=" * 74); print("VISUAL DISTRACTION × SPEECH HIERARCHY — Kavin (sub001), session 1"); print("=" * 74)

    trials, events, ch_names, prov = preprocess_and_save(cfg)

    # spectrogram dropped per request; 4-level hierarchy (acoustic → linguistic)
    levels = [l for l in hierarchy.LEVELS
              if l != "spectrogram" and not (l == "gpt2_surprisal" and args.no_gpt2)]
    gpt2 = None
    if "gpt2_surprisal" in levels:
        try:
            gpt2 = hierarchy.load_gpt2(); print("GPT-2 loaded")
        except Exception as e:
            print(f"GPT-2 unavailable ({e})"); levels.remove("gpt2_surprisal")

    # build features + collect EEG
    EEG, FEATS, LAB = [], {l: [] for l in levels}, []
    for _, row in events.iterrows():
        t = int(row["trial"]); wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
        words = json.load(open(os.path.join(audio_dir, f"{row['audio_stim']}.words.json"))).get("words", []) \
            if os.path.exists(os.path.join(audio_dir, f"{row['audio_stim']}.words.json")) else []
        if t not in trials or not os.path.exists(wav) or not words:
            continue
        EEG.append(trials[t]["eeg"].T); LAB.append(str(row["label"]))
        for l in levels:
            FEATS[l].append({"envelope": lambda: hierarchy.envelope(wav, fs, n_times, cfg, band),
                             "spectrogram": lambda: hierarchy.spectrogram(wav, fs, n_times, band),
                             "word_onset": lambda: hierarchy.word_onset(words, fs, n_times),
                             "word_frequency": lambda: hierarchy.word_frequency(words, fs, n_times),
                             "gpt2_surprisal": lambda: hierarchy.gpt2_surprisal(words, fs, n_times, *gpt2)}[l]())
    LAB = np.array(LAB); aud = LAB == "Audio"; N = len(EEG)
    kinds = [hierarchy.LEVEL_KIND[l] for l in levels]
    print(f"{N} trials | focused(Audio) {int(aud.sum())}  distracted(Visual) {int((~aud).sum())} | {levels}")

    # tracking per level (aud_cca)
    EEGr = [e.astype(np.float64) for e in EEG]; kcc = max(5, args.k); track = {}
    for l in levels:
        track[l] = feature_tracking(EEGr, [f.astype(np.float64) for f in FEATS[l]], "cca2plus", kcc)
        print(f"  {l:15s}: focused {track[l][aud].mean():+.4f}  distracted {track[l][~aud].mean():+.4f}")

    # ---- statistics (per-level effect + trend) ----
    T = np.vstack([track[l] for l in levels]); lev = np.arange(len(levels))
    def cohend(x):
        a, v = x[aud], x[~aud]
        sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(a)+len(v)-2))
        return (a.mean() - v.mean()) / (sp + 1e-12)
    d = np.array([cohend(track[l]) for l in levels])
    raw_eff = np.array([track[l][aud].mean() - track[l][~aud].mean() for l in levels])
    rng = np.random.RandomState(1); lab = aud.copy(); Pm = np.empty((args.perms, len(levels)))
    for b in range(args.perms):
        rng.shuffle(lab); Pm[b] = [track[l][lab].mean() - track[l][~lab].mean() for l in levels]
    perm_p = np.array([(Pm[:, j] >= raw_eff[j]).mean() for j in range(len(levels))])
    brng = np.random.RandomState(2); ai, vi = np.where(aud)[0], np.where(~aud)[0]
    dboot = np.empty((args.boots, len(levels)))         # bootstrap only for per-level CIs
    for b in range(args.boots):
        bi = np.r_[brng.choice(ai, len(ai), True), brng.choice(vi, len(vi), True)]
        ba = np.r_[np.ones(len(ai), bool), np.zeros(len(vi), bool)]; db = []
        for j in range(len(levels)):
            x = T[j, bi]; a, v = x[ba], x[~ba]
            sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(a)+len(v)-2))
            db.append((a.mean()-v.mean())/(sp+1e-12))
        dboot[b] = db
    d_ci = np.percentile(dboot, [2.5, 97.5], axis=0)
    rho, rho_p = stats.spearmanr(lev, d)
    print(f"\n  Cohen d by level: {np.round(d,2)}")
    print(f"  TREND Spearman ρ={rho:+.2f} (p={rho_p:.3f})")

    # ---- figures ----
    for i, l in enumerate(levels, 1):
        plot_level(i, l, FEATS[l][0], track[l], aud, kinds[i-1],
                   {"d": d[i-1], "p": perm_p[i-1]}, fs)
    plot_compare(levels, track, aud, d, d_ci, perm_p, rho, rho_p, kinds)

    # ---- save + report ----
    out = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "subject": "sub001_Kavin",
           "session": 1, "hypothesis": "visual distraction reduces speech tracking more at higher hierarchy levels",
           "n_trials": N, "n_focused": int(aud.sum()), "n_distracted": int((~aud).sum()),
           "levels": levels, "kinds": kinds, "method": "aud_cca regularised CCA (cca2plus), top-%d canonical r" % kcc,
           "tracking_focused": {l: float(track[l][aud].mean()) for l in levels},
           "tracking_distracted": {l: float(track[l][~aud].mean()) for l in levels},
           "distraction_effect_d": {l: float(d[j]) for j, l in enumerate(levels)},
           "distraction_effect_ci": {l: [float(d_ci[0, j]), float(d_ci[1, j])] for j, l in enumerate(levels)},
           "per_level_perm_p": {l: float(perm_p[j]) for j, l in enumerate(levels)},
           "trend": {"spearman_rho": float(rho), "spearman_p": float(rho_p)}}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    _report(out, levels)
    print("=" * 74); print(f"saved preprocessed data, {len(os.listdir(FIG))} figures, results.json + report.md -> {HERE}")


def _report(o, levels):
    lines = [f"# Visual distraction × speech hierarchy — Kavin (sub001), session 1\n",
             f"*{o['created']}*\n",
             f"**Hypothesis.** {o['hypothesis'].capitalize()}.\n",
             "**Design.** attend-Audio = focused on speech; attend-Visual (Tetris) = visual "
             "distraction. Per-level EEG→speech tracking via the aud_cca regularised CCA; the "
             "distraction effect is the focused−distracted difference (Cohen's d).\n",
             f"**Data.** {o['n_trials']} trials ({o['n_focused']} focused / {o['n_distracted']} distracted), "
             "corrected cable-swap montage, 1–8 Hz, 64 Hz.\n",
             "## Preprocessing\n![preprocessing](figures/00_preprocessing.png)\n",
             "## Each level of the hierarchy\n"]
    for i, l in enumerate(levels, 1):
        d = o["distraction_effect_d"][l]; p = o["per_level_perm_p"][l]
        lines.append(f"### Level {i}: {l} ({o['kinds'][i-1]})\n"
                     f"distraction effect d = {d:+.2f} (p = {p:.3f}); focused {o['tracking_focused'][l]:+.3f} "
                     f"vs distracted {o['tracking_distracted'][l]:+.3f}\n\n![{l}](figures/{i:02d}_{l}.png)\n")
    tr = o["trend"]
    lines += ["## Comparative\n![tracking](figures/10_tracking_by_level.png)\n",
              "![hypothesis](figures/11_distraction_vs_hierarchy.png)\n",
              "## Result\n",
              f"Trend of the distraction effect across the hierarchy: Spearman ρ = {tr['spearman_rho']:+.2f} "
              f"(p = {tr['spearman_p']:.3f}). "
              + ("Directionally consistent with the hypothesis (distraction grows up the hierarchy)"
                 if tr["spearman_rho"] > 0 else "Not in the hypothesised direction")
              + f" — single subject, so treat as a pilot.\n"]
    open(os.path.join(HERE, "report.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
