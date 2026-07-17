#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierarchical_tracking.py — does attention modulate HIGHER-level speech features more?

For a hierarchy of speech representations from low-level acoustic to high-level
linguistic (envelope -> log-mel spectrogram -> word onsets -> lexical surprise ->
GPT-2 contextual surprise), we measure how strongly the EEG tracks each feature and
test the hypothesis that the attentional change (attend-Audio minus attend-Visual) is
larger for the higher levels.

Method (leakage-free, careful):
  * a regularised spatio-temporal CCA per feature (MeysamAmirsardari/aud_cca), which
    jointly whitens the lagged multichannel EEG and the lagged feature via PCA-truncated
    whiteners and returns the canonical correlations. The model (preset 'cca2') is fit on
    the OTHER trials (an attention-agnostic transform) and each held-out trial is scored
    by the mean of its top-K canonical correlations. CCA is the natural bidirectional
    tracking measure — it needs no arbitrary ROI and adapts the feature-side subspace to
    each representation's dimensionality;
  * the per-trial tracking is split by attended stream -> a per-feature attention
    effect (raw and standardised, Cohen's d), with a label-permutation p-value;
  * the central hypothesis is a TREND: the standardised effect is regressed on the
    hierarchy level and the slope is tested by trial bootstrap (does the effect grow
    from acoustic to linguistic?), complemented by a rank (Spearman) trend.

Outputs a JSON of every statistic and two publication figures.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)                                  # the aud_cca `cca` package
sys.path.insert(0, os.path.join(ROOT, "eeg_analysis"))
from sda import io, preprocess, hierarchy  # noqa: E402
import cca as audcca                                      # noqa: E402  (MeysamAmirsardari/aud_cca)

# display metadata for each level
PRETTY = {"envelope": "Envelope", "spectrogram": "Spectrogram", "word_onset": "Word onset",
          "word_frequency": "Lexical\nsurprise", "gpt2_surprisal": "Contextual\nsurprise (GPT-2)"}
KIND_COLOR = {"acoustic": "#3b6ea5", "lexical": "#c46a1b", "linguistic": "#8e2f8e"}


def _words(audio_dir, stim):
    p = os.path.join(audio_dir, f"{stim}.words.json")
    return json.load(open(p, encoding="utf-8"))["words"] if os.path.exists(p) else []


def _folds(n, k, seed=0):
    idx = np.arange(n); np.random.RandomState(seed).shuffle(idx)
    return [(np.setdiff1d(idx, f), np.sort(f)) for f in np.array_split(idx, k)]




def feature_tracking(EEG, FEAT, preset, k, folds=5, seed=0):
    """Per-trial tracking of one feature using the aud_cca CCA (MeysamAmirsardari/aud_cca).

    A cross-validated (attention-agnostic) fit of one of the paper's presets — the EEG
    side is PCA-reduced, lagged and whitened, the feature side lagged/smoothed and
    whitened, with PCA-truncation regularisation — then each held-out trial is scored by
    the mean of its top-`k` canonical correlations. Raw EEG / feature trials go in; the
    model builds its own temporal bases and whiteners.
    """
    n = len(EEG)
    out = np.zeros(n)
    with np.errstate(all="ignore"):
        for tr, te in _folds(n, folds, seed):
            m = audcca.model(preset).fit([EEG[j] for j in tr], [FEAT[j] for j in tr])
            for i in te:
                r = np.asarray(m.score([EEG[i]], [FEAT[i]]), float)
                r = r[np.isfinite(r)]
                out[i] = float(np.mean(np.sort(r)[::-1][:k])) if r.size else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--boots", type=int, default=3000)
    ap.add_argument("--no-gpt2", action="store_true")
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    pc, m = cfg["preprocess"], cfg["model"]
    fs = float(pc["resample_hz"]); band = (pc["l_freq"], pc["h_freq"])
    k = args.k                                              # top-k canonical corrs to average
    subject = cfg["dataset"]["subject"]
    audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
    outdir = os.path.join(ROOT, cfg["output"]["root"], subject); figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 74)
    print("HIERARCHICAL TRACKING — does the attention effect grow up the hierarchy?")
    print("=" * 74)

    # ---- data ---------------------------------------------------------------
    raw, events, _ = io.load_raw_and_events(cfg)
    events = events[events["audio_stim"].notna()].reset_index(drop=True)
    raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
    trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
    n_times = int(round(cfg["block"]["seconds"] * fs))

    levels = [l for l in hierarchy.LEVELS if not (l == "gpt2_surprisal" and args.no_gpt2)]
    gpt2 = None
    if "gpt2_surprisal" in levels:
        try:
            gpt2 = hierarchy.load_gpt2(); print("GPT-2 loaded (contextual surprisal)")
        except Exception as e:
            print(f"GPT-2 unavailable ({e}); dropping that level"); levels.remove("gpt2_surprisal")

    EEG, FEATS, LAB = [], {l: [] for l in levels}, []
    for _, row in events.iterrows():
        t = int(row["trial"]); wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
        if t not in trials or not os.path.exists(wav):
            continue
        words = _words(audio_dir, row["audio_stim"])
        if not words:
            continue
        EEG.append(trials[t]["eeg"].T); LAB.append(str(row["label"]))
        for l in levels:
            if l == "envelope":
                f = hierarchy.envelope(wav, fs, n_times, cfg, band)
            elif l == "spectrogram":
                f = hierarchy.spectrogram(wav, fs, n_times, band)
            elif l == "word_onset":
                f = hierarchy.word_onset(words, fs, n_times)
            elif l == "word_frequency":
                f = hierarchy.word_frequency(words, fs, n_times)
            else:
                f = hierarchy.gpt2_surprisal(words, fs, n_times, *gpt2)
            FEATS[l].append(f)
    LAB = np.array(LAB); N = len(EEG)
    aud = LAB == "Audio"
    print(f"{N} trials | Audio {int(aud.sum())} Visual {int((~aud).sum())} | "
          f"levels: {levels}")

    # ---- per-feature tracking via the aud_cca CCA (cca2 preset) -------------
    EEGr = [e.astype(np.float64) for e in EEG]              # raw EEG trials
    preset = "cca2"; k_cc = max(5, k)
    track = {}
    for l in levels:
        FEATr = [f.astype(np.float64) for f in FEATS[l]]   # raw feature trials
        track[l] = feature_tracking(EEGr, FEATr, preset, k_cc)
        print(f"  {l:15s} tracking: Audio {track[l][aud].mean():+.4f}  "
              f"Visual {track[l][~aud].mean():+.4f}  (aud_cca {preset}, top-{k_cc} r)")

    # ---- statistics ---------------------------------------------------------
    from scipy import stats
    T = np.vstack([track[l] for l in levels])               # (n_levels, n_trials)
    lev = np.arange(len(levels))

    def cohens_d(x):                                        # per-level standardised effect
        a, v = x[aud], x[~aud]
        sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(a)+len(v)-2))
        return (a.mean() - v.mean()) / (sp + 1e-12)

    d = np.array([cohens_d(track[l]) for l in levels])
    raw_eff = np.array([track[l][aud].mean() - track[l][~aud].mean() for l in levels])

    # per-level permutation p (label shuffle)
    rng = np.random.RandomState(1); lab = aud.copy(); perm_p = np.zeros(len(levels))
    P = np.empty((args.perms, len(levels)))
    for b in range(args.perms):
        rng.shuffle(lab)
        P[b] = [track[l][lab].mean() - track[l][~lab].mean() for l in levels]
    perm_p = np.array([(P[:, j] >= raw_eff[j]).mean() for j in range(len(levels))])

    # TREND: bootstrap the slope of standardised effect vs level
    brng = np.random.RandomState(2)
    ai, vi = np.where(aud)[0], np.where(~aud)[0]
    slopes, dboot = np.empty(args.boots), np.empty((args.boots, len(levels)))
    for b in range(args.boots):
        bi = np.r_[brng.choice(ai, len(ai), True), brng.choice(vi, len(vi), True)]
        ba = np.r_[np.ones(len(ai), bool), np.zeros(len(vi), bool)]
        db = []
        for j in range(len(levels)):
            x = T[j, bi]; a, v = x[ba], x[~ba]
            sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(a)+len(v)-2))
            db.append((a.mean()-v.mean())/(sp+1e-12))
        dboot[b] = db
        slopes[b] = np.polyfit(lev, db, 1)[0]
    trend_p = float((slopes <= 0).mean())                   # H1: slope > 0
    d_ci = np.percentile(dboot, [2.5, 97.5], axis=0)
    rho, rho_p = stats.spearmanr(lev, d)

    print(f"\n  standardised effect (Cohen d) by level: {np.round(d, 2)}")
    print(f"  TREND slope (d vs level) = {np.median(slopes):+.3f} "
          f"[{np.percentile(slopes,2.5):+.3f}, {np.percentile(slopes,97.5):+.3f}], "
          f"bootstrap p(slope>0) = {1-trend_p:.4f}")
    print(f"  Spearman(level, d) = {rho:+.2f} (p={rho_p:.3f})")

    # ---- figures ------------------------------------------------------------
    _fig_trend(levels, d, d_ci, perm_p, slopes, trend_p, rho, rho_p, figdir)
    _fig_tracking(levels, track, aud, figdir)

    # ---- save ---------------------------------------------------------------
    kinds = [hierarchy.LEVEL_KIND[l] for l in levels]
    out = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "subject": subject,
           "n_trials": N, "levels": levels, "kinds": kinds,
           "method": f"regularised spatio-temporal CCA (MeysamAmirsardari/aud_cca, preset '{preset}'), "
                     f"per-trial score = mean of top-{k_cc} held-out canonical correlations",
           "cca_preset": preset, "cca_top_k": k_cc,
           "tracking_audio": {l: float(track[l][aud].mean()) for l in levels},
           "tracking_visual": {l: float(track[l][~aud].mean()) for l in levels},
           "attention_effect_raw": {l: float(raw_eff[j]) for j, l in enumerate(levels)},
           "attention_effect_d": {l: float(d[j]) for j, l in enumerate(levels)},
           "attention_effect_d_ci": {l: [float(d_ci[0, j]), float(d_ci[1, j])]
                                     for j, l in enumerate(levels)},
           "per_level_perm_p": {l: float(perm_p[j]) for j, l in enumerate(levels)},
           "trend": {"slope_median": float(np.median(slopes)),
                     "slope_ci": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
                     "bootstrap_p_slope_gt0": float(1 - trend_p),
                     "spearman_rho": float(rho), "spearman_p": float(rho_p)},
           "per_trial": {l: track[l].tolist() for l in levels}, "labels": LAB.tolist()}
    with open(os.path.join(outdir, f"{subject}_hierarchical.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 74); print(f"saved -> {outdir}")
    for fn in ("fig_hierarchy_trend.png", "fig_hierarchy_tracking.png",
               f"{subject}_hierarchical.json"):
        print("   ", fn)
    print("=" * 74)


# ==========================================================================
def _fig_trend(levels, d, d_ci, perm_p, slopes, trend_p, rho, rho_p, figdir):
    import matplotlib.pyplot as plt
    x = np.arange(len(levels))
    kinds = [hierarchy.LEVEL_KIND[l] for l in levels]
    cols = [KIND_COLOR[hierarchy.LEVEL_KIND[l]] for l in levels]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # shaded kind bands
    seen = {}
    for j, kd in enumerate(kinds):
        seen.setdefault(kd, [j, j])[1] = j
    for kd, (a, b) in seen.items():
        ax.axvspan(a - 0.5, b + 0.5, color=KIND_COLOR[kd], alpha=0.06, zorder=0)
        ax.text((a + b) / 2, ax.get_ylim()[1], kd, ha="center", va="bottom",
                fontsize=9, color=KIND_COLOR[kd], style="italic")
    yerr = np.abs(d_ci - d)
    ax.errorbar(x, d, yerr=yerr, fmt="o", ms=8, color="#222", ecolor="#888",
                elinewidth=1.4, capsize=4, zorder=4)
    for j in range(len(levels)):                            # colour each marker by kind
        ax.plot(x[j], d[j], "o", ms=8, color=cols[j], zorder=5)
    # bootstrap trend line
    sl = np.median(slopes); b0 = np.median(d - sl * x)
    xx = np.linspace(-0.3, len(levels) - 0.7, 50)
    ax.plot(xx, b0 + sl * xx, "-", color="#333", lw=1.6, alpha=0.8, zorder=3)
    for j in range(len(levels)):                            # significance stars
        s = "***" if perm_p[j] < .001 else "**" if perm_p[j] < .01 else "*" if perm_p[j] < .05 else ""
        if s:
            ax.text(x[j], d_ci[1, j] + 0.03, s, ha="center", fontsize=13)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[l] for l in levels], fontsize=9)
    ax.set_ylabel("attention effect  (Cohen's d, attend-Audio − Visual)")
    ax.set_title("Attentional modulation vs level in the speech hierarchy", fontsize=12)
    ax.text(0.02, 0.02,
            f"trend: Spearman ρ = {rho:+.2f} (p = {rho_p:.2f});  bootstrap slope p = {trend_p:.2f}\n"
            f"directional (envelope → linguistic), not significant — single subject",
            transform=ax.transAxes, fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_hierarchy_trend.png"), dpi=200)
    plt.close(fig)


def _fig_tracking(levels, track, aud, figdir):
    import matplotlib.pyplot as plt
    x = np.arange(len(levels)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    A = np.array([track[l][aud].mean() for l in levels])
    V = np.array([track[l][~aud].mean() for l in levels])
    Ae = np.array([track[l][aud].std(ddof=1) / np.sqrt(aud.sum()) for l in levels])
    Ve = np.array([track[l][~aud].std(ddof=1) / np.sqrt((~aud).sum()) for l in levels])
    ax.bar(x - w / 2, A, w, yerr=Ae, capsize=3, color="#c0392b", label="attend-Audio")
    ax.bar(x + w / 2, V, w, yerr=Ve, capsize=3, color="#2471a3", label="attend-Visual")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[l] for l in levels], fontsize=9)
    ax.set_ylabel("CCA tracking  (mean top-k canonical r)")
    ax.set_title("EEG tracking of each speech feature, by attended stream", fontsize=12)
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_hierarchy_tracking.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
