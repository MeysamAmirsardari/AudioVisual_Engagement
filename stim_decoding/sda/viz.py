#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz.py — figures: tracking, decoding, time-resolved, and spatiotemporal TRF."""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def tracking_scatter(res, outdir):
    ra = np.array(res["r_audio"]); rv = np.array(res["r_visual"])
    lab = np.array(res["labels"])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for m, col in [("Audio", "#d1495b"), ("Visual", "#2e7d9b")]:
        k = lab == m
        ax.scatter(ra[k], rv[k], c=col, label=f"attend {m}", s=48,
                   edgecolor="k", linewidth=0.4, alpha=0.85)
    ax.set_xlabel("auditory tracking  (envelope reconstruction r)")
    ax.set_ylabel("visual tracking  (embedding encoding r)")
    ax.set_title("Per-trial neural tracking of each stream")
    ax.legend(frameon=False)
    ax.axline((np.nanmean(ra), np.nanmean(rv)), slope=1, color="gray", ls="--", lw=0.8)
    return _save(fig, outdir, "tracking_scatter.png")


def accuracy_bars(res, outdir):
    m = res["metrics"]
    names = ["direct\n(z-compare)", "LDA\n(indices)"]
    accs = [m["acc_direct"], m["acc_lda"]]
    ps = [m["p_direct"], m["p_lda"]]
    aucs = [m["auc_direct"], m["auc_lda"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, accs, color=["#4c72b0", "#55a868"], edgecolor="k")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    for b, a, p, au in zip(bars, accs, ps, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02,
                f"acc={a:.2f}\nAUC={au:.2f}\np={p:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("attention decoding accuracy")
    ax.set_title(f"Attended-modality decoding "
                 f"(n={m['n_trials']}: {m['n_audio']}A/{m['n_visual']}V)")
    ax.legend(frameon=False)
    return _save(fig, outdir, "accuracy.png")


def time_resolved(res, outdir):
    tr = res["time_resolved"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    W = [float(x) for x in tr["window_lengths_s"]]
    acc = [tr["wl_accuracy"][str(w)] for w in tr["window_lengths_s"]]
    axes[0].plot(W, acc, "o-", color="#4c72b0")
    axes[0].axhline(0.5, color="k", ls="--", lw=1)
    axes[0].set_xscale("log"); axes[0].set_xlabel("decision-window length (s)")
    axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0.3, 1.0)
    axes[0].set_title("Accuracy vs decision-window length")
    axes[1].plot(tr["sliding_centers_s"], tr["sliding_accuracy"], "-", color="#c44e52")
    axes[1].axhline(0.5, color="k", ls="--", lw=1)
    axes[1].set_xlabel("time in block (s)"); axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.3, 1.0); axes[1].set_title("Time-resolved accuracy (sliding window)")
    return _save(fig, outdir, "time_resolved.png")


def spatiotemporal(res, info, outdir, lags_ms=(50, 100, 150, 200, 300, 400)):
    import mne
    from matplotlib.gridspec import GridSpec
    P = res["patterns"]
    fig = plt.figure(figsize=(2.0 * len(lags_ms), 7))
    gs = GridSpec(3, len(lags_ms), figure=fig, hspace=0.5, wspace=0.2)
    rows = [
        (np.array(P["audio_fwd_trf"]), P["delays_fwd_s"],
         "Auditory forward TRF\n(envelope -> EEG)"),
        (np.array(P["audio_bwd_pattern"]), P["delays_bwd_s"],
         "Auditory decoder pattern\n(envelope reconstruction)"),
        (np.array(P["visual_fwd_trf"])[:, 0, :], P["delays_fwd_s"],
         "Visual forward TRF\n(motion -> EEG)"),
    ]
    for r, (data, delays, title) in enumerate(rows):
        vmax = np.nanmax(np.abs(data)) or 1.0
        for i, lag in enumerate(lags_ms):
            di = int(np.argmin(np.abs(np.array(delays) - lag / 1000.0)))
            ax = fig.add_subplot(gs[r, i])
            mne.viz.plot_topomap(data[:, di], info, axes=ax, show=False,
                                 cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
            if r == 0:
                ax.set_title(f"{lag:.0f} ms", fontsize=9)
            if i == 0:
                ax.text(-0.35, 0.5, title, transform=ax.transAxes, fontsize=9,
                        rotation=90, va="center", ha="center")
    fig.suptitle("Spatiotemporal response patterns", y=0.98, fontsize=12)
    return _save(fig, outdir, "spatiotemporal_patterns.png")


def trf_timecourses(res, ch_names, outdir):
    P = res["patterns"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, ttl in [
        (axes[0], "audio_fwd_trf", "Auditory TRF (GFP)"),
        (axes[1], "audio_bwd_pattern", "Auditory decoder (GFP)"),
        (axes[2], None, "Visual TRF (GFP, motion)"),
    ]:
        if key:
            data = np.array(P[key]); delays = P["delays_fwd_s"] if "fwd" in key else P["delays_bwd_s"]
        else:
            data = np.array(P["visual_fwd_trf"])[:, 0, :]; delays = P["delays_fwd_s"]
        gfp = data.std(axis=0)
        ax.plot(np.array(delays) * 1000, gfp, color="#333")
        ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("lag (ms)"); ax.set_title(ttl)
    axes[0].set_ylabel("GFP (a.u.)")
    return _save(fig, outdir, "trf_timecourses.png")


def trf_heatmaps(res, ch_names, outdir):
    """Channels x lags weight images for each TRF / decoder."""
    P = res["patterns"]
    items = [("audio_fwd_trf", "Auditory forward TRF\n(envelope -> EEG)", "delays_fwd_s"),
             ("audio_bwd_weights", "Auditory decoder weights\n(EEG -> envelope)", "delays_bwd_s"),
             (None, "Visual forward TRF\n(motion -> EEG)", "delays_fwd_s")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for ax, (key, ttl, dk) in zip(axes, items):
        data = np.array(P[key]) if key else np.array(P["visual_fwd_trf"])[:, 0, :]
        d = np.array(P[dk]) * 1000
        vmax = np.nanmax(np.abs(data)) or 1.0
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[d[0], d[-1], len(ch_names), 0], interpolation="nearest")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("lag (ms)"); ax.set_title(ttl, fontsize=10)
        ax.set_yticks(np.arange(0, len(ch_names), 3) + 0.5)
        ax.set_yticklabels([ch_names[i] for i in range(0, len(ch_names), 3)], fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("channel")
    fig.suptitle("TRF / decoder weights (channel x lag)", y=1.0)
    return _save(fig, outdir, "trf_heatmaps.png")


def trf_butterfly(res, outdir):
    """All-channel TRF overlays (butterfly)."""
    P = res["patterns"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    specs = [(axes[0], "audio_fwd_trf", "Auditory forward TRF", "delays_fwd_s"),
             (axes[1], None, "Visual forward TRF (motion)", "delays_fwd_s")]
    for ax, key, ttl, dk in specs:
        data = np.array(P[key]) if key else np.array(P["visual_fwd_trf"])[:, 0, :]
        d = np.array(P[dk]) * 1000
        for c in range(data.shape[0]):
            ax.plot(d, data[c], lw=0.4, alpha=0.4, color="#888")
        ax.plot(d, data.mean(0), "k", lw=1.8, label="mean")
        ax.axvline(0, color="gray", lw=0.6); ax.set_xlabel("lag (ms)"); ax.set_title(ttl)
    axes[0].set_ylabel("weight (a.u.)"); axes[0].legend(frameon=False)
    return _save(fig, outdir, "trf_butterfly.png")


def visual_trf_features(res, info, outdir):
    """Per embedding-feature visual TRF: GFP time-course + peak-lag topography."""
    import mne
    P = res["patterns"]
    vt = np.array(P["visual_fwd_trf"])                     # (n_ch, n_vfeat, n_delays)
    names = res.get("visual_features") or [f"f{i}" for i in range(vt.shape[1])]
    d = np.array(P["delays_fwd_s"]) * 1000
    nf = min(vt.shape[1], len(names))
    fig, axes = plt.subplots(nf, 2, figsize=(8, 2.0 * nf),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    axes = np.atleast_2d(axes)
    for i in range(nf):
        w = vt[:, i, :]; gfp = w.std(0)
        axes[i, 0].plot(d, gfp, color="#333"); axes[i, 0].axvline(0, color="gray", lw=0.6)
        axes[i, 0].set_ylabel(names[i], fontsize=9)
        if i == nf - 1:
            axes[i, 0].set_xlabel("lag (ms)")
        li = int(np.argmax(gfp)); vmax = np.nanmax(np.abs(w[:, li])) or 1.0
        mne.viz.plot_topomap(w[:, li], info, axes=axes[i, 1], show=False,
                             cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
        axes[i, 1].set_title(f"{d[li]:.0f} ms", fontsize=8)
    fig.suptitle("Visual encoding TRF per embedding feature (GFP + peak topo)", y=1.0)
    return _save(fig, outdir, "visual_trf_features.png")


def cca_details(res, fitted, info, ch_names, outdir):
    """Canonical correlations, CCA spatial pattern, per-trial index, variate scatter."""
    import mne
    cd = fitted["cca_details"]
    corrs = cd["canonical_corrs"]
    xload = np.array(cd["x_loadings"])                     # (n_lags, n_ch, n_comp)
    lags_ms = np.array(cd["eeg_lags_s"]) * 1000
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(1, len(corrs) + 1), corrs, color="#4c72b0", edgecolor="k")
    ax.set_xlabel("canonical component"); ax.set_ylabel("canonical correlation (train)")
    ax.set_title("CCA canonical correlations")

    gfp = xload[:, :, 0].std(1); li = int(np.argmax(gfp))
    ax = fig.add_subplot(gs[0, 1]); vmax = np.nanmax(np.abs(xload[li, :, 0])) or 1.0
    mne.viz.plot_topomap(xload[li, :, 0], info, axes=ax, show=False, cmap="RdBu_r",
                         vlim=(-vmax, vmax), contours=4)
    ax.set_title(f"CCA comp-1 EEG pattern\n(lag {lags_ms[li]:.0f} ms)")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(lags_ms, gfp, "o-", color="#c44e52")
    ax.set_xlabel("EEG lag (ms)"); ax.set_ylabel("loading GFP")
    ax.set_title("CCA comp-1 loading vs lag")

    ax = fig.add_subplot(gs[1, 0])
    rc = np.array(fitted["r_cca"], float); lab = np.array(fitted["labels"])
    ax.boxplot([rc[lab == "Audio"], rc[lab == "Visual"]], tick_labels=["Audio", "Visual"])
    ax.set_ylabel("per-trial CCA r (LOO)"); ax.set_title("Auditory CCA index by attended")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(cd["comp1_u"], cd["comp1_v"], s=4, alpha=0.25, color="#4c72b0")
    ax.set_xlabel("EEG canonical variate  u"); ax.set_ylabel("envelope canonical variate  v")
    ax.set_title(f"CCA comp-1 canonical variates (r={corrs[0]:.2f})")

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = (f"CCA model\n\ncomponents: {len(corrs)}\n"
           f"canonical r: {', '.join(f'{c:.3f}' for c in corrs)}\n"
           f"EEG lags: {lags_ms[0]:.0f}-{lags_ms[-1]:.0f} ms\n"
           f"alpha(audio bwd): {fitted['alpha_audio']:g}\n"
           f"alpha(visual fwd): {fitted['alpha_visual']:g}")
    ax.text(0.0, 0.9, txt, fontsize=10, va="top", family="monospace")
    fig.suptitle("CCA model details", y=1.0)
    return _save(fig, outdir, "cca_details.png")


def make_all(res, fitted, info, ch_names, outdir):
    todo = [
        ("tracking", lambda: tracking_scatter(res, outdir)),
        ("accuracy", lambda: accuracy_bars(res, outdir)),
        ("time_resolved", lambda: time_resolved(res, outdir)),
        ("spatiotemporal", lambda: spatiotemporal(res, info, outdir)),
        ("trf_timecourses", lambda: trf_timecourses(res, ch_names, outdir)),
        ("trf_heatmaps", lambda: trf_heatmaps(res, ch_names, outdir)),
        ("trf_butterfly", lambda: trf_butterfly(res, outdir)),
        ("visual_trf_features", lambda: visual_trf_features(res, info, outdir)),
        ("cca_details", lambda: cca_details(res, fitted, info, ch_names, outdir)),
    ]
    figs = {}
    for name, fn in todo:
        try:
            figs[name] = fn()
        except Exception as e:            # never let one plot sink the run
            figs[name] = f"FAILED: {e}"
    return figs
