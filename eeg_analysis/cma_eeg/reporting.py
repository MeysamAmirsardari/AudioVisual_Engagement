"""Figures + a single self-contained HTML report."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from .utils import LOG


# ---------------------------------------------------------------------------
def fig_alignment(events, align_res):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(events.delay_log, events.gap_dur_eeg, s=28, alpha=.8)
    lim = [events.delay_log.min() - .05, events.delay_log.max() + .05]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="logged delay (s)", ylabel="EEG gap duration (s)",
              title=f"Gap-duration cross-check\nmax err {events.dur_err_ms.max():.1f} ms")
    counts = events.label.value_counts()
    ax[1].bar(counts.index, counts.values, color=["#d1495b", "#30638e"])
    ax[1].set(title=f"Usable trials = {len(events)}\n"
                    f"clock offset {align_res.offset_s:.3f}s, "
                    f"edge-match {align_res.match_error_ms:.0f} ms",
              ylabel="trials")
    fig.tight_layout()
    return fig


def fig_psd(raw):
    fig = raw.compute_psd(fmax=45, verbose="ERROR").plot(show=False, amplitude=False)
    fig.suptitle("Cleaned continuous PSD", y=1.02)
    return fig


def fig_ica(prov):
    """Topographies of the removed ICs (only possible with a montage)."""
    ica = prov.get("_ica_object")
    if ica is None or not ica.exclude or not prov.get("montage_applied"):
        return None
    try:
        return ica.plot_components(picks=ica.exclude, show=False)
    except Exception as e:                               # pragma: no cover
        LOG.warning("ICA topomap failed: %s", e)
        return None


def html_ica(prov) -> str:
    """Text summary of the ICA decomposition + why each IC was removed."""
    ica = prov.get("ica")
    if not ica:
        return "<p>ICA not run.</p>"
    reasons = ica.get("reasons", {})
    rows = "".join(
        f"<tr><td>IC{ic:02d}</td><td>{', '.join(reasons.get(ic, ['?']))}</td></tr>"
        for ic in ica.get("excluded", []))
    body = (f"<p>Method <b>{ica.get('method')}</b>, {ica.get('n_components')} "
            f"components; removed <b>{len(ica.get('excluded', []))}</b>.</p>")
    if rows:
        body += ("<table border=1 cellpadding=4 style='border-collapse:collapse'>"
                 "<tr><th>component</th><th>flagged as</th></tr>" + rows + "</table>")
    return body


def fig_evoked_contrast(epochs):
    ev_a = epochs["Audio"].average()
    ev_v = epochs["Visual"].average()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ev_a.times, ev_a.data.std(0) * 1e6, color="#d1495b", label="Audio (GFP)")
    ax.plot(ev_v.times, ev_v.data.std(0) * 1e6, color="#30638e", label="Visual (GFP)")
    ax.axvline(-1.5, color="grey", ls=":", lw=1)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.text(-0.75, ax.get_ylim()[1] * .95, "cue on screen", ha="center", fontsize=8)
    ax.text(0.7, ax.get_ylim()[1] * .95, "anticipatory gap", ha="center", fontsize=8)
    ax.set(xlabel="time from gap onset (s)", ylabel="GFP (uV)",
           title="Global field power by attended modality")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_time_resolved(tr, gap_window):
    t = tr["times"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lo, hi = tr["null_band"]
    ax.fill_between(t, lo, hi, color="#9aa0a6", alpha=.25,
                    label="permutation null (95%)")
    ax.plot(t, tr["thr"], color="#eb5e28", ls="--", lw=1,
            label="cluster-forming thr")
    ax.plot(t, tr["mean"], color="#1b1b1e", lw=2, label="observed AUC")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.axvline(-1.5, color="grey", ls=":", lw=1)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.axvspan(gap_window[0], gap_window[1], color="#f2c94c", alpha=.15,
               label="anticipatory gap")
    if tr["sig"].any():
        ax.plot(t[tr["sig"]], np.full(tr["sig"].sum(), 0.46),
                "s", color="#eb5e28", ms=4, label="p<.05 (cluster)")
    ax.text(-0.75, 0.42, "cue on screen", ha="center", fontsize=8, color="grey")
    ax.set(xlabel="time from gap onset (s)", ylabel="ROC AUC",
           title="Time-resolved decoding of attended modality (Audio vs Visual)",
           ylim=(0.4, min(1.0, tr["mean"].max() + .1)))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fig_gat(gat):
    t = gat["times"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(gat["gat"], origin="lower", cmap="RdBu_r", vmin=.3, vmax=.7,
                   extent=[t[0], t[-1], t[0], t[-1]], aspect="auto")
    ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel="test time (s)", ylabel="train time (s)",
           title="Temporal generalisation (AUC)")
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    return fig


def fig_windows(gap_res, cue_res):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # AUC bars
    names = [cue_res["name"], gap_res["name"]]
    aucs = [cue_res["auc"], gap_res["auc"]]
    ps = [cue_res["p"], gap_res["p"]]
    bars = ax[0].bar(names, aucs, color=["#9aa0a6", "#eb5e28"])
    ax[0].axhline(0.5, color="k", ls="--", lw=1)
    for b, p in zip(bars, ps):
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + .01,
                   f"AUC={b.get_height():.2f}\np={p:.3f}", ha="center", fontsize=8)
    ax[0].set(ylabel="ROC AUC", ylim=(0.4, 1.0), title="Whole-window decoding")
    # permutation null for the gap window
    ax[1].hist(gap_res["perm_null"], bins=30, color="#cfd8dc")
    ax[1].axvline(gap_res["auc"], color="#eb5e28", lw=2,
                  label=f"observed {gap_res['auc']:.2f}")
    ax[1].axvline(0.5, color="k", ls="--", lw=1)
    ax[1].set(xlabel="AUC", ylabel="count",
              title=f"Gap: permutation null (p={gap_res['p']:.3f})")
    ax[1].legend(fontsize=8)
    # confusion matrix for the gap window
    cm = gap_res["confusion"]
    im = ax[2].imshow(cm, cmap="Blues")
    ax[2].set(xticks=[0, 1], yticks=[0, 1],
              xticklabels=["Visual", "Audio"], yticklabels=["Visual", "Audio"],
              xlabel="predicted", ylabel="true", title="Gap confusion")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    return fig


def fig_alpha(alpha, montage_applied):
    if montage_applied:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, data, ttl in zip(ax, [alpha["audio"], alpha["visual"], alpha["diff"]],
                                ["Attend Audio", "Attend Visual", "Audio - Visual"]):
            mne.viz.plot_topomap(data, alpha["info"], axes=a, show=False,
                                 cmap="RdBu_r", contours=4)
            a.set_title(ttl, fontsize=10)
        fig.suptitle(f"Anticipatory alpha ({alpha['band'][0]:.0f}-{alpha['band'][1]:.0f} Hz) "
                     f"log-power | decoding AUC {alpha['auc']:.2f}")
    else:
        order = np.argsort(alpha["diff"])
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(range(len(order)), alpha["diff"][order], color="#30638e")
        ax.set(xlabel="channel (sorted)", ylabel="log alpha power  Audio - Visual",
               title=f"Anticipatory alpha contrast by channel | "
                     f"decoding AUC {alpha['auc']:.2f}  (no montage -> no topomap)")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([alpha["ch_names"][i] for i in order], fontsize=5, rotation=90)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def build_report(paths, raw_clean, epochs, prov, align_res, events,
                 tr, gat, gap_res, cue_res, alpha, cfg) -> str:
    rep = mne.Report(title=f"Anticipatory-gap decoding — {paths.subject}",
                     verbose="ERROR")

    intro = (
        f"<h3>Cross-modal attention — decoding the instruction→stimulus gap</h3>"
        f"<p><b>Subject:</b> {paths.subject} &nbsp; <b>Usable trials:</b> {len(events)} "
        f"(Audio {int((events.label=='Audio').sum())} / "
        f"Visual {int((events.label=='Visual').sum())})</p>"
        f"<p><b>Target:</b> attended modality, decoded during the anticipatory gap "
        f"(screen identical across conditions). Probe responses ignored by design.</p>"
        f"<p><b>Cleaning:</b> band-pass {prov['bandpass']} Hz, "
        f"reference={prov['reference']}, bads={prov['bad_channels'] or 'none'} "
        f"({prov.get('bad_channel_action','n/a')}), "
        f"ICA removed {len(prov.get('ica',{}).get('excluded',[]))} component(s), "
        f"montage {'applied (ASSUMED layout)' if prov['montage_applied'] else 'not applied'}.</p>"
        f"<p><b>Headline result — gap window {cfg['decode']['gap_window']} s:</b> "
        f"AUC = {gap_res['auc']:.3f} (permutation p = {gap_res['p']:.3f}).</p>")
    rep.add_html(intro, title="Summary")

    def add(fig, title, section):
        if fig is None:
            return
        figs = fig if isinstance(fig, list) else [fig]
        rep.add_figure(figs, title=title, section=section)
        slug = title.lower().replace(" ", "_").replace("↔", "").replace("/", "_")
        for i, f in enumerate(figs):
            suffix = f"{slug}.png" if len(figs) == 1 else f"{slug}_{i}.png"
            try:
                f.savefig(paths.fig(suffix), dpi=130, bbox_inches="tight")
            except Exception as e:            # pragma: no cover
                LOG.warning("could not save %s: %s", suffix, e)
            plt.close(f)

    add(fig_alignment(events, align_res), "Marker↔behaviour alignment", "QC")
    add(fig_psd(raw_clean), "Cleaned PSD", "QC")
    rep.add_html(html_ica(prov), title="ICA decomposition", section="Cleaning")
    add(fig_ica(prov), "Removed ICA components", "Cleaning")
    add(fig_evoked_contrast(epochs), "Evoked contrast (GFP)", "Sensor")
    add(fig_time_resolved(tr, cfg["decode"]["gap_window"]),
        "Time-resolved AUC", "Decoding")
    if gat is not None:
        add(fig_gat(gat), "Temporal generalisation", "Decoding")
    add(fig_windows(gap_res, cue_res), "Whole-window decoding + permutation", "Decoding")
    if alpha is not None:
        add(fig_alpha(alpha, prov["montage_applied"]), "Anticipatory alpha", "Decoding")

    out = paths.file("report.html")
    rep.save(out, overwrite=True, open_browser=False)
    LOG.info("Report -> %s", out)
    return out


def save_results(paths, align_res, events, tr, gap_res, cue_res, alpha,
                 gap_window=(0.0, 1.5)) -> None:
    """Machine-readable results: events table + a metrics JSON."""
    events.to_csv(paths.file("events.csv"), index=False)
    # peak of the time-resolved AUC restricted to the anticipatory gap window
    gmask = (tr["times"] >= gap_window[0]) & (tr["times"] <= gap_window[1])
    gpeak_auc = float(tr["mean"][gmask].max())
    gpeak_t = float(tr["times"][gmask][tr["mean"][gmask].argmax()])
    metrics = dict(
        subject=paths.subject,
        n_trials=int(len(events)),
        n_audio=int((events.label == "Audio").sum()),
        n_visual=int((events.label == "Visual").sum()),
        clock_offset_s=round(align_res.offset_s, 4),
        edge_match_error_ms=round(align_res.match_error_ms, 2),
        gap_window=gap_res["window"],
        gap_auc=round(gap_res["auc"], 4), gap_p=round(gap_res["p"], 4),
        cue_window=cue_res["window"], cue_auc=round(cue_res["auc"], 4),
        cue_p=round(cue_res["p"], 4),
        time_resolved_peak_auc=round(float(tr["mean"].max()), 4),
        time_resolved_peak_time_s=round(float(tr["times"][tr["mean"].argmax()]), 4),
        gap_time_resolved_peak_auc=round(gpeak_auc, 4),
        gap_time_resolved_peak_time_s=round(gpeak_t, 4),
        significant_clusters=[[round(a, 3), round(b, 3), round(p, 4)]
                              for a, b, p in tr["sig_clusters"]],
        alpha_auc=(round(alpha["auc"], 4) if alpha else None),
    )
    with open(paths.file("metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("Metrics -> %s", paths.file("metrics.json"))
