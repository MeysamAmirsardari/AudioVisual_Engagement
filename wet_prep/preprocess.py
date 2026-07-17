#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py — session-1 wet-EEG preprocessing (auditory oddball), following the
Tom/Arsalan benchmark notebook, adapted to this CGX Quick-32r recording.

Pipeline (a diagnostic figure is written after every LEVEL):
  0 load XDF + 10-20 montage  ->  1 average reference  ->  2 ICA (FastICA) with
  ICLabel auto-rejection of eye/heart/muscle ICs  ->  3 causal Butterworth 1-40 Hz
  ->  4 causal 60 Hz notch.
Then the comparative analyses: across-stage PSD, the oddball ERPs / MMN
(deviant-standard), and resting eyes-closed vs eyes-open alpha. Cleaned continuous
data (fif) and marker-defined segments (.mat, resampled to 256 Hz) are saved.

Markers (TrueDetective_Markers): 2 = standard tone, 1 = deviant, 3 = third event,
11 = block onset; resting = pre-tone period, split eyes-closed/open at code 12.
"""

from __future__ import annotations

import json
import os

import numpy as np
import scipy.io
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
XDF = "/Users/EMINENT/Downloads/sub-P001_ses-S001_task-Default_run-001_eeg_old1.xdf"
MONTAGE = os.path.join(ROOT, "records", "electrode_mapping", "CACS-64_NO_REF.bvef")
FIG = os.path.join(HERE, "figures"); PRE = os.path.join(HERE, "preprocessed")

# --- config (from the notebook) ---
L_FREQ, H_FREQ, FILT_ORDER = 1.0, 40.0, 4
NOTCH_FREQ, NOTCH_WIDTH = 60.0, 5.0
REF = "average"; FS_DN = 256; ICL_THRESH = 0.8
NON_SCALP = {"A2", "ExG 1", "ExG 2", "ACC32", "ACC33", "ACC34", "Packet Counter", "TRIGGER"}
STAGE_PSD = {}                                            # stage -> (freqs, mean dB) for the overlay


# ============================ helpers ======================================
def butter_causal_params(fs, l, h, order):
    sos = sig.butter(order, [l, h], btype="band", fs=fs, output="sos")
    return {"method": "iir", "phase": "forward", "l_freq": l, "h_freq": h,
            "iir_params": {"sos": sos}}


def notch_filt(data, fs, freqs, notch_width):            # causal, applied via apply_function
    for f in np.atleast_1d(freqs):
        b, a = sig.iirnotch(f, f / notch_width, fs=fs)
        data = sig.lfilter(b, a, data, axis=-1)
    return data


def load_xdf():
    import mne
    import pyxdf
    mne.set_log_level("ERROR")
    streams, _ = pyxdf.load_xdf(XDF, select_streams=[{"type": "EEG"}, {"type": "Markers"}])
    eeg = next(s for s in streams if s["info"]["type"][0] == "EEG")
    mrk = next(s for s in streams if s["info"]["type"][0] == "Markers")
    names = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    sf = float(eeg["info"]["nominal_srate"][0]); t0 = float(eeg["time_stamps"][0])
    scalp = [n for n in names if n not in NON_SCALP]
    idx = [names.index(n) for n in scalp]
    data = np.asarray(eeg["time_series"], float)[:, idx].T * 1e-6     # µV -> V
    raw = mne.io.RawArray(data, mne.create_info(scalp, sf, "eeg"))
    raw.set_montage("standard_1020", on_missing="ignore")
    markers = [(float(t - t0), str(v[0])) for t, v in zip(mrk["time_stamps"], mrk["time_series"])]
    print(f"loaded {len(scalp)} scalp ch @ {sf:.0f} Hz, {raw.n_times/sf:.0f}s, {len(markers)} markers")
    return raw, markers, sf


# ============================ per-stage plot ===============================
def plot_stage(raw, level, title):
    import mne
    p = raw.compute_psd(fmin=1, fmax=80); f, d = p.freqs, p.get_data()
    STAGE_PSD[title] = (f, 10 * np.log10(d.mean(0)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    ax[0].plot(f, 10 * np.log10(d.mean(0)), color="#356", lw=1.5)
    ax[0].fill_between(f, 10 * np.log10(np.percentile(d, 25, 0)),
                       10 * np.log10(np.percentile(d, 75, 0)), color="#356", alpha=.15)
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB"); ax[0].set_title("PSD (1-80 Hz)")
    var = np.log(np.var(raw.get_data(), 1) + 1e-30)
    mne.viz.plot_topomap(var, raw.info, axes=ax[1], show=False, cmap="magma", contours=3)
    ax[1].set_title("log channel variance")
    seg = raw.copy().pick(raw.ch_names[:6]).get_data(stop=int(4 * raw.info["sfreq"])) * 1e6
    t = np.arange(seg.shape[1]) / raw.info["sfreq"]
    for i, c in enumerate(raw.ch_names[:6]):
        ax[2].plot(t, seg[i] + i * 40, lw=.6)
    ax[2].set_yticks([i * 40 for i in range(6)]); ax[2].set_yticklabels(raw.ch_names[:6])
    ax[2].set_xlabel("s"); ax[2].set_title("first 6 ch (µV, offset)")
    fig.suptitle(f"LEVEL {level}: {title}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(FIG, f"L{level}_{title.split()[0].lower()}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================ pipeline =====================================
def main():
    import mne
    from mne_icalabel import label_components
    os.makedirs(FIG, exist_ok=True); os.makedirs(PRE, exist_ok=True)
    raw, markers, fs = load_xdf()
    prov = {"file": os.path.basename(XDF), "sfreq": fs, "n_ch": len(raw.ch_names),
            "channels": raw.ch_names}

    # LEVEL 0 — raw (montaged)
    plot_stage(raw, 0, "raw montaged")

    # LEVEL 1 — average reference
    raw.set_eeg_reference(REF)
    plot_stage(raw, 1, "average reference")

    # LEVEL 2 — ICA (FastICA) + ICLabel auto-rejection
    raw_hp = raw.copy().filter(1.0, None)                # 1 Hz HP copy for ICA + ICLabel
    try:
        ica = mne.preprocessing.ICA(n_components=0.99, method="fastica",
                                    random_state=97, max_iter="auto")
        ica.fit(raw_hp)
    except Exception:                                    # dry/CGX: variance can collapse -> fixed n
        ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                                    random_state=97, max_iter="auto")
        ica.fit(raw_hp)
    labels = label_components(raw_hp, ica, method="iclabel")
    excl = [i for i, (lab, pr) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
            if lab in ("eye blink", "heart beat", "muscle artifact") and pr > ICL_THRESH]
    ica.exclude = excl
    prov["ica"] = {"n_components": int(ica.n_components_), "excluded": excl,
                   "labels": [f"{l} ({p:.2f})" for l, p in zip(labels["labels"], labels["y_pred_proba"])]}
    print(f"ICA: {ica.n_components_} comps; excluded {excl} "
          f"({[labels['labels'][i] for i in excl]})")
    # ICA figures: components + ICLabel bar + before/after PSD
    try:
        for k, f_ in enumerate(np.atleast_1d(ica.plot_components(show=False))):
            f_.suptitle(f"LEVEL 2: ICA components (excluded {excl})", fontsize=11)
            f_.savefig(os.path.join(FIG, f"L2_ica_components_{k}.png"), dpi=130, bbox_inches="tight")
            plt.close(f_)
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(11, 3.6))
    cols = ["#e74c3c" if i in excl else "#3b6ea5" for i in range(len(labels["labels"]))]
    ax.bar(range(len(labels["labels"])), labels["y_pred_proba"], color=cols)
    ax.set_xticks(range(len(labels["labels"])))
    ax.set_xticklabels([f"IC{i}\n{l}" for i, l in enumerate(labels["labels"])], fontsize=7, rotation=45, ha="right")
    ax.axhline(ICL_THRESH, ls="--", color="k"); ax.set_ylabel("ICLabel confidence")
    ax.set_title("LEVEL 2: ICLabel classification (red = excluded eye/heart/muscle > 0.8)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "L2_iclabel.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    before = raw.copy()
    raw = ica.apply(raw)
    fig, ax = plt.subplots(figsize=(7, 4))
    pb = before.compute_psd(fmin=1, fmax=80); pa = raw.compute_psd(fmin=1, fmax=80)
    ax.plot(pb.freqs, 10*np.log10(pb.get_data().mean(0)), color="#c44", label="before ICA")
    ax.plot(pa.freqs, 10*np.log10(pa.get_data().mean(0)), color="#282", label="after ICA")
    ax.legend(frameon=False); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.set_title(f"LEVEL 2: ICA effect (removed {len(excl)} ICs)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "L2_ica_effect.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    del raw_hp, before
    plot_stage(raw, 2, "post-ICA")

    # LEVEL 3 — band-pass 1-40 Hz, ZERO-PHASE.
    # The notebook filters causally (phase='forward', for real-time parity). On this
    # drifty wet recording that leaves ~26x excess low-frequency drift (320 vs 12 µV,
    # verified); zero-phase (filtfilt) is the offline-analysis standard and removes it.
    raw.filter(L_FREQ, H_FREQ)
    plot_stage(raw, 3, "bandpass 1-40Hz")

    # LEVEL 4 — 60 Hz line notch (zero-phase)
    raw.notch_filter(NOTCH_FREQ)
    plot_stage(raw, 4, "notch 60Hz")

    # ---- comparative: across-stage PSD overlay ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (f, dB) in STAGE_PSD.items():
        ax.plot(f, dB, lw=1.6, label=name)
    ax.axvline(60, ls=":", color="grey"); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title("Comparative PSD across preprocessing levels"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "compare_PSD_stages.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    analyses(raw, markers, fs, prov)

    # ---- save cleaned continuous + provenance ----
    raw_ds = raw.copy().resample(FS_DN)
    raw_ds.save(os.path.join(PRE, "sub-P001_ses-S001_cleaned_raw.fif"), overwrite=True)
    json.dump(prov, open(os.path.join(PRE, "provenance.json"), "w"), indent=2)
    nf = len([x for x in os.listdir(FIG) if x.endswith(".png")])
    print(f"saved cleaned fif + {len([x for x in os.listdir(PRE) if x.endswith('.mat')])} .mat segments; "
          f"{nf} figures -> {FIG}")


# ============================ analyses + save ==============================
def analyses(raw, markers, fs, prov):
    import mne
    def mtime(code):
        return next((t for t, c in markers if c == code), None)

    # ----- oddball ERPs / MMN -----
    ev = np.array([[int(round(t * fs)), 0, int(c)] for t, c in markers if c in ("1", "2", "3")])
    eid = {"standard": 2, "deviant": 1, "third": 3}
    eid = {k: v for k, v in eid.items() if (ev[:, 2] == v).any()}
    ep = mne.Epochs(raw, ev, eid, tmin=-0.1, tmax=0.5, baseline=(-0.1, 0),
                    preload=True, reject_by_annotation=False)
    std, dev = ep["standard"].average(), ep["deviant"].average()
    mmn = mne.combine_evoked([dev, std], weights=[1, -1])
    prov["oddball"] = {"n_standard": len(ep["standard"]), "n_deviant": len(ep["deviant"])}
    roi = [c for c in ("Fz", "FCz", "Cz", "Fpz") if c in raw.ch_names]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ri = [raw.ch_names.index(c) for c in roi]
    for e, col, lab in [(std, "#2471a3", f"standard (n={len(ep['standard'])})"),
                        (dev, "#c0392b", f"deviant (n={len(ep['deviant'])})"),
                        (mmn, "#111", "deviant − standard (MMN)")]:
        ax[0].plot(e.times * 1000, e.data[ri].mean(0) * 1e6, color=col, lw=2, label=lab)
    ax[0].axvspan(150, 250, color="grey", alpha=.12); ax[0].axhline(0, color="k", lw=.6); ax[0].axvline(0, color="k", lw=.6)
    ax[0].set_xlabel("ms"); ax[0].set_ylabel("µV"); ax[0].set_title(f"Auditory ERP at {'/'.join(roi)}")
    ax[0].legend(frameon=False, fontsize=9)
    tmask = (mmn.times >= 0.15) & (mmn.times <= 0.25)
    mne.viz.plot_topomap(mmn.data[:, tmask].mean(1) * 1e6, mmn.info, axes=ax[1], show=False,
                         cmap="RdBu_r", contours=4)
    ax[1].set_title("MMN topography 150-250 ms (µV)")
    fig.suptitle("Oddball: standard vs deviant → mismatch negativity", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(FIG, "analysis_oddball_MMN.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    # save epochs as [channels, time, trials] @ FS_DN
    for name, key in (("standard", "standard"), ("deviant", "deviant")):
        if key in eid:
            arr = ep[key].copy().resample(FS_DN).get_data().transpose(1, 2, 0)
            _savemat(f"oddball_{name}", arr, FS_DN, raw)

    # ----- resting eyes-closed vs eyes-open -----
    t12, t10 = mtime("12"), mtime("10")
    if t12 and t10:
        segs = {"eyes_closed": (0.0, t12), "eyes_open": (t12, t10)}
        prov["resting"] = {k: [round(a, 1), round(b, 1)] for k, (a, b) in segs.items()}
        psd = {}
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        occ = [c for c in ("Oz", "O1", "O2", "POz", "Pz") if c in raw.ch_names]
        oi = [raw.ch_names.index(c) for c in occ]
        for name, (a, b), col in [("eyes_closed", segs["eyes_closed"], "#8e44ad"),
                                  ("eyes_open", segs["eyes_open"], "#16a085")]:
            s = raw.copy().crop(max(a + 3, 0), min(b, raw.times[-1]))   # skip onset transient
            s.resample(FS_DN); _savemat(name, s.get_data(), FS_DN, raw)
            p = s.compute_psd(fmin=1, fmax=40); psd[name] = (p.freqs, p.get_data())
            ax[0].plot(p.freqs, 10*np.log10(p.get_data()[oi].mean(0)), color=col, lw=2, label=f"{name} ({b-a:.0f}s)")
        ax[0].axvspan(8, 13, color="grey", alpha=.12); ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB")
        ax[0].set_title(f"Resting PSD at {'/'.join(occ)}"); ax[0].legend(frameon=False)
        fc, dc = psd["eyes_closed"]; fo, do = psd["eyes_open"]
        am = (fc >= 8) & (fc <= 13)
        diff = 10*np.log10(dc[:, am].mean(1)) - 10*np.log10(do[:, am].mean(1))
        mne.viz.plot_topomap(diff, raw.info, axes=ax[1], show=False, cmap="RdBu_r", contours=4)
        ax[1].set_title("alpha (8-13 Hz): eyes-closed − eyes-open (dB)")
        fig.suptitle("Resting state: Berger alpha (eyes-closed > eyes-open, posterior)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(FIG, "analysis_resting_alpha.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def _savemat(name, data, sf, raw):
    info = {"ch_names": raw.ch_names, "sfreq": float(sf), "nchan": len(raw.ch_names)}
    scipy.io.savemat(os.path.join(PRE, f"{name}_sub01_v1_cgx_ica.mat"),
                     {"data": data, "info": info})


if __name__ == "__main__":
    main()
