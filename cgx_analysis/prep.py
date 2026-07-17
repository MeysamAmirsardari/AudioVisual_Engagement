#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep.py — read the CGX XDF once and cache it for fast re-analysis.

The recording (sub-Meysam ses-S002, the load-modulation AV task) is an XDF with four
LSL streams: the CGX Quick-32r EEG (500 Hz, 10-20 names — no cable-swap issue here),
its impedance stream, the ExpAudioMarkers stim-marker stream (trial_start / av_onset /
audio_onset / block_end / beep / probe events, with difficulty + audio id per trial),
and the ExpAudio waveform. Crucially EEG and markers share ONE synchronised XDF
timeline, so trial onsets come straight from the audio_onset markers (no photodiode).

Writes to cgx_analysis/cache/: raw.fif (29 scalp EEG + 2 ExG as EOG, µV->V, standard
10-20 montage) and trials.json (per-trial onset seconds, difficulty, audio id, probe).
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
XDF = os.path.join(ROOT, "records", "CGX",
                   "sub-Meysam_ses-S002_task-AV_Eng_run-002_eeg.xdf")
CACHE = os.path.join(HERE, "cache")

NON_SCALP = {"A2", "ExG 1", "ExG 2", "ACC32", "ACC33", "ACC34",
             "Packet Counter", "TRIGGER"}
EOG_CH = {"ExG 1", "ExG 2"}
DIFFICULTY_ORDER = ["very_easy", "easy", "medium", "hard", "very_hard", "super_hard"]


def main():
    import mne
    import pyxdf
    mne.set_log_level("ERROR")
    os.makedirs(CACHE, exist_ok=True)
    print("loading XDF (589 MB, ~1 min) ...", flush=True)
    streams, _ = pyxdf.load_xdf(XDF, dejitter_timestamps=True)
    by_type = {}
    for s in streams:
        by_type.setdefault(s["info"]["type"][0], []).append(s)
    eeg = by_type["EEG"][0]
    mrk = by_type["Markers"][0]
    imp = next((v[0] for k, v in by_type.items() if "mped" in k.lower()), None)

    labs = [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    X = np.asarray(eeg["time_series"], dtype=np.float64)          # (n_samp, n_ch), µV
    sfreq = float(eeg["info"]["nominal_srate"][0])
    eeg_t0 = float(eeg["time_stamps"][0])

    scalp = [l for l in labs if l not in NON_SCALP]
    eog = [l for l in labs if l in EOG_CH]
    keep = scalp + eog
    idx = [labs.index(l) for l in keep]
    data = X[:, idx].T * 1e-6                                     # µV -> V, (n_ch, n_samp)
    info = mne.create_info(keep, sfreq, ["eeg"] * len(scalp) + ["eog"] * len(eog))
    raw = mne.io.RawArray(data, info)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="ignore")
    print(f"EEG: {len(scalp)} scalp + {len(eog)} EOG @ {sfreq:.0f} Hz, "
          f"{raw.n_times/sfreq:.0f}s. median|µV| after 1Hz HP = "
          f"{np.median(np.abs(raw.copy().filter(1, None).get_data(picks='eeg')))*1e6:.1f}")

    # --- impedance QC (median over time, per scalp channel) ----------------
    imped = {}
    if imp is not None:
        Z = np.asarray(imp["time_series"], dtype=np.float64)
        zl = [c["label"][0] for c in imp["info"]["desc"][0]["channels"][0]["channel"]]
        for l in scalp:
            zi = zl.index(l + "-Z") if (l + "-Z") in zl else (zl.index(l) if l in zl else None)
            if zi is not None:
                imped[l] = float(np.median(Z[:, zi]))

    # --- trial table from markers ------------------------------------------
    ts = mrk["time_stamps"]
    vals = [json.loads(v[0]) for v in mrk["time_series"]]
    trials = {}
    for t_xdf, v in zip(ts, vals):
        tr = v.get("trial")
        if tr is None:
            continue
        d = trials.setdefault(int(tr), {"trial": int(tr)})
        d["difficulty"] = v.get("difficulty")
        d["audio_stim_id"] = v.get("audio_stim_id")
        lab = v["label"]
        if lab == "audio_onset":
            d["onset_s"] = float(t_xdf - eeg_t0)
        elif lab == "av_onset":
            d.setdefault("onset_s", float(t_xdf - eeg_t0))
        elif lab == "block_end":
            d["block_dur_s"] = float(v.get("block_duration_s", 0))
            d["resets"] = v.get("resets")
        elif lab == "probe_response":
            d["probe_correct"] = v.get("correct"); d["probe_rt_s"] = v.get("rt_s")
        elif lab == "beep_response":
            d["beep_rt_s"] = v.get("rt_s")
    valid = [d for d in trials.values()
             if d.get("block_dur_s", 0) > 5 and "onset_s" in d]
    valid.sort(key=lambda d: d["trial"])
    for d in valid:                                              # load level 0..5
        d["load"] = DIFFICULTY_ORDER.index(d["difficulty"])

    raw.save(os.path.join(CACHE, "raw.fif"), overwrite=True)
    meta = {"sfreq": sfreq, "eeg_t0": eeg_t0, "scalp": scalp, "eog": eog,
            "impedance_kohm": imped, "difficulty_order": DIFFICULTY_ORDER,
            "n_valid": len(valid), "trials": valid, "xdf": os.path.basename(XDF)}
    with open(os.path.join(CACHE, "trials.json"), "w") as f:
        json.dump(meta, f, indent=2)
    from collections import Counter
    print(f"cached raw.fif + trials.json | {len(valid)} valid trials | "
          f"difficulty {dict(Counter(d['difficulty'] for d in valid))}")
    print(f"probe acc = {np.mean([bool(d.get('probe_correct')) for d in valid if d.get('probe_correct') is not None]):.2f}")


if __name__ == "__main__":
    main()
