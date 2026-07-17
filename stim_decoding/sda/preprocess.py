#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py — clean the continuous EEG ONCE, then crop constant-geometry blocks.

Why this matters for CCA (the "per-trial interpolation trap"): CCA learns spatial
filters from the channel covariance Cxx. If bad channels are interpolated and the
data re-referenced PER TRIAL, the spatial rank, geometry, and reference of Cxx
change trial-by-trial, so a filter fit on the pooled training trials is invalid on
the held-out one. The fix here is to do ALL spatial operations at the continuous /
session level, so every extracted block lives in exactly the same, fixed channel
space:

  1. rename physical electrode numbers -> 10-20 labels and set the REAL cap montage
     (ground-truth .bvef; map BY NAME so a missing electrode never shifts the rest);
  2. band-pass;
  3. interpolate the SESSION bad channels ONCE (channels bad in a significant
     fraction of trials, from the time-varying schedule) — before ICA;
  4. ICA fit on a separate broadband 1--40 Hz copy (ocular / muscle), then apply
     it to the continuous recording;
  5. apply the final model band, add the online-reference electrode back as a flat
     channel, and apply a GLOBAL average reference (recovers that electrode as
     -average);
  6. resample to the model rate.

`extract_trials` then merely CROPS each trial's av-onset-locked block from the
already-cleaned, already-referenced continuous data — no per-trial interpolation,
no per-trial reference. Cxx is identical in shape/geometry across trials.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EEG = os.path.join(ROOT, "eeg_analysis")
if _EEG not in sys.path:
    sys.path.insert(0, _EEG)
from cma_eeg import preprocessing as cmapp  # noqa: E402  (run_ica / apply_montage)


# --------------------------------------------------------------------------
# Montage / channel-name mapping
# --------------------------------------------------------------------------
def _swap_cable_halves(num2name):
    """Exchange the two 32-channel cable bundles: a recorded channel numbered n is
    remapped to the POSITION of electrode n+32 (n<=32) or n-32 (n>32).

    Used when the two coloured 32-channel cables were plugged in REVERSED for a
    recording (we assumed 1:32=green / 33:64=yellow, but the first recordings had
    1:32=yellow / 33:64=green). The .bvef geometry is unchanged; only which recorded
    channel sits at which electrode swaps."""
    def swap(n):
        return n + 32 if n <= 32 else n - 32
    return {str(n): num2name[str(swap(n))]
            for n in range(1, 65) if str(swap(n)) in num2name}


def _bvef(cfg):
    """(mne montage, {physical_number_str: 10-20 name}) from the .bvef, or (None, None).
    If preprocess.cable_halves_swapped is set, the two 32-channel halves are swapped."""
    import mne
    import xml.etree.ElementTree as ET
    p = cfg["preprocess"].get("montage_bvef")
    if not p:
        return None, None
    p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    if not os.path.exists(p):
        print(f"  !! WARNING: montage_bvef not found ({p}); falling back to the "
              f"ASSUMED template, KNOWN WRONG for electrodes 40-64. Spatial "
              f"results will be corrupt.")
        return None, None
    montage = mne.channels.read_custom_montage(p)
    num2name = {}
    for e in ET.parse(p).getroot().findall("Electrode"):
        num = e.findtext("Number")
        if num is not None:
            num2name[str(int(num))] = e.findtext("Name")
    if cfg["preprocess"].get("cable_halves_swapped"):
        num2name = _swap_cable_halves(num2name)
        print("  MONTAGE: cable halves SWAPPED (1:32 <-> 33:64) — recorded channel n "
              "placed at electrode n±32 (reversed cable colours on this recording).")
    return montage, num2name


def _load_channel_map(cfg: dict) -> dict:
    import json
    mm = cfg["preprocess"]["montage_map"]
    mm = mm if os.path.isabs(mm) else os.path.normpath(
        os.path.join(ROOT, "stim_decoding", mm))
    with open(mm, encoding="utf-8") as f:
        return json.load(f)["map"]


# --------------------------------------------------------------------------
# Electrode impedances (measured at recording start, stored in the .vhdr)
# --------------------------------------------------------------------------
def _read_impedances(cfg: dict) -> dict[str, float]:
    """Per recorded-channel electrode impedance (kOhm), parsed from the BrainVision
    .vhdr `[Comment]` block ('<n>:  <kOhm>'). Keyed by recorded/input NUMBER (str),
    so the cable-half swap is applied later via num2name. Empty dict if unavailable."""
    vhdr = cfg.get("dataset", {}).get("vhdr")
    if not vhdr:
        return {}
    vhdr = vhdr if os.path.isabs(vhdr) else os.path.join(ROOT, vhdr)
    if not os.path.exists(vhdr):
        return {}
    imp: dict[str, float] = {}
    in_block = False
    try:
        with open(vhdr, encoding="latin-1") as f:
            for line in f:
                if "Impedance" in line and "kOhm" in line:
                    in_block = True                      # 'Impedance [kOhm] at HH:MM:SS :'
                    continue
                m = re.match(r"^\s*(\d+)\s*:\s+(\d+)\s*$", line)
                if in_block and m:
                    imp[m.group(1)] = float(m.group(2))
    except Exception:
        return {}
    return imp


def impedance_bad_channels(cfg: dict, num2name: dict) -> tuple[list[str], dict]:
    """Electrodes whose measured impedance is at/above the configured bad level.

    Impedance is keyed by recorded input number; num2name places it at the (possibly
    cable-swapped) 10-20 label. High contact impedance is a leading cause of noisy /
    disconnected channels, so it is used here as an INDEPENDENT bad-channel flag that
    catches poor contacts even when their variance looks normal. Returns
    (bad_names_sorted, {name: kOhm for every mapped channel})."""
    pc = cfg["preprocess"]
    thr = float(pc.get("impedance_bad_kohm", 60.0))
    imp = _read_impedances(cfg)
    by_name: dict[str, float] = {}
    bad: list[str] = []
    for num, k in imp.items():
        nm = num2name.get(str(num)) if num2name else None
        if nm is None:
            continue
        by_name[nm] = k
        if k >= thr:
            bad.append(nm)
    return sorted(set(bad)), by_name


def _neighbour_correlation(raw) -> dict[str, float]:
    """Mean correlation of each EEG channel with its <=5 nearest spatial neighbours."""
    from scipy.spatial import cKDTree
    r = raw.copy().pick("eeg")
    mont = r.get_montage()
    pos = mont.get_positions()["ch_pos"] if mont is not None else {}
    names = [n for n in r.ch_names if n in pos and np.all(np.isfinite(pos[n]))]
    if len(names) < 6:
        return {}
    P = np.asarray([pos[n] for n in names])
    C = np.corrcoef(r.copy().pick(names).get_data())
    k = min(5, len(names) - 1)
    _, nn = cKDTree(P).query(P, k=k + 1)
    return {names[i]: float(np.mean(C[i, nn[i, 1:]])) for i in range(len(names))}


def interpolate_spatial_outliers(raw, montage, cfg: dict, prov: dict):
    """AFTER cleaning + referencing, interpolate any channel that has become a focal
    spatial "bullseye" -- one whose data no longer correlates with its nearest
    neighbours (mean neighbour-corr below the floor). These are NOT bad electrodes
    (they pass raw impedance/amplitude QC); they are channels damaged by aggressive
    ICA component removal and then amplified by the average reference into a lone
    hot/cold spot in every topography. Interpolating at the session level preserves
    the constant CCA channel geometry; the average reference is re-applied after.
    The floor (~0.2) spares good channels merely *adjacent* to the outlier (their
    neighbour-corr is dragged down but stays well above the floor)."""
    floor = float(cfg["preprocess"].get("post_clean_min_neighbor_corr", 0.20))
    nbr = _neighbour_correlation(raw)
    outliers = sorted(n for n, c in nbr.items() if c < floor)
    prov["post_clean_neighbour_correlation"] = nbr
    prov["post_clean_spatial_outliers"] = outliers
    if outliers and montage is not None:
        raw.info["bads"] = outliers
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


# --------------------------------------------------------------------------
# Session-level bad channels
# --------------------------------------------------------------------------
def session_bad_channels(cfg: dict, num2name: dict) -> list[str]:
    """
    Channels to interpolate ONCE for the whole session: from the time-varying
    `bad_channel_schedule` (trial-range -> electrode numbers), keep those flagged
    bad in >= `session_bad_min_frac` of the covered trials, mapped to 10-20 names.
    Doing this at the session level (instead of per trial) keeps the channel space
    constant, which is what CCA's covariance needs.
    """
    sched = cfg["preprocess"].get("bad_channel_schedule", {}) or {}
    if not sched:
        return []
    per_chan: dict[int, set] = {}
    max_trial = 0
    for rng, chans in sched.items():
        m = re.match(r"(\d+)\s*-\s*(\d+)", str(rng))
        lo, hi = (int(m.group(1)), int(m.group(2))) if m else (int(rng), int(rng))
        max_trial = max(max_trial, hi)
        for c in chans:
            per_chan.setdefault(int(c), set()).update(range(lo, hi + 1))
    total = max(max_trial, 1)
    min_frac = float(cfg["preprocess"].get("session_bad_min_frac", 0.1))
    names = []
    for c, trials in per_chan.items():
        if len(trials) / total >= min_frac:
            names.append(num2name.get(str(c), str(c)) if num2name else str(c))
    return sorted(set(names))


# --------------------------------------------------------------------------
# Continuous QC: conservative, auditable automatic bad-channel screening
# --------------------------------------------------------------------------
def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z scores, robust to the handful of bad channels sought here."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return .6745 * (x - med) / max(mad, np.finfo(float).eps)


def continuous_channel_qc(raw, cfg: dict) -> tuple[list[str], dict[str, Any]]:
    """Return *conservatively* flagged channels and a JSON-safe QC ledger.

    Manual/scheduled bad-channel knowledge remains authoritative.  This detector is
    deliberately an additional safety net: it looks for non-finite/flat channels,
    variance outliers, weak common-mode correlation, and unusually large 20--45 Hz
    noise.  Metrics are computed on a decimated view of the continuous recording so
    QC cannot silently become the memory bottleneck of preprocessing.
    """
    pc = cfg["preprocess"]
    qcfg = pc.get("channel_qc", {}) or {}
    if not qcfg.get("enable", True):
        return [], {"enabled": False}

    import scipy.signal as sig

    import mne
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    target_hz = float(qcfg.get("sample_hz", 250.0))
    step = max(1, int(round(raw.info["sfreq"] / target_hz)))
    # Materialise a compact float32, decimated QC view one channel at a time.
    # Advanced indexing all channels at once can temporarily allocate another full
    # recording-sized array; that is unacceptable for 0.5+ GB XDF/BrainVision runs.
    n_qc = (raw.n_times + step - 1) // step
    data = np.empty((len(picks), n_qc), np.float32)
    for j, pick in enumerate(picks):
        source = raw._data[pick] if raw.preload else raw.get_data(picks=[pick])[0]
        data[j] = source[::step]
    finite = np.isfinite(data).all(axis=1)
    # Work in microvolts for thresholds that a reviewer can understand.
    data[~np.isfinite(data)] = 0.0
    p2p_uv = np.ptp(data, axis=1) * 1e6
    std_uv = np.nanstd(data, axis=1) * 1e6
    log_std_z = _robust_z(np.log(np.maximum(std_uv, 1e-12)))

    # A disconnected/noisy electrode is often weakly correlated with the robust
    # across-channel reference.  This is intentionally permissive: it must be very
    # unlike the rest of the cap before it is automatically flagged.
    common = np.mean(data, axis=0, dtype=np.float64)
    common = common - common.mean()
    denom_common = np.sqrt(np.sum(common ** 2))
    corr = np.empty(len(names))
    for j, row in enumerate(data):
        centred = row - row.mean()
        corr[j] = np.dot(centred, common) / max(
            np.sqrt(np.sum(centred ** 2)) * denom_common, np.finfo(float).eps)

    hf_ratio = np.full(len(names), np.nan)
    qc_sfreq = raw.info["sfreq"] / step
    nyq = qc_sfreq / 2
    hi = min(45.0, nyq - 1.0)
    if hi > 22.0:
        sos_wide = sig.butter(3, [1.0, hi], btype="bandpass", fs=qc_sfreq,
                              output="sos")
        sos_hf = sig.butter(3, [20.0, hi], btype="bandpass", fs=qc_sfreq,
                            output="sos")
        for j, row in enumerate(data):
            wide = sig.sosfiltfilt(sos_wide, row)
            hf = sig.sosfiltfilt(sos_hf, row)
            hf_ratio[j] = np.std(hf) / max(np.std(wide), 1e-20)
    hf_z = _robust_z(hf_ratio)

    flat = p2p_uv < float(qcfg.get("flat_uv", .5))
    variance = np.abs(log_std_z) > float(qcfg.get("variance_z", 5.0))
    weak_corr = corr < float(qcfg.get("min_common_corr", -.15))
    hf_noisy = hf_z > float(qcfg.get("hf_ratio_z", 5.0))
    automatic = (~finite) | flat | variance | weak_corr | hf_noisy
    reasons = {
        n: [label for label, mask in (("nonfinite", ~finite), ("flat", flat),
                                      ("variance", variance), ("weak_common_corr", weak_corr),
                                      ("high_frequency_noise", hf_noisy)) if mask[i]]
        for i, n in enumerate(names) if automatic[i]
    }
    metrics = {
        "enabled": True, "sample_hz": float(qc_sfreq),
        "p2p_uv": {n: float(v) for n, v in zip(names, p2p_uv)},
        "std_uv": {n: float(v) for n, v in zip(names, std_uv)},
        "log_std_robust_z": {n: float(v) for n, v in zip(names, log_std_z)},
        "common_correlation": {n: float(v) for n, v in zip(names, corr)},
        "hf_ratio": {n: float(v) for n, v in zip(names, hf_ratio)},
        "hf_ratio_robust_z": {n: float(v) for n, v in zip(names, hf_z)},
        "automatic_bads": sorted(reasons), "automatic_reasons": reasons,
    }
    return sorted(reasons), metrics


# --------------------------------------------------------------------------
# Continuous cleaning (all spatial ops here -> constant geometry downstream)
# --------------------------------------------------------------------------
def preprocess_continuous(raw, cfg):
    """Return (raw_model, prov): cleaned, montaged, session-interpolated, ICA'd,
    globally average-referenced (with the online reference recovered), resampled to
    the model rate."""
    pc = cfg["preprocess"]
    prov = {"orig_sfreq": float(raw.info["sfreq"])}
    montage, num2name = _bvef(cfg)
    has_montage = montage is not None

    # 1) rename physical numbers -> 10-20 labels; set the ground-truth montage.
    if has_montage:
        raw.rename_channels({ch: num2name[ch] for ch in raw.ch_names
                             if ch in num2name})
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
        prov["montage_source"] = "CACS-64.bvef (ground truth)"
    else:                                             # degraded fallback (warned above)
        has_montage = cmapp.apply_montage(
            raw, os.path.normpath(os.path.join(ROOT, "stim_decoding", pc["montage_map"])),
            pc.get("montage_name", "standard_1005"))
        num2name = _load_channel_map(cfg)
        prov["montage_source"] = "assumed template"

    # 2) derive a session-level bad-channel decision BEFORE ICA.  Automated QC is
    # deliberately conservative and is combined with the experimenter's schedule;
    # it never does per-trial interpolation, preserving a fixed CCA channel space.
    auto_bads, qc = continuous_channel_qc(raw, cfg)
    scheduled_bads = session_bad_channels(cfg, num2name)
    imp_bads, imp_by_name = impedance_bad_channels(cfg, num2name)
    sess_bads = sorted({b for b in (scheduled_bads + auto_bads + imp_bads)
                        if b in raw.ch_names})
    prov["channel_qc"] = qc
    prov["scheduled_bad_channels"] = scheduled_bads
    prov["impedance_kohm"] = imp_by_name
    prov["impedance_bad_channels"] = [b for b in imp_bads if b in raw.ch_names]
    prov["impedance_bad_kohm_threshold"] = float(pc.get("impedance_bad_kohm", 60.0))
    # Auditable breakdown of WHY each channel is interpolated.
    prov["session_bad_sources"] = {
        b: [src for src, lst in (("schedule", scheduled_bads),
                                 ("impedance", imp_bads),
                                 ("auto_qc", auto_bads)) if b in lst]
        for b in sess_bads}

    # 3) interpolate session bads ONCE, before ICA and before all trial crops.
    if sess_bads and has_montage:
        raw.info["bads"] = sess_bads
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    prov["session_bads_interpolated"] = sess_bads

    # 4) ICA needs broadband data. Fitting it after the final 1--8 Hz modelling
    # filter suppresses the spectral information used to identify muscle artifacts.
    # Filter the continuous object itself to 1--40 Hz (rather than making another
    # recording-sized copy), fit/apply ICA, then construct the model band below.
    # This keeps peak memory practical for long 64-channel recordings.
    if pc.get("ica", {}).get("enable", True):
        ica_band = pc.get("ica", {}).get("fit_band", [1.0, 40.0])
        raw.filter(float(ica_band[0]), float(ica_band[1]), fir_design="firwin", verbose="ERROR")
        ica, ica_info = cmapp.run_ica(raw, pc["ica"], has_montage)
        ica.apply(raw, verbose="ERROR")
        prov["ica"] = ica_info
        prov["ica_fit_band"] = [float(ica_band[0]), float(ica_band[1])]

    # 5) final zero-phase model band-pass, after artifact removal.
    model_lo = pc["l_freq"]
    if pc.get("ica", {}).get("enable", True):
        model_lo = None if float(pc["l_freq"]) <= float(ica_band[0]) else pc["l_freq"]
    raw.filter(model_lo, pc["h_freq"], fir_design="firwin", verbose="ERROR")
    prov["bandpass"] = [pc["l_freq"], pc["h_freq"]]

    # 6) add the online-reference electrode back as a flat channel, then a
    #    GLOBAL average reference (recovers Fz = -average). Reorder to electrode
    #    order for a clean, index==electrode channel space.
    ref_num = str(pc.get("reference_electrode", "")).strip()
    ref_name = num2name.get(ref_num) if (num2name and ref_num) else None
    if ref_name and ref_name not in raw.ch_names:
        raw.add_reference_channels([ref_name])
        if has_montage:
            raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
        order = [num2name[str(n)] for n in range(1, 65)
                 if str(n) in num2name and num2name[str(n)] in raw.ch_names]
        if set(order) == set(raw.ch_names):
            raw.reorder_channels(order)
        prov["reference_electrode_added"] = ref_name
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    prov["reference"] = f"average (global, {len(raw.ch_names)} ch)"

    # 6b) POST-CLEAN spatial-outlier repair. Some channels survive raw QC yet are
    # turned into a focal "bullseye" by aggressive ICA removal + the average reference
    # (e.g. FCz here: healthy at 0.997 neighbour-corr in the raw, driven to -0.6 after
    # cleaning). Detect and interpolate them now, on the final referenced data, so no
    # lone hot/cold spot pollutes the topographies. Session-level -> constant geometry.
    raw = interpolate_spatial_outliers(raw, montage if has_montage else None, cfg, prov)
    prov["post_clean_interpolated"] = prov.get("post_clean_spatial_outliers", [])

    # 7) resample to the model rate only after a low-pass has made it safe.
    fs = float(pc["resample_hz"])
    raw.resample(fs, verbose="ERROR")
    prov["model_sfreq"] = fs
    prov["n_channels"] = len(raw.ch_names)
    prov["montage_applied"] = has_montage
    return raw, prov


# --------------------------------------------------------------------------
# Trial extraction (crop only — constant geometry, already referenced)
# --------------------------------------------------------------------------
def extract_trials(raw_model, events, cfg, prov):
    """
    Crop each trial's av-onset-locked block from the already-cleaned continuous
    data. No per-trial interpolation or re-referencing (all done at the session
    level), so every trial shares one fixed channel space.

    Returns {trial: {"eeg": (n_ch, n_times), "label", "audio_stim"}}, a shared
    mne.Info, and ch_names.
    """
    fs = raw_model.info["sfreq"]
    orig_sf = prov["orig_sfreq"]
    seconds = float(cfg["block"]["seconds"])
    pad = float(cfg["block"].get("onset_pad_s", 0.0))
    n_times = int(round(seconds * fs))
    qcfg = cfg["block"].get("quality", {}) or {}
    reject_uv = float(qcfg.get("reject_uv", 250.0))
    min_std_uv = float(qcfg.get("min_std_uv", .02))
    max_bad_frac = float(qcfg.get("max_bad_channel_fraction", .10))

    picks = raw_model.copy().pick("eeg")               # fixed geometry for all trials
    info = picks.info
    out = {}
    trial_qc = []
    for _, row in events.iterrows():
        trial = int(row["trial"])
        av_time = float(row["av_sample"]) / orig_sf + pad
        if av_time < 0 or av_time + seconds > picks.times[-1]:
            trial_qc.append({"trial": trial, "keep": False, "reason": "incomplete_block"})
            continue
        ep = picks.copy().crop(av_time, av_time + seconds, include_tmax=False)
        data = ep.get_data()                           # (n_ch, n_t)
        # Do not pad a short recording tail: synthetic samples can inflate held-out
        # CCA tracking.  A small one-sample rounding mismatch is cropped only.
        if data.shape[1] < n_times:
            trial_qc.append({"trial": trial, "keep": False, "reason": "short_block",
                             "n_samples": int(data.shape[1])})
            continue
        data = data[:, :n_times]
        p2p_uv = np.ptp(data, axis=1) * 1e6
        std_uv = np.std(data, axis=1) * 1e6
        nonfinite = ~np.isfinite(data).all(axis=1)
        too_large = p2p_uv > reject_uv
        too_flat = std_uv < min_std_uv
        n_bad = int(np.sum(nonfinite | too_large | too_flat))
        if n_bad / data.shape[0] > max_bad_frac:
            trial_qc.append({"trial": trial, "keep": False, "reason": "block_artifact",
                             "n_bad_channels": n_bad, "max_p2p_uv": float(p2p_uv.max())})
            continue
        out[trial] = {"eeg": data.astype(np.float64),
                      "label": row["label"],
                      "audio_stim": row.get("audio_stim"),
                      "bads": []}
        trial_qc.append({"trial": trial, "keep": True, "n_bad_channels": n_bad,
                         "max_p2p_uv": float(p2p_uv.max())})
    prov["block_qc"] = trial_qc
    prov["block_qc_summary"] = {"kept": len(out), "rejected": len(trial_qc) - len(out),
                                  "reject_uv": reject_uv, "max_bad_channel_fraction": max_bad_frac}
    return out, info, list(info["ch_names"])
