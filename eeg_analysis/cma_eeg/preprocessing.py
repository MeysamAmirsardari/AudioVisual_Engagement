"""Clean the continuous recording: filter, montage, bad channels, reference, ICA.

Every step is logged into a ``dict`` (returned alongside the cleaned raw) so the
HTML report can show exactly what was done. Optional dependencies (picard,
mne-icalabel) are auto-detected; the pipeline degrades cleanly without them.
"""
from __future__ import annotations

import json
import os

import mne
import numpy as np

from .utils import LOG


# ---------------------------------------------------------------------------
def apply_montage(raw: mne.io.BaseRaw, map_path: str, montage_name: str) -> bool:
    """Rename numbered channels to 10-20 labels and set a standard montage.

    Returns True on success. This is spatial-visualisation only; decoding never
    depends on it. Loudly warns that the layout is an ASSUMPTION.
    """
    if not os.path.exists(map_path):
        LOG.warning("Montage map %s not found; keeping numbered channels.", map_path)
        return False
    with open(map_path) as f:
        mapping = json.load(f).get("map", {})
    rename = {ch: mapping[ch] for ch in raw.ch_names if ch in mapping}
    if len(rename) < len(raw.ch_names):
        LOG.warning("Montage map covers %d/%d channels; not applying.",
                    len(rename), len(raw.ch_names))
        return False
    raw.rename_channels(rename)
    try:
        raw.set_montage(mne.channels.make_standard_montage(montage_name),
                        on_missing="warn")
    except Exception as e:                       # pragma: no cover
        LOG.warning("set_montage failed (%s); continuing without positions.", e)
        return False
    LOG.warning("Montage APPLIED from %s (assumed actiCAP-64 layout — VERIFY). "
                "Topographies/interpolation now rely on this assumption.", map_path)
    return True


# ---------------------------------------------------------------------------
def detect_bad_channels(raw: mne.io.BaseRaw, zscore: float = 4.0,
                        flat_uv: float = 0.5, manual=None) -> list[str]:
    """Flag channels by robust log-variance outliers and near-flat traces."""
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    names = [raw.ch_names[i] for i in picks]

    logvar = np.log(np.var(data, axis=1) + 1e-30)
    med, mad = np.median(logvar), np.median(np.abs(logvar - np.median(logvar))) + 1e-30
    robust_z = 0.6745 * (logvar - med) / mad
    p2p_uv = (data.max(axis=1) - data.min(axis=1)) * 1e6

    bads = set(manual or [])
    for n, z, p2p in zip(names, robust_z, p2p_uv):
        if abs(z) > zscore or p2p < flat_uv:
            bads.add(n)
    bads = sorted(bads)
    if bads:
        LOG.info("Bad channels flagged: %s", bads)
    else:
        LOG.info("No bad channels flagged.")
    return bads


# ---------------------------------------------------------------------------
def _pick_ica_method(method: str) -> str:
    if method == "picard":
        try:
            import picard  # noqa: F401
            return "picard"
        except Exception:
            LOG.warning("python-picard not installed; falling back to infomax.")
            return "infomax"
    return method


def run_ica(raw: mne.io.BaseRaw, cfg: dict, has_montage: bool) -> tuple[mne.preprocessing.ICA, dict]:
    """Fit ICA and flag ocular / muscle / non-brain components for removal."""
    method = _pick_ica_method(cfg.get("method", "infomax"))
    fit_kwargs = dict(method=method, max_iter="auto", random_state=97)
    if method == "infomax":
        fit_kwargs["fit_params"] = dict(extended=True)
    ica = mne.preprocessing.ICA(n_components=cfg.get("n_components", 0.99),
                                **fit_kwargs)
    ica.fit(raw, decim=int(cfg.get("fit_decim", 4)), verbose="ERROR")
    LOG.info("ICA fit: %d components (method=%s)", ica.n_components_, method)

    info = {"method": method, "n_components": int(ica.n_components_),
            "excluded": [], "reasons": {}}
    exclude: set[int] = set()

    # --- ocular: correlate ICs with frontal EOG-proxy channels ---------------
    for proxy in cfg.get("eog_proxy", []):
        if proxy not in raw.ch_names:
            continue
        try:
            idx, _ = ica.find_bads_eog(raw, ch_name=proxy,
                                       threshold=cfg.get("eog_threshold", 3.0),
                                       verbose="ERROR")
            for i in idx:
                exclude.add(i); info["reasons"].setdefault(i, []).append(f"eog:{proxy}")
        except Exception as e:
            LOG.warning("find_bads_eog(%s) failed: %s", proxy, e)

    # --- muscle: high-frequency spatially-broad components --------------------
    if cfg.get("muscle", True):
        try:
            idx, _ = ica.find_bads_muscle(raw, verbose="ERROR")
            for i in idx:
                exclude.add(i); info["reasons"].setdefault(i, []).append("muscle")
        except Exception as e:
            LOG.warning("find_bads_muscle failed: %s", e)

    # --- optional ICLabel (needs positions + mne-icalabel) -------------------
    if cfg.get("use_iclabel", False) and has_montage:
        try:
            from mne_icalabel import label_components
            labels = label_components(raw, ica, method="iclabel")
            probs, cats = labels["y_pred_proba"], labels["labels"]
            for i, (c, p) in enumerate(zip(cats, probs)):
                if c not in ("brain", "other") and p >= cfg.get("iclabel_prob", 0.8):
                    exclude.add(i); info["reasons"].setdefault(i, []).append(f"iclabel:{c}")
            info["iclabel"] = list(zip(cats, [round(float(p), 2) for p in probs]))
        except Exception as e:
            LOG.warning("ICLabel unavailable/failed (%s); using heuristics only.", e)

    ica.exclude = sorted(exclude)
    info["excluded"] = ica.exclude
    info["reasons"] = {int(k): v for k, v in info["reasons"].items()}
    LOG.info("ICA excluding %d components: %s", len(ica.exclude), ica.exclude)
    return ica, info


# ---------------------------------------------------------------------------
def preprocess(raw: mne.io.BaseRaw, cfg: dict) -> tuple[mne.io.BaseRaw, dict]:
    """Full cleaning chain. Returns (clean_raw, provenance dict)."""
    prov: dict = {}
    raw = raw.copy()

    # 1) montage (optional, spatial only)
    has_montage = False
    if cfg.get("apply_montage", False):
        has_montage = apply_montage(raw, cfg["montage_map"], cfg["montage_name"])
    prov["montage_applied"] = has_montage

    # 2) filtering
    l, h = cfg.get("l_freq"), cfg.get("h_freq")
    raw.filter(l, h, fir_design="firwin", verbose="ERROR")
    prov["bandpass"] = [l, h]
    if cfg.get("notch"):
        raw.notch_filter(cfg["notch"], verbose="ERROR")
        prov["notch"] = cfg["notch"]

    # 3) bad channels
    bc = cfg.get("bad_channels", {})
    bads = []
    if bc.get("detect", True):
        bads = detect_bad_channels(raw, bc.get("zscore", 4.0),
                                   bc.get("flat_uv", 0.5), bc.get("manual", []))
    raw.info["bads"] = bads
    prov["bad_channels"] = bads
    if bads:
        if has_montage:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            prov["bad_channel_action"] = "interpolated"
        else:
            raw.drop_channels(bads)
            prov["bad_channel_action"] = "dropped (no montage)"

    # 4) reference
    if cfg.get("reference", "average") == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        prov["reference"] = "average"
    else:
        prov["reference"] = "unchanged (online ref = ch2)"

    # 5) ICA
    if cfg.get("ica", {}).get("enable", True):
        ica, ica_info = run_ica(raw, cfg["ica"], has_montage)
        ica.apply(raw, verbose="ERROR")
        prov["ica"] = ica_info
        prov["_ica_object"] = ica            # kept for the report; not serialised

    # NB: downsampling happens at the epoching stage (integer `decim`), so the
    # photodiode event samples stay valid in the original sample space and no
    # aliasing occurs (data is already low-passed well below the new Nyquist).
    LOG.info("Preprocessing complete: %d channels @ %.0f Hz",
             len(raw.ch_names), raw.info["sfreq"])
    return raw, prov
