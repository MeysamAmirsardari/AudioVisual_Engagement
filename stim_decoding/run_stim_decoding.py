#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_stim_decoding.py — decode attended modality from stimulus-response tracking.

Pipeline:
  1. load raw + photodiode-aligned trials (reuses eeg_analysis alignment)
  2. clean EEG (verified montage, session-level QC/interpolation, broadband ICA,
     model-band filtering) and extract artifact-gated av-onset blocks
  3. build the speech envelope and the visual embedding for each trial
  4. LOO tracking indices (backward TRF + CCA for audio; forward TRF for visual)
  5. decode attention (per-trial + time-resolved) with permutation tests
  6. figures (tracking, accuracy, time-resolved, spatiotemporal TRF patterns)

    python stim_decoding/run_stim_decoding.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from sda import io, preprocess, stimuli, attention, viz  # noqa: E402


def banner(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stimulus-response attention decoding.")
    ap.add_argument("--config", default=os.path.join(HERE, "config_stimdec.yaml"))
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse cached cleaned trials + features (skip stages 1-3)")
    ap.add_argument("--preprocess-only", action="store_true",
                    help="clean, QC, save the continuous EEG + ledger, then stop")
    args = ap.parse_args()
    if args.preprocess_only:
        args.use_cache = False       # a cached result cannot demonstrate current QC

    cfg = io.load_config(args.config)
    if args.no_ica:
        cfg["preprocess"]["ica"]["enable"] = False
    fs = float(cfg["preprocess"]["resample_hz"])
    band = (cfg["preprocess"]["l_freq"], cfg["preprocess"]["h_freq"])
    n_times = int(round(float(cfg["block"]["seconds"]) * fs))
    subject = cfg["dataset"]["subject"]
    outdir = io.abspath(os.path.join(cfg["output"]["root"], subject))
    figdir = os.path.join(outdir, "figures")

    import pickle
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, "_cache.pkl")
    if args.use_cache and os.path.exists(cache_path):
        banner("LOAD CACHE (cleaned trials + features)")
        with open(cache_path, "rb") as f:
            C = pickle.load(f)
        trials, info, ch_names = C["trials"], C["info"], C["ch_names"]
        envelopes, visuals, vnames = C["envelopes"], C["visuals"], C["vnames"]
        recon_map = C.get("recon_map", {})
        print(f"  loaded {len(trials)} cached trials")
    else:
        banner("1/5  LOAD + ALIGN")
        raw, events, ali = io.load_raw_and_events(cfg)
        print(f"  {len(events)} usable trials; clock offset {ali.offset_s:.4f}s, "
              f"edge err {ali.match_error_ms:.2f}ms")

        audio_dir = io.abspath(cfg["dataset"]["audio_dir"])
        usable, records = [], {}
        for _, row in events.iterrows():
            t = int(row["trial"])
            wav = os.path.join(audio_dir, f"{row['audio_stim']}.wav")
            gp = io.game_record_path(cfg, t)
            if os.path.exists(wav) and gp:
                usable.append(t)
                records[t] = json.load(open(gp))
        events = events[events["trial"].isin(usable)].reset_index(drop=True)
        # per-trial provenance: which trials' av-onset was reconstructed (gap+delay)
        recon_map = dict(zip(events["trial"].astype(int),
                             events["av_reconstructed"].astype(bool)))
        print(f"  {len(usable)} trials have both audio + game records "
              f"({sum(recon_map.values())} with reconstructed av-onset)")

        banner("2/5  CLEAN + EXTRACT BLOCKS")
        raw_model, prov = preprocess.preprocess_continuous(raw, cfg)
        print(f"  cleaned: {len(raw_model.ch_names)} ch @ {raw_model.info['sfreq']:.0f}Hz; "
              f"ICA removed {len(prov.get('ica', {}).get('excluded', []))} comps")
        trials, info, ch_names = preprocess.extract_trials(raw_model, events, cfg, prov)
        qcs = prov.get("block_qc_summary", {})
        print(f"  extracted {len(trials)} artifact-gated blocks; "
              f"{qcs.get('rejected', 0)} rejected")
        min_per_condition = int(cfg["block"].get("quality", {}).get("min_trials_per_condition", 8))
        condition_counts = {label: sum(t["label"] == label for t in trials.values())
                            for label in ("Audio", "Visual")}
        if min(condition_counts.values()) < min_per_condition:
            raise RuntimeError("Too few clean blocks for cross-validated attention decoding: "
                               f"{condition_counts}; need >= {min_per_condition} per condition.")
        qc_path = os.path.join(outdir, f"{subject}_preprocessing_qc.json")
        with open(qc_path, "w") as f:
            json.dump(prov, f, indent=2)
        print(f"  preprocessing QC ledger -> {qc_path}")
        if cfg["output"].get("save_clean_raw", True):
            clean_path = os.path.join(outdir, f"{subject}_clean_stimdec_raw.fif")
            raw_model.save(clean_path, overwrite=True, verbose="ERROR")
            print(f"  cleaned continuous EEG -> {clean_path}")
        if args.preprocess_only:
            banner("PREPROCESSING COMPLETE — REVIEW QC LEDGER BEFORE DECODING")
            return

        banner("3/5  STIMULUS FEATURES")
        envelopes = {}
        for t in trials:
            wav = os.path.join(audio_dir, f"{trials[t]['audio_stim']}.wav")
            envelopes[t] = stimuli.audio_envelope(wav, fs, n_times, cfg, band)
        visuals, vnames = stimuli.build_visual_embeddings(
            {t: records[t] for t in trials}, fs, n_times, cfg, band)
        print(f"  envelopes for {len(envelopes)} trials; visual embedding = {vnames}")

        for t in list(trials.keys()):
            trials[t]["eeg"] = trials[t]["eeg"][:, :n_times]
            envelopes[t] = envelopes[t][:n_times]
            visuals[t] = visuals[t][:n_times]
        with open(cache_path, "wb") as f:
            pickle.dump({"trials": trials, "info": info, "ch_names": ch_names,
                         "envelopes": envelopes, "visuals": visuals, "vnames": vnames,
                         "recon_map": recon_map}, f)

    banner("4/5  DECODE ATTENTION (LOO tracking + permutation)")
    res, fitted = attention.run(trials, envelopes, visuals, ch_names, cfg, fs, log=print)
    res["visual_features"] = vnames
    fitted["visual_features"] = vnames
    res["subject"] = subject
    # Per-trial av-onset provenance (aligned with res["trial_ids"]) so any analysis
    # can restrict to the hardware-timed trials.
    res["av_reconstructed"] = [bool(recon_map.get(int(t), False))
                               for t in res["trial_ids"]]
    m = res["metrics"]
    m["n_av_reconstructed"] = int(sum(res["av_reconstructed"]))
    m["n_hardware_timed"] = len(res["trial_ids"]) - m["n_av_reconstructed"]
    print(f"\n  direct : acc={m['acc_direct']:.3f} AUC={m['auc_direct']:.3f} p={m['p_direct']:.4f}")
    print(f"  LDA    : acc={m['acc_lda']:.3f} AUC={m['auc_lda']:.3f} p={m['p_lda']:.4f}")
    print(f"  peak sliding-window acc: {max(res['time_resolved']['sliding_accuracy']):.3f}")

    banner("5/5  FIGURES + SAVE MODELS")
    os.makedirs(outdir, exist_ok=True)
    figs = viz.make_all(res, fitted, info, ch_names, figdir) \
        if cfg["output"].get("figures", True) else {}
    for k, v in figs.items():
        print(f"  {k}: {v}")

    # trained models (ReceptiveField encoders/decoder + CCA) for reuse/inspection
    import joblib
    models_path = os.path.join(outdir, f"{subject}_models.joblib")
    joblib.dump(fitted, models_path)
    print(f"  models -> {models_path}")

    summary = {k: v for k, v in res.items() if k != "patterns"}
    with open(os.path.join(outdir, f"{subject}_stimdec.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(outdir, f"{subject}_patterns.npz"),
                        **{k: np.asarray(v, dtype=object) for k, v in res["patterns"].items()})
    print(f"  -> {outdir}/{subject}_stimdec.json")
    banner("DONE")


if __name__ == "__main__":
    main()
