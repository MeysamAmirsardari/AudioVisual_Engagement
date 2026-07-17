#!/usr/bin/env python3
"""
End-to-end EEG pipeline: clean the recording and DECODE THE ANTICIPATORY GAP
between the attention instruction and the audiovisual stimulus onset.

    python run_eeg_pipeline.py                         # uses config_eeg.yaml
    python run_eeg_pipeline.py --vhdr <file> --behavior <json> --subject sub001
    python run_eeg_pipeline.py --apply-montage         # enable topographies
    python run_eeg_pipeline.py --no-tempgen            # skip the slow GAT matrix

Outputs land under output.root/<subject>/:
    <sub>_report.html      one self-contained report (open this first)
    <sub>_events.csv       the aligned, labelled trial table
    <sub>_metrics.json     headline decoding numbers
    <sub>_clean_raw.fif    cleaned continuous data     (optional)
    <sub>_gap-epo.fif      epoched data                (optional)
    figures/               every panel as a PNG
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# make `cma_eeg` importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cma_eeg import alignment, decoding, epoching, loading, preprocessing, reporting
from cma_eeg.utils import Paths, banner, ensure_dir, load_config, setup_logging


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "config_eeg.yaml"))
    ap.add_argument("--vhdr", default=None, help="override dataset.vhdr")
    ap.add_argument("--behavior", default=None,
                    help="override dataset.behavior (or 'auto' to discover)")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--apply-montage", action="store_true",
                    help="enable the assumed actiCAP-64 montage (topographies)")
    ap.add_argument("--no-tempgen", action="store_true",
                    help="skip the temporal-generalisation matrix (faster)")
    ap.add_argument("--no-ica", action="store_true", help="skip ICA")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(logging.WARNING if args.quiet else logging.INFO)
    cfg = load_config(args.config)

    # resolve paths relative to the repo root (config paths are repo-relative)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    def rel(p):  # noqa: E306
        return p if os.path.isabs(p) else os.path.join(repo, p)

    if args.vhdr:      cfg["dataset"]["vhdr"] = args.vhdr
    if args.behavior:  cfg["dataset"]["behavior"] = args.behavior
    if args.subject:   cfg["dataset"]["subject"] = args.subject
    if args.apply_montage: cfg["preprocess"]["apply_montage"] = True
    if args.no_ica:    cfg["preprocess"]["ica"]["enable"] = False

    subject = cfg["dataset"]["subject"]
    vhdr = rel(cfg["dataset"]["vhdr"])
    vmrk = os.path.splitext(vhdr)[0] + ".vmrk"
    paths = Paths(root=rel(cfg["output"]["root"]), subject=subject)
    ensure_dir(paths.subject_dir)

    # ---- 1. load ----------------------------------------------------------
    banner("1/6  LOAD recording + photodiode markers")
    raw = loading.load_raw(vhdr)
    markers = loading.load_markers(vmrk, raw.info["sfreq"],
                                   cfg["markers"]["edge_code"])

    # ---- 2. align to behaviour -> labels ----------------------------------
    banner("2/6  ALIGN markers to behaviour trials")
    beh = cfg["dataset"].get("behavior")
    if not beh or beh == "auto":
        beh = alignment.discover_behavior(vmrk)
    else:
        beh = rel(beh)
    ali = alignment.align(markers, beh,
                          match_tol_s=cfg["alignment"]["match_tol_s"],
                          offset_search_s=cfg["alignment"]["offset_search_s"],
                          gap_lo_s=cfg["markers"]["gap_lo_s"],
                          gap_hi_s=cfg["markers"]["gap_hi_s"])
    events = ali.events
    if len(events) < cfg["alignment"]["min_usable_trials"]:
        raise RuntimeError(f"Only {len(events)} usable trials after alignment "
                           f"(< {cfg['alignment']['min_usable_trials']}); aborting.")

    # ---- 3. preprocess / clean --------------------------------------------
    banner("3/6  CLEAN (filter, montage, bad channels, reference, ICA)")
    raw_clean, prov = preprocessing.preprocess(raw, cfg["preprocess"])
    if cfg["output"].get("save_clean_raw"):
        raw_clean.save(paths.file("clean_raw.fif"), overwrite=True, verbose="ERROR")

    # ---- 4. epoch ---------------------------------------------------------
    banner("4/6  EPOCH around gap onset")
    epochs = epoching.make_epochs(raw_clean, events, cfg["epochs"],
                                  resample_hz=cfg["preprocess"].get("resample_hz"))
    if cfg["output"].get("save_epochs"):
        epochs.save(paths.file("gap-epo.fif"), overwrite=True, verbose="ERROR")

    # ---- 5. decode --------------------------------------------------------
    banner("5/6  DECODE attended modality across the gap")
    dc = cfg["decode"]
    y = decoding.get_labels(epochs, cfg["epochs"]["event_id"]["Audio"])

    tr = decoding.time_resolved(epochs, y, dc) if dc.get("time_resolved", True) else None
    gat = None
    if dc.get("temporal_generalization", True) and not args.no_tempgen:
        gat = decoding.temporal_generalization(epochs, y, dc)
    gap_res = decoding.whole_window(epochs, y, dc, dc["gap_window"], "gap",
                                    dc.get("alpha_band", [8, 14]))
    cue_res = decoding.whole_window(epochs, y, dc, dc["cue_window"], "cue",
                                    dc.get("alpha_band", [8, 14]))
    alpha = (decoding.alpha_analysis(epochs, y, dc, dc["gap_window"])
             if dc.get("alpha_analysis", True) else None)

    # ---- 6. report --------------------------------------------------------
    banner("6/6  REPORT")
    reporting.save_results(paths, ali, events, tr, gap_res, cue_res, alpha,
                           gap_window=dc["gap_window"])
    if cfg["output"].get("report_html", True):
        reporting.build_report(paths, raw_clean, epochs, prov, ali, events,
                               tr, gat, gap_res, cue_res, alpha, cfg)

    banner("DONE")
    print(f"\nHeadline: gap-window AUC = {gap_res['auc']:.3f} "
          f"(p = {gap_res['p']:.3f}); cue-window AUC = {cue_res['auc']:.3f}.")
    print(f"Open: {paths.file('report.html')}\n")


if __name__ == "__main__":
    main()
