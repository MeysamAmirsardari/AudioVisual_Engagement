#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — the experiment launcher.

Collects participant / session metadata, creates a clean per-subject/per-session
output directory, records everything for provenance, and then runs the session.

Everything a session produces is organised under:

    data/<subject>/<session>/
        session_info.json     participant + session metadata (this launcher)
        behavior/             <subject>.csv + per-session JSON (trial data)
        games/                Tetris reconstruction records (-> mp4 later)
        webcam/               webcam recording(s)

The shared stimulus library (stims/) is NOT per-session.

Usage
-----
    python run.py                                  # GUI dialog for the metadata
    python run.py --subject P01 --session 01       # prefilled GUI
    python run.py --subject P01 --session 01 --no-gui --age 24 --sex F \
                  --handedness right --experimenter MA
    python run.py --subject P01 --session 01 --trials 4 --no-gui   # short test

`--seed` overrides the fixed config seed; otherwise the whole session is
reproducible from config.experiment.seed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys

import cma_common as cc

# The demographic fields collected for each session (besides subject/session).
_FIELDS = ["age", "sex", "handedness", "experimenter", "notes"]
_SEX_CHOICES = ["", "female", "male", "other", "prefer not to say"]
_HAND_CHOICES = ["", "right", "left", "ambidextrous"]


def _git_commit() -> str | None:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=cc.BASE_DIR)
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_metadata(args) -> dict:
    """Gather metadata from CLI args, optionally confirmed via a GUI dialog."""
    info = {"subject": args.subject or "", "session": args.session or "",
            "age": args.age or "", "sex": args.sex or "",
            "handedness": args.handedness or "",
            "experimenter": args.experimenter or "", "notes": args.notes or ""}

    if not args.no_gui:
        try:
            from psychopy import gui
            dlg_dict = dict(info)
            # Present sex/handedness as dropdowns (selected default first).
            dlg_dict["sex"] = _ordered_choices(_SEX_CHOICES, info["sex"])
            dlg_dict["handedness"] = _ordered_choices(_HAND_CHOICES,
                                                      info["handedness"])
            dlg = gui.DlgFromDict(
                dlg_dict, title="Load-modulation experiment — session info",
                order=["subject", "session", "age", "sex", "handedness",
                       "experimenter", "notes"])
            if not dlg.OK:
                print("Cancelled at the info dialog.")
                sys.exit(0)
            info.update({k: (v if not isinstance(v, list) else v[0])
                         for k, v in dlg_dict.items()})
        except Exception as exc:
            print(f"GUI dialog unavailable ({exc}); using command-line values.")

    info["subject"] = str(info["subject"]).strip()
    info["session"] = str(info["session"]).strip()
    if not info["subject"] or not info["session"]:
        sys.exit("ERROR: subject and session are required "
                 "(pass --subject/--session or fill the dialog).")
    return info


def _ordered_choices(choices: list, selected: str) -> list:
    """Return the dropdown list with `selected` moved to the front (default)."""
    if selected and selected in choices:
        return [selected] + [c for c in choices if c != selected]
    return list(choices)


def make_session_dir(cfg: dict, subject: str, session: str, force: bool) -> str:
    """Create (and return) data/<subject>/<session>/, refusing to overwrite."""
    data_root = cc.abspath(cfg["paths"].get("data_root", "data"))
    session_dir = os.path.join(data_root, subject, session)
    if os.path.isdir(session_dir) and os.listdir(session_dir) and not force:
        sys.exit(f"ERROR: {session_dir} already exists and is not empty.\n"
                 f"Use a different --session, or pass --force to add to it.")
    for sub in ("behavior", "games", "webcam"):
        os.makedirs(os.path.join(session_dir, sub), exist_ok=True)
    return session_dir


def write_session_info(session_dir: str, info: dict, seed: int, cfg: dict,
                       started: str) -> str:
    """Write session_info.json with metadata + provenance."""
    import platform
    try:
        import psychopy
        psychopy_ver = psychopy.__version__
    except Exception:
        psychopy_ver = None

    payload = {
        **info,
        "started": started,
        "seed": seed,
        "n_trials": int(cfg["experiment"].get("n_trials")),
        "visual_mode": cfg["visual"].get("mode"),
        "config_file": os.path.relpath(cc.DEFAULT_CONFIG, cc.BASE_DIR),
        "git_commit": _git_commit(),
        "software": {"python": platform.python_version(),
                     "psychopy": psychopy_ver, "platform": platform.platform()},
        "session_dir": os.path.relpath(session_dir, cc.BASE_DIR),
    }
    out = os.path.join(session_dir, "session_info.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch a cross-modal attention session.")
    ap.add_argument("--subject", default=None, help="Participant id (required).")
    ap.add_argument("--session", default=None, help="Session id, e.g. 01 (required).")
    ap.add_argument("--age", default=None)
    ap.add_argument("--sex", default=None, choices=None)
    ap.add_argument("--handedness", default=None)
    ap.add_argument("--experimenter", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--no-gui", action="store_true",
                    help="Skip the info dialog; use command-line values.")
    ap.add_argument("--force", action="store_true",
                    help="Write into an existing, non-empty session directory.")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--trials", type=int, default=None, help="Override n_trials.")
    ap.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    info = collect_metadata(args)
    subject, session = info["subject"], info["session"]

    # --- Per-session output directory ---------------------------------------
    session_dir = make_session_dir(cfg, subject, session, args.force)
    # Route all outputs into the session directory (stims/ stays shared).
    cfg["paths"]["behavior_dir"] = os.path.join(session_dir, "behavior")
    cfg["paths"]["games_dir"] = os.path.join(session_dir, "games")
    cfg.setdefault("webcam", {})["dir"] = os.path.join(session_dir, "webcam")

    if args.trials is not None:
        cfg["experiment"]["n_trials"] = args.trials
    seed = (args.seed if args.seed is not None
            else int(cfg["experiment"].get("seed", 0)))

    started = _dt.datetime.now().isoformat(timespec="seconds")
    info_path = write_session_info(session_dir, info, seed, cfg, started)

    cc.log(f"Subject '{subject}', session '{session}' -> "
           f"{os.path.relpath(session_dir, cc.BASE_DIR)}")
    cc.log(f"Session info: {os.path.relpath(info_path, cc.BASE_DIR)}")

    # --- Run the session (imported here so the metadata step stays lightweight)
    import run_block
    paths = cc.Paths.from_config(cfg).ensure()
    manifest = cc.read_manifest(paths.manifest)
    rng = random.Random(seed)
    run_block.run_session(cfg, paths, manifest, subject, rng, seed, session)


if __name__ == "__main__":
    main()
