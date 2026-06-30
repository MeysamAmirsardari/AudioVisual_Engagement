#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_block.py — r un ONE block of the cross-modal (audio vs. visual) attention
experiment, using a stimulus library that was prepared ONCE by build_stimuli.py.

What this script does NOT do anymore: it does not synthesise audio or run any
forced alignment. All auditory stimuli + word onsets are read from the prepared
library (stims/ + stims/manifest.json). That makes every session fast, reusable,
and reproducible. (Because it never imports faster-whisper / PyAV, there is also
no PyAV-vs-ffpyplayer conflict to work around.)

Block flow
----------
  1. Instruction  — LEFT arrow = attend Audio, RIGHT arrow = attend Visual.
  2. Buffer gap   — fixation cross for exactly 1.000 s (frame-counted).
  3. Audiovisual  — muted, scaled tetris.mp4 + a selected speech clip, started
                    synchronously (PTB flip-scheduled audio onset), capped.
  4. Attention    — a yes/no probe about the ATTENDED stream (a target word for
       check         audio; a configured game question for visual), scored.
  5. Logging      — choice, AV onset, stimulus ids, probe + response + accuracy
                    written to behavior/.

Everything configurable lives in config.yaml. Run the builder first:
    python build_stimuli.py
    python run_block.py --subject sub01
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import random
import sys
import time

import cma_common as cc


# ===========================================================================
# Stimulus selection
# ===========================================================================
def select_audio_stim(manifest: dict, rng: random.Random,
                      audio_id: str | None) -> dict:
    """Pick the audio stimulus for this block (specific id, or random)."""
    items = manifest.get("audio", [])
    if not items:
        raise RuntimeError(
            "No audio stimuli in the library. Run:  python build_stimuli.py")
    if audio_id:
        for e in items:
            if e["id"] == audio_id:
                return e
        raise RuntimeError(f"audio id '{audio_id}' not found in manifest.")
    return rng.choice(items)


def _resolve(paths: cc.Paths, rel: str) -> str:
    """Resolve a manifest path (relative to the stim dir) to an absolute path."""
    return rel if os.path.isabs(rel) else os.path.join(paths.stim_dir, rel)


# ===========================================================================
# Experiment
# ===========================================================================
def run_experiment(cfg: dict, paths: cc.Paths, audio_entry: dict,
                   subject: str, block_index: int, rng: random.Random) -> None:
    """Open the PsychoPy window and run the single block + probe."""
    exp = cfg["experiment"]

    # --- Resolve + validate assets BEFORE opening any window ----------------
    audio_wav = _resolve(paths, audio_entry["audio_path"])
    words_file = _resolve(paths, audio_entry["words_json"])
    if not os.path.exists(audio_wav):
        raise FileNotFoundError(f"Missing audio clip: {audio_wav}")
    audio_duration = cc.audio_duration_seconds(audio_wav)

    # Visual stimulus mode: 'gabor' (default, fixation-locked) or 'video'.
    vis = cfg["visual"]
    visual_mode = vis.get("mode", "gabor")
    video_file = cc.abspath(vis["video"]["file"])
    if visual_mode == "video" and not os.path.exists(video_file):
        raise FileNotFoundError(f"Missing video: {video_file}")

    # Block length: 'audio' mode runs for the clip's duration (capped), else fixed.
    block_cap = float(exp["block_seconds"])
    block_target = (min(audio_duration, block_cap)
                    if exp.get("block_mode", "audio") == "audio" else block_cap)

    # --- Select the precise PTB audio backend BEFORE importing sound --------
    from psychopy import prefs
    prefs.hardware["audioLib"] = list(exp["audio_lib"])

    from psychopy import visual, core, event, logging, monitors
    from psychopy.sound import Sound
    from psychopy.constants import FINISHED

    logging.console.setLevel(logging.WARNING)

    # --- Window -------------------------------------------------------------
    mon = monitors.Monitor("expMonitor")
    win = visual.Window(fullscr=bool(exp["fullscreen"]), color=exp["bg_color"],
                        units="height", monitor=mon, allowGUI=False,
                        winType="pyglet")
    win.mouseVisible = False
    txt_color = exp["text_color"]

    # --- Frame-accurate gap: measure the true refresh rate ------------------
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                       nWarmUpFrames=10, threshold=1)
    refresh_hz = measured if measured else float(exp["fallback_refresh_hz"])
    gap_frames = int(round(float(exp["gap_seconds"]) * refresh_hz))
    cc.log(f"Refresh ~{refresh_hz:.2f} Hz -> {gap_frames} frames for the "
           f"{exp['gap_seconds']:.3f}s gap.")

    # --- Reusable visual components -----------------------------------------
    instruction = visual.TextStim(
        win,
        text=("Decide your attentional focus:\n\n"
              "Press the LEFT arrow for Audio,\n"
              "or the RIGHT arrow for Visual."),
        color=txt_color, height=0.05, wrapWidth=1.4, alignText="center")
    fixation = visual.TextStim(win, text="+", color=txt_color, height=0.12)

    # --- Pre-load audio (PTB) -----------------------------------------------
    snd = Sound(audio_wav, stereo=True, hamming=False)

    # --- Build the visual stimulus for the chosen mode ----------------------
    movie = None
    gabor = None
    gabor_fix = None
    switch_times: list[float] = []
    if visual_mode == "video":
        # Muted gameplay video, scaled to N x its native pixel size.
        movie = visual.MovieStim(win, video_file, noAudio=True, volume=0.0,
                                 loop=False, size=None)
        native_w, native_h = movie.frameSize
        if not native_w or not native_h:
            native_w, native_h = 640, 360
        scale = float(vis["video"].get("scale", 1.8))
        movie.size = (native_w * scale, native_h * scale)
        cc.log(f"Video native {native_w}x{native_h}px -> "
               f"{int(native_w * scale)}x{int(native_h * scale)}px (x{scale}).")
    else:
        # Centred Gabor patch: a gaussian-masked sinusoidal grating. It rotates
        # at a fixed speed and reverses direction at random scheduled times; the
        # attend-visual task is to count those reversals. Staying at fixation
        # avoids the eye-movement confound of a moving video.
        g = vis["gabor"]
        gsize = float(g.get("size", 0.16))
        gabor = visual.GratingStim(
            win, tex="sin", mask="gauss", units="height", pos=(0, 0),
            size=gsize, sf=float(g.get("cycles", 6.0)) / gsize,
            contrast=float(g.get("contrast", 1.0)), ori=0.0)
        if g.get("show_fixation", True):
            gabor_fix = visual.Circle(win, radius=0.004, units="height",
                                      fillColor=txt_color, lineColor=txt_color,
                                      pos=(0, 0))
        switch_times = _gen_switch_times(
            block_target, float(g.get("switch_min_interval_s", 1.2)),
            float(g.get("switch_max_interval_s", 3.5)), rng)
        cc.log(f"Gabor patch (size {gsize}h): {len(switch_times)} reversal(s) "
               f"scheduled over {block_target:.1f}s.")

    global_clock = core.Clock()

    # --- Optional webcam recording (separate process; see cma_webcam.py) -----
    # Started here so the camera is warmed up before the timing-critical block.
    webcam, webcam_info = _start_webcam(cfg, subject, block_index, win, visual,
                                        core, txt_color)

    # ---------------------------------------------------------------
    # STEP 1 — Instruction, wait for LEFT / RIGHT
    # ---------------------------------------------------------------
    instruction.draw()
    win.flip()
    event.clearEvents()
    keys = event.waitKeys(keyList=["left", "right", "escape"])
    if "escape" in keys:
        _abort(win, core)
    choice_key = keys[0]
    attended = "Audio" if choice_key == "left" else "Visual"
    cc.log(f"Subject chose: {choice_key.upper()} -> attend {attended}")

    # ---------------------------------------------------------------
    # STEP 2 — 1.0 s buffer gap (frame-counted = frame-accurate)
    # ---------------------------------------------------------------
    for _ in range(gap_frames):
        fixation.draw()
        win.flip()

    # ---------------------------------------------------------------
    # STEP 3 — Synchronous, capped audiovisual playback
    # ---------------------------------------------------------------
    try:
        when_ptb = win.getFutureFlipTime(clock="ptb")   # next flip, ptb timebase
        snd.play(when=when_ptb)                           # sample-accurate schedule
    except Exception as exc:
        cc.log(f"Scheduled audio unavailable ({exc}); starting immediately.")
        snd.play()

    if movie is not None:
        movie.play()
    block_clock = core.Clock()
    flip_t = win.flip()                                  # <- AV onset
    block_clock.reset()
    av_onset_global = global_clock.getTime()
    onset_wall_clock = _dt.datetime.now().isoformat(timespec="milliseconds")
    if webcam is not None:
        # Wall-clock anchor for aligning webcam frames (in <name>.frames.csv) to
        # the audiovisual onset.
        webcam_info["av_onset_wallclock"] = time.time()
    cc.log(f"AUDIOVISUAL ONSET at t={av_onset_global:.4f}s "
           f"(audio {audio_duration:.1f}s, running {block_target:.1f}s).")

    aborted = False
    gabor_switches = 0          # reversals actually presented (ground truth)
    if visual_mode == "video":
        while movie.status != FINISHED and block_clock.getTime() < block_target:
            movie.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                aborted = True
                break
    else:
        # Rotate the Gabor at a fixed speed, reversing direction at each scheduled
        # switch time; count the reversals as they occur (the ground truth).
        speed = float(vis["gabor"].get("rotation_speed_dps", 75.0))
        direction = 1
        sw_idx = 0
        prev_t = 0.0
        while block_clock.getTime() < block_target:
            t = block_clock.getTime()
            dt = t - prev_t
            prev_t = t
            while sw_idx < len(switch_times) and t >= switch_times[sw_idx]:
                direction *= -1
                gabor_switches += 1
                sw_idx += 1
            gabor.ori += direction * speed * dt      # time-based -> smooth
            gabor.draw()
            if gabor_fix is not None:
                gabor_fix.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                aborted = True
                break
    block_duration = block_clock.getTime()

    try:
        snd.stop()
    except Exception:
        pass
    if movie is not None:
        try:
            movie.stop()
        except Exception:
            pass
    cc.log(f"Block ended at {block_duration:.3f}s"
           + (f"; {gabor_switches} reversals shown." if visual_mode == "gabor" else "."))

    # ---------------------------------------------------------------
    # STEP 4 — Attention-check probe (about the ATTENDED stream)
    # ---------------------------------------------------------------
    probe_result = {"modality": attended, "probe_id": None, "question": None,
                    "correct_answer": None, "response": None, "rt_s": None,
                    "correct": None}
    if not aborted:
        if attended == "Audio":
            probe_result = _probe_audio(win, event, core, cfg, audio_entry, rng,
                                        txt_color)
        elif visual_mode == "gabor":
            # Count probe: scored against the reversals we actually presented.
            probe_result = _probe_count(win, event, core, cfg, gabor_switches,
                                        txt_color)
        else:
            probe_result = _probe_visual(win, event, core, cfg, rng, txt_color)
        if probe_result.get("aborted"):
            aborted = True

    # --- Stop the webcam recorder (captures the block + the probe) -----------
    if webcam is not None:
        st = webcam.stop() or {}
        webcam_info.update(opened=st.get("opened", webcam_info["opened"]),
                           frames=st.get("frames"),
                           duration_s=st.get("duration_s"))
        cc.log(f"Webcam stopped: {st.get('frames')} frames, "
               f"{st.get('duration_s')}s -> {webcam_info['file']}")

    # ---------------------------------------------------------------
    # STEP 5 — Behavioural logging
    # ---------------------------------------------------------------
    _write_behavior(paths, {
        "subject": subject,
        "block": block_index,
        "timestamp": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "attended_modality": attended,
        "choice_key": choice_key,
        "audio_stim_id": audio_entry["id"],
        "audio_source": audio_entry.get("source"),
        "speaker": audio_entry.get("speaker"),
        "visual_mode": visual_mode,
        "visual_stim": ("gabor" if visual_mode == "gabor"
                        else os.path.relpath(video_file, cc.BASE_DIR)),
        "gabor_switches": gabor_switches if visual_mode == "gabor" else None,
        "word_boundary_file": os.path.relpath(words_file, cc.BASE_DIR),
        "av_onset_global_s": round(av_onset_global, 4),
        "av_onset_wall_clock": onset_wall_clock,
        "audio_duration_s": round(audio_duration, 3),
        "block_target_s": round(block_target, 3),
        "block_duration_s": round(block_duration, 3),
        "refresh_hz": round(refresh_hz, 3),
        "gap_frames": gap_frames,
        "probe_modality": probe_result["modality"],
        "probe_id": probe_result["probe_id"],
        "probe_question": probe_result["question"],
        "probe_correct_answer": probe_result["correct_answer"],
        "probe_response": probe_result["response"],
        "probe_rt_s": probe_result["rt_s"],
        "probe_correct": probe_result["correct"],
        "probe_abs_error": probe_result.get("abs_error"),
        "webcam_enabled": webcam_info["enabled"],
        "webcam_file": webcam_info["file"],
        "webcam_opened": webcam_info["opened"],
        "webcam_frames": webcam_info["frames"],
        "webcam_duration_s": webcam_info["duration_s"],
        "webcam_start_wallclock": webcam_info["start_wallclock"],
        "webcam_av_onset_wallclock": webcam_info["av_onset_wallclock"],
        "aborted": aborted,
    })

    # --- Goodbye ------------------------------------------------------------
    visual.TextStim(win, text="Block complete.\nThank you.",
                    color=txt_color, height=0.06).draw()
    win.flip()
    core.wait(1.5)
    win.close()
    core.quit()


# ===========================================================================
# Probe presentation (yes/no)
# ===========================================================================
def _key_label(key: str) -> str:
    return {"left": "←", "right": "→", "up": "↑",
            "down": "↓"}.get(key, key.upper())


def _run_yes_no(win, event, core, cfg, question: str, correct_present,
                txt_color) -> dict:
    """
    Show a yes/no question and collect a response.

    `correct_present` is True/False (whether the correct answer is "yes") or
    None when ground truth is unavailable (then `correct` is logged as None).
    """
    from psychopy import visual
    keys_cfg = cfg["probe"]["keys"]
    yes_key, no_key = keys_cfg["yes"], keys_cfg["no"]
    quit_key = cfg["probe"].get("quit_key", "escape")
    header = cfg["probe"].get("instruction", "")

    prompt = (f"{header}\n\n{question}\n\n"
              f"YES ({_key_label(yes_key)})        NO ({_key_label(no_key)})")
    visual.TextStim(win, text=prompt, color=txt_color, height=0.05,
                    wrapWidth=1.5, alignText="center").draw()
    win.flip()
    event.clearEvents()
    clock = core.Clock()
    pressed = event.waitKeys(keyList=[yes_key, no_key, quit_key],
                             timeStamped=clock)
    key, rt = pressed[0]
    if key == quit_key:
        return {"response": None, "rt_s": None, "correct": None, "aborted": True}

    response = "yes" if key == yes_key else "no"
    correct = (None if correct_present is None
               else (response == "yes") == bool(correct_present))
    return {"response": response, "rt_s": round(rt, 4), "correct": correct,
            "aborted": False}


def _probe_audio(win, event, core, cfg, audio_entry, rng, txt_color) -> dict:
    """Yes/no 'was WORD spoken?' probe for the attended-audio condition."""
    probes = audio_entry.get("probes", [])
    if not probes:
        return {"modality": "Audio", "probe_id": None, "question": None,
                "correct_answer": None, "response": None, "rt_s": None,
                "correct": None}
    p = rng.choice(probes)
    res = _run_yes_no(win, event, core, cfg, p["question"], p["present"], txt_color)
    return {"modality": "Audio", "probe_id": p["target_word"],
            "question": p["question"],
            "correct_answer": "yes" if p["present"] else "no",
            "response": res["response"], "rt_s": res["rt_s"],
            "correct": res["correct"], "aborted": res.get("aborted", False)}


def _probe_visual(win, event, core, cfg, rng, txt_color) -> dict:
    """Yes/no game question for the attended-visual VIDEO condition (config)."""
    vprobes = cfg["visual"]["video"].get("probes", [])
    usable = [p for p in vprobes if p.get("correct") in (True, False)]
    if usable:
        p = rng.choice(usable)
        correct_present = bool(p["correct"])
    elif vprobes:
        # Ground truth not filled in config -> still ask, but log unscored.
        p = vprobes[0]
        correct_present = None
        cc.log("WARNING: no video probe has 'correct' set in config; logging "
               "the response UNSCORED. Fill visual.video.probes[].correct.")
    else:
        return {"modality": "Visual", "probe_id": None, "question": None,
                "correct_answer": None, "response": None, "rt_s": None,
                "correct": None}
    res = _run_yes_no(win, event, core, cfg, p["question"], correct_present,
                      txt_color)
    ca = (None if correct_present is None else ("yes" if correct_present else "no"))
    return {"modality": "Visual", "probe_id": p.get("id"),
            "question": p["question"], "correct_answer": ca,
            "response": res["response"], "rt_s": res["rt_s"],
            "correct": res["correct"], "aborted": res.get("aborted", False)}


# ===========================================================================
# Gabor reversal schedule + count probe
# ===========================================================================
def _gen_switch_times(duration: float, min_iv: float, max_iv: float,
                      rng: random.Random) -> list[float]:
    """
    Random reversal onset times (seconds from block onset) for the Gabor: draw
    inter-reversal intervals uniformly from [min_iv, max_iv] until `duration`.
    Reproducible given the runner's seed.
    """
    times: list[float] = []
    t = rng.uniform(min_iv, max_iv)
    while t < duration:
        times.append(round(t, 4))
        t += rng.uniform(min_iv, max_iv)
    return times


def _collect_number(win, event, core, cfg, prompt: str, txt_color):
    """
    Simple on-screen numeric entry (digits, backspace, ENTER to submit). Returns
    (value_int_or_None, rt_seconds_or_None, aborted_bool).
    """
    from psychopy import visual
    quit_key = cfg["probe"].get("quit_key", "escape")
    digit_keys = [str(d) for d in range(10)] + [f"num_{d}" for d in range(10)]
    accept = digit_keys + ["backspace", "return", "num_enter", quit_key]

    entered = ""
    event.clearEvents()
    clock = core.Clock()
    while True:
        shown = entered if entered else "_"
        visual.TextStim(
            win, color=txt_color, height=0.05, wrapWidth=1.6, alignText="center",
            text=f"{prompt}\n\nType the number, then press ENTER:\n\n{shown}"
        ).draw()
        win.flip()
        key, rt = event.waitKeys(keyList=accept, timeStamped=clock)[0]
        if key == quit_key:
            return None, None, True
        if key in ("return", "num_enter"):
            if entered:
                return int(entered), round(rt, 4), False
        elif key == "backspace":
            entered = entered[:-1]
        else:
            entered = (entered + key.replace("num_", ""))[:3]   # cap at 3 digits


def _probe_count(win, event, core, cfg, true_count: int, txt_color) -> dict:
    """Numeric 'how many reversals?' probe for the attended-Gabor condition."""
    question = cfg["visual"]["gabor"].get(
        "probe_question",
        "How many times did the grating reverse its rotation direction?")
    value, rt, aborted = _collect_number(win, event, core, cfg, question, txt_color)
    correct = None if value is None else (value == true_count)
    abs_err = None if value is None else abs(value - true_count)
    return {"modality": "Visual", "probe_id": "gabor_switch_count",
            "question": question, "correct_answer": str(true_count),
            "response": (None if value is None else str(value)),
            "rt_s": rt, "correct": correct, "abs_error": abs_err,
            "aborted": aborted}


# ===========================================================================
# Webcam
# ===========================================================================
def _start_webcam(cfg, subject, block_index, win, visual, core, txt_color):
    """
    Start the webcam recorder subprocess if `webcam.enabled`. Returns a tuple
    (controller_or_None, info_dict). The recorder runs in its own process, so
    it never touches this process's draw-loop timing.
    """
    wcfg = cfg.get("webcam", {}) or {}
    info = {"enabled": bool(wcfg.get("enabled")), "file": None, "opened": False,
            "frames": None, "duration_s": None, "start_wallclock": None,
            "av_onset_wallclock": None}
    if not wcfg.get("enabled"):
        return None, info

    from cma_webcam import WebcamController
    wdir = cc.abspath(wcfg.get("dir", "webcam"))
    os.makedirs(wdir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(wdir, f"{subject}_{ts}_b{block_index}."
                                  f"{wcfg.get('container', 'mp4')}")
    res = wcfg.get("resolution") or None
    ctrl = WebcamController(
        out_path,
        device=int(wcfg.get("device", 0)),
        fps=float(wcfg.get("fps", 0) or 0),
        resolution=tuple(res) if res else None,
        fourcc=wcfg.get("fourcc", "mp4v"),
        max_seconds=float(wcfg.get("max_seconds", 600)),
    )
    info["file"] = os.path.relpath(out_path, cc.BASE_DIR)

    # Brief on-screen note while the camera warms up (can take ~1-2 s).
    visual.TextStim(win, text="Preparing camera…", color=txt_color,
                    height=0.05).draw()
    win.flip()

    opened = ctrl.start()
    info["opened"] = opened
    info["start_wallclock"] = ctrl.start_wallclock
    if opened:
        cc.log(f"Webcam recording -> {out_path}")
        return ctrl, info

    # Failed to open: abort if required, otherwise carry on without recording.
    st = ctrl.read_status() or {}
    msg = st.get("error", "camera did not open in time")
    ctrl.stop()
    if wcfg.get("required"):
        win.close()
        core.quit()
        raise RuntimeError(f"Webcam is required but failed to start: {msg}")
    cc.log(f"WARNING: webcam not recording ({msg}); continuing without it.")
    return None, info


# ===========================================================================
# Behaviour logging + helpers
# ===========================================================================
def _write_behavior(paths: cc.Paths, record: dict) -> None:
    """Append a row to behavior/<subject>.csv and drop a per-block JSON."""
    os.makedirs(paths.behavior_dir, exist_ok=True)
    subject = record["subject"]
    ts = record["timestamp"]

    json_path = os.path.join(paths.behavior_dir, f"{subject}_{ts}_b{record['block']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(paths.behavior_dir, f"{subject}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            w.writeheader()
        w.writerow(record)

    acc = record["probe_correct"]
    acc_s = "n/a" if acc is None else ("CORRECT" if acc else "incorrect")
    cc.log(f"Behaviour logged ({acc_s}) -> {csv_path}")
    cc.log(f"                        -> {json_path}")


def _abort(win, core):
    cc.log("Aborted by user.")
    win.close()
    core.quit()
    sys.exit(0)


# ===========================================================================
# Orchestration
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Run one cross-modal attention block.")
    ap.add_argument("--subject", default="sub01", help="Participant id.")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--block", type=int, default=1, help="Block index (logging).")
    ap.add_argument("--audio-id", default=None,
                    help="Use a specific audio clip id (default: random).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for stimulus/probe randomisation.")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    paths = cc.Paths.from_config(cfg).ensure()
    manifest = cc.read_manifest(paths.manifest)

    seed = args.seed if args.seed is not None else _dt.datetime.now().microsecond
    rng = random.Random(seed)

    audio_entry = select_audio_stim(manifest, rng, args.audio_id)
    cc.log(f"Selected audio stimulus: {audio_entry['id']} "
           f"({audio_entry['duration_s']:.1f}s, speaker {audio_entry.get('speaker')}).")

    run_experiment(cfg, paths, audio_entry, args.subject, args.block, rng)


if __name__ == "__main__":
    main()
