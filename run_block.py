#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_block.py — run a SESSION of the cross-modal (audio vs. visual) attention
experiment: many trials back-to-back, using a stimulus library prepared once by
build_stimuli.py.

Per trial
---------
  1. Instruction  — LEFT arrow = attend Audio, RIGHT arrow = attend Visual.
  2. Buffer gap   — fixation cross for exactly `gap_seconds` (frame-counted).
  3. Audiovisual  — a visual stimulus + a speech clip, started synchronously
                    (PTB flip-scheduled audio onset), run for `block_seconds`.
                    In video mode each trial seeks to a RANDOM offset, so every
                    trial shows a different ~block_seconds cut of the clip.
  4. Attention    — a probe about the ATTENDED stream (yes/no "was WORD spoken?"
       check         for audio; yes/no game question or Gabor reversal count for
                    visual), scored.
  5. Logging      — one CSV row per trial in behavior/<subject>.csv, plus a
                    per-session JSON.

The webcam (if enabled) records the WHOLE session as one file; each trial logs
its own audiovisual-onset wall-clock for later alignment.

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
# Trial planning helpers
# ===========================================================================
def _assign_audio(items: list[dict], n_trials: int,
                  rng: random.Random) -> list[dict]:
    """
    Assign an audio clip to each of `n_trials` trials, balanced: repeatedly
    shuffle the whole library and concatenate, so each clip is used about
    equally often and no clip repeats until all have appeared.
    """
    plan: list[dict] = []
    while len(plan) < n_trials:
        block = items[:]
        rng.shuffle(block)
        plan.extend(block)
    return plan[:n_trials]


def _resolve(paths: cc.Paths, rel: str) -> str:
    """Resolve a manifest path (relative to the stim dir) to an absolute path."""
    return rel if os.path.isabs(rel) else os.path.join(paths.stim_dir, rel)


def _probe_video_duration(path: str) -> float:
    """Video duration in seconds via ffprobe (fallback when MovieStim can't say)."""
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _assign_focus(n_trials: int, rng: random.Random) -> list[str]:
    """Balanced 50/50 Audio/Visual assignment, shuffled (for focus_mode: auto)."""
    half = n_trials // 2
    seq = ["Audio"] * half + ["Visual"] * (n_trials - half)
    rng.shuffle(seq)
    return seq


def _load_cut_ranges(cfg: dict, vid_dur: float, block_cap: float):
    """
    Valid [lo, hi] start-offset ranges for random video cuts. Prefers the
    black-and-white game segments (visual.video.segments_file); falls back to the
    whole clip. Returns (ranges, source_label).
    """
    vcfg = cfg["visual"]["video"]
    min_gap = float(vcfg.get("min_gap_from_end_s", 0.5))
    seg_rel = vcfg.get("segments_file")
    seg_path = cc.abspath(seg_rel) if seg_rel else ""
    if seg_path and os.path.exists(seg_path):
        try:
            with open(seg_path, encoding="utf-8") as f:
                segs = json.load(f).get("segments", [])
            ranges = [(float(s["start_s"]), float(s["end_s"]) - block_cap)
                      for s in segs if float(s["end_s"]) - block_cap >= float(s["start_s"])]
            if ranges:
                return ranges, f"segments({len(ranges)})"
            cc.log("WARNING: no game segment is long enough for a block; "
                   "falling back to whole-clip cuts.")
        except Exception as exc:
            cc.log(f"Could not read segments file ({exc}); whole-clip cuts.")
    hi = max(0.0, vid_dur - block_cap - min_gap)
    return [(0.0, hi)], "whole-clip"


def _sample_offset(ranges: list, rng: random.Random) -> float:
    """Sample a start offset uniformly across the union of ranges (length-weighted)."""
    weights = [hi - lo for lo, hi in ranges]
    total = sum(weights)
    if total <= 0:                       # only point ranges (segment == block)
        return rng.choice(ranges)[0] if ranges else 0.0
    r = rng.uniform(0.0, total)
    acc = 0.0
    for (lo, hi), w in zip(ranges, weights):
        acc += w
        if r <= acc:
            return rng.uniform(lo, hi)
    return ranges[-1][1]


# ===========================================================================
# Session
# ===========================================================================
def run_session(cfg: dict, paths: cc.Paths, manifest: dict, subject: str,
                rng: random.Random, seed: int) -> None:
    """Open the PsychoPy window ONCE and run all trials back-to-back."""
    exp = cfg["experiment"]
    vis = cfg["visual"]
    visual_mode = vis.get("mode", "video")
    n_trials = int(exp.get("n_trials", 1))
    block_cap = float(exp["block_seconds"])
    block_mode = exp.get("block_mode", "fixed")
    iti = float(exp.get("inter_trial_interval_s", 0.8))
    focus_mode = exp.get("focus_mode", "auto")
    cue_seconds = float(exp.get("cue_seconds", 1.5))
    min_word_len = int(cfg["probe"]["audio"]["min_word_len"])

    # --- Validate the library + assets BEFORE opening any window ------------
    audio_items = manifest.get("audio", [])
    if not audio_items:
        raise RuntimeError(
            "No audio stimuli in the library. Run:  python build_stimuli.py")
    video_file = cc.abspath(vis["video"]["file"])
    if visual_mode == "video" and not os.path.exists(video_file):
        raise FileNotFoundError(f"Missing video: {video_file}")

    # Per-trial plans: which audio clip, and (auto mode) which stream to attend.
    audio_plan = _assign_audio(audio_items, n_trials, rng)
    focus_plan = (_assign_focus(n_trials, rng) if focus_mode == "auto"
                  else [None] * n_trials)

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
           f"{exp['gap_seconds']:.3f}s gap. Session: {n_trials} trials of "
           f"{block_cap:.0f}s ({visual_mode}).")

    # --- Reusable visual components -----------------------------------------
    instruction = visual.TextStim(
        win,
        text=("Decide your attentional focus:\n\n"
              "Press the LEFT arrow for Audio,\n"
              "or the RIGHT arrow for Visual."),
        color=txt_color, height=0.05, wrapWidth=1.4, alignText="center")
    fixation = visual.TextStim(win, text="+", color=txt_color, height=0.12)
    counter = visual.TextStim(win, text="", color=txt_color, height=0.03,
                              pos=(0, -0.42))

    # --- Photodiode trigger square (shown for the whole gap) ----------------
    trigger = None
    tcfg = cfg.get("trigger", {}) or {}
    if tcfg.get("enabled"):
        size_px = float(tcfg.get("size_px", 100))
        ww, wh = win.size
        sx, sy = ww / 2.0 - size_px / 2.0, wh / 2.0 - size_px / 2.0
        corner = tcfg.get("corner", "bottom-right")
        pos = {"bottom-right": (sx, -sy), "bottom-left": (-sx, -sy),
               "top-right": (sx, sy), "top-left": (-sx, sy)}.get(corner, (sx, -sy))
        trigger = visual.Rect(win, width=size_px, height=size_px, units="pix",
                              pos=pos, fillColor=tcfg.get("color", "white"),
                              lineColor=tcfg.get("color", "white"))
        cc.log(f"Trigger square {int(size_px)}px at {corner}; on during the gap "
               f"(rising edge = gap onset, falling edge = AV onset).")

    # --- Load the movie ONCE (video mode); each trial seeks a random cut -----
    movie = None
    vid_dur = 0.0
    if visual_mode == "video":
        # autoStart=False so playback only begins when we call play() at onset.
        movie = visual.MovieStim(win, video_file, noAudio=True, volume=0.0,
                                 loop=False, size=None, autoStart=False)
        native_w, native_h = movie.frameSize
        if not native_w or not native_h:
            native_w, native_h = 640, 360
        scale = float(vis["video"].get("scale", 1.8))
        movie.size = (native_w * scale, native_h * scale)
        vid_dur = float(getattr(movie, "duration", 0) or 0) \
            or _probe_video_duration(video_file)
        cc.log(f"Video {native_w}x{native_h}px x{scale}, duration {vid_dur:.1f}s "
               f"-> random {block_cap:.0f}s cuts per trial.")

    # Valid start-offset ranges for random cuts (game segments if available).
    cut_ranges, cut_source = ((_load_cut_ranges(cfg, vid_dur, block_cap))
                              if visual_mode == "video" else ([], "n/a"))
    if visual_mode == "video":
        usable = sum(hi - lo for lo, hi in cut_ranges)
        cc.log(f"Random cuts drawn from {cut_source} "
               f"({usable:.0f}s of valid start range).")

    global_clock = core.Clock()

    # --- Session webcam (ONE recording for the whole session) ---------------
    session_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    webcam, webcam_info = _start_webcam(cfg, subject, f"{session_ts}_session",
                                        win, visual, core, txt_color)

    # --- Audio Sound cache (avoid reloading the same clip repeatedly) -------
    snd_cache: dict[str, object] = {}

    def get_sound(entry):
        wav = _resolve(paths, entry["audio_path"])
        if wav not in snd_cache:
            snd_cache[wav] = Sound(wav, stereo=True, hamming=False)
        return snd_cache[wav]

    # --- Random cut offset for a trial (from the valid game segments) -------
    def video_offset() -> float:
        if not (visual_mode == "video"
                and vis["video"].get("random_start", True) and vid_dur > 0):
            return 0.0
        return _sample_offset(cut_ranges, rng)

    # -----------------------------------------------------------------------
    # One trial (closure over the shared session state above)
    # -----------------------------------------------------------------------
    def run_one_trial(trial_idx: int, entry: dict, assigned_focus,
                      vid_offset: float) -> dict:
        audio_wav = _resolve(paths, entry["audio_path"])
        words_file = _resolve(paths, entry["words_json"])
        audio_duration = float(entry.get("duration_s")
                               or cc.audio_duration_seconds(audio_wav))
        btarget = (min(audio_duration, block_cap)
                   if block_mode == "audio" else block_cap)
        snd = get_sound(entry)

        # Per-trial visual setup
        gabor = gabor_fix = None
        switch_times: list[float] = []
        if visual_mode == "gabor":
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
                btarget, float(g.get("switch_min_interval_s", 1.2)),
                float(g.get("switch_max_interval_s", 3.5)), rng)
        elif movie is not None:
            # Seek to the random offset now (pre-onset), so the decode latency is
            # paid before the timing-critical block starts.
            try:
                movie.seek(vid_offset)
            except Exception as exc:
                cc.log(f"Movie seek to {vid_offset:.1f}s failed ({exc}).")

        # Defaults (also cover the escape-before-block case)
        choice_key = attended = None
        aborted = False
        gabor_switches = 0
        gap_onset_wallclock = None
        av_onset_global = onset_wall = av_onset_wallclock = None
        block_duration = 0.0
        probe = {"modality": None, "probe_id": None, "question": None,
                 "correct_answer": None, "response": None, "rt_s": None,
                 "correct": None}

        # STEP 1 — focus: auto = cue the assigned stream; manual = subject chooses
        counter.text = f"Trial {trial_idx} / {n_trials}"
        if focus_mode == "auto":
            attended = assigned_focus
            choice_key = "auto"
            visual.TextStim(win, text=f"Attend to the {attended.upper()}",
                            color=txt_color, height=0.06).draw()
            counter.draw()
            win.flip()
            event.clearEvents()
            got = event.waitKeys(maxWait=cue_seconds, keyList=["escape"])
            if got and "escape" in got:
                aborted = True
        else:
            instruction.draw()
            counter.draw()
            win.flip()
            event.clearEvents()
            keys = event.waitKeys(keyList=["left", "right", "escape"])
            if "escape" in keys:
                aborted = True
            else:
                choice_key = keys[0]
                attended = "Audio" if choice_key == "left" else "Visual"

        if not aborted:
            # STEP 2 — frame-counted gap (photodiode trigger square shown here)
            for gi in range(gap_frames):
                fixation.draw()
                if trigger is not None:
                    trigger.draw()
                win.flip()
                if gi == 0:
                    gap_onset_wallclock = time.time()

            # STEP 3 — synchronous onset
            try:
                snd.play(when=win.getFutureFlipTime(clock="ptb"))
            except Exception as exc:
                cc.log(f"Scheduled audio unavailable ({exc}); immediate start.")
                snd.play()
            if movie is not None:
                movie.play()
            block_clock = core.Clock()
            win.flip()                                   # <- AV onset
            block_clock.reset()
            av_onset_global = round(global_clock.getTime(), 4)
            onset_wall = _dt.datetime.now().isoformat(timespec="milliseconds")
            av_onset_wallclock = time.time()

            if visual_mode == "video":
                while movie.status != FINISHED and block_clock.getTime() < btarget:
                    movie.draw()
                    win.flip()
                    if event.getKeys(keyList=["escape"]):
                        aborted = True
                        break
            else:
                speed = float(vis["gabor"].get("rotation_speed_dps", 75.0))
                direction, sw_idx, prev_t = 1, 0, 0.0
                while block_clock.getTime() < btarget:
                    t = block_clock.getTime()
                    dt = t - prev_t
                    prev_t = t
                    while sw_idx < len(switch_times) and t >= switch_times[sw_idx]:
                        direction *= -1
                        gabor_switches += 1
                        sw_idx += 1
                    gabor.ori += direction * speed * dt
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
                    movie.pause()          # freeze (reused next trial via seek)
                except Exception:
                    pass

            # STEP 4 — probe about the attended stream
            if not aborted:
                if attended == "Audio":
                    probe = _probe_audio(win, event, core, cfg, entry, words_file,
                                         btarget, min_word_len, rng, txt_color)
                elif visual_mode == "gabor":
                    probe = _probe_count(win, event, core, cfg, gabor_switches,
                                         txt_color)
                elif vis["video"].get("random_start", True):
                    # Random tetris cuts have no fixed ground truth, so a config
                    # game question can't be scored (and would mislead). Skip the
                    # probe: attend-visual trials are recorded but UNSCORED.
                    probe = {"modality": "Visual", "probe_id": None,
                             "question": None, "correct_answer": None,
                             "response": None, "rt_s": None, "correct": None}
                else:
                    probe = _probe_visual(win, event, core, cfg, rng, txt_color)
                if probe.get("aborted"):
                    aborted = True

        return {
            "subject": subject,
            "trial": trial_idx,
            "n_trials": n_trials,
            "timestamp": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "focus_mode": focus_mode,
            "attended_modality": attended,
            "choice_key": choice_key,
            "audio_stim_id": entry["id"],
            "audio_source": entry.get("source"),
            "speaker": entry.get("speaker"),
            "visual_mode": visual_mode,
            "visual_stim": ("gabor" if visual_mode == "gabor"
                            else os.path.relpath(video_file, cc.BASE_DIR)),
            "video_start_s": (round(vid_offset, 3) if visual_mode == "video"
                              else None),
            "gabor_switches": gabor_switches if visual_mode == "gabor" else None,
            "word_boundary_file": os.path.relpath(words_file, cc.BASE_DIR),
            "gap_onset_wall_clock": gap_onset_wallclock,
            "av_onset_global_s": av_onset_global,
            "av_onset_wall_clock": onset_wall,
            "audio_duration_s": round(audio_duration, 3),
            "block_target_s": round(btarget, 3),
            "block_duration_s": round(block_duration, 3),
            "refresh_hz": round(refresh_hz, 3),
            "gap_frames": gap_frames,
            "probe_modality": probe["modality"],
            "probe_id": probe["probe_id"],
            "probe_question": probe["question"],
            "probe_correct_answer": probe["correct_answer"],
            "probe_response": probe["response"],
            "probe_rt_s": probe["rt_s"],
            "probe_correct": probe["correct"],
            "probe_abs_error": probe.get("abs_error"),
            "webcam_enabled": webcam_info["enabled"],
            "webcam_file": webcam_info["file"],
            "webcam_opened": webcam_info["opened"],
            "webcam_av_onset_wallclock": av_onset_wallclock,
            "aborted": aborted,
        }

    # -----------------------------------------------------------------------
    # Trial loop
    # -----------------------------------------------------------------------
    records: list[dict] = []
    n_correct = n_scored = 0
    for i in range(n_trials):
        rec = run_one_trial(i + 1, audio_plan[i], focus_plan[i], video_offset())
        _append_behavior_csv(paths, rec)
        records.append(rec)
        if rec["probe_correct"] is not None:
            n_scored += 1
            n_correct += int(bool(rec["probe_correct"]))
        cc.log(f"Trial {i + 1}/{n_trials}: attend {rec['attended_modality']}, "
               f"audio {rec['audio_stim_id']}"
               + (f", cut@{rec['video_start_s']}s" if rec['video_start_s'] is not None else "")
               + f" -> probe {rec['probe_correct']}.")
        if rec["aborted"]:
            cc.log("Session aborted by user.")
            break
        if i < n_trials - 1:
            fixation.draw()
            win.flip()
            core.wait(iti)

    # --- Stop webcam + write the session summary ----------------------------
    if webcam is not None:
        st = webcam.stop() or {}
        webcam_info.update(opened=st.get("opened", webcam_info["opened"]),
                           frames=st.get("frames"),
                           duration_s=st.get("duration_s"))
        cc.log(f"Webcam stopped: {st.get('frames')} frames, "
               f"{st.get('duration_s')}s -> {webcam_info['file']}")

    _write_session_json(paths, subject, session_ts, seed, cfg, webcam_info, records)
    acc = f"{n_correct}/{n_scored}" if n_scored else "n/a"
    cc.log(f"Session complete: {len(records)} trial(s), probe accuracy {acc}.")

    # --- Goodbye ------------------------------------------------------------
    visual.TextStim(win, text="Session complete.\nThank you.",
                    color=txt_color, height=0.06).draw()
    win.flip()
    core.wait(1.5)
    win.close()
    core.quit()


# ===========================================================================
# Probe presentation
# ===========================================================================
def _key_label(key: str) -> str:
    return {"left": "←", "right": "→", "up": "↑", "down": "↓"}.get(key, key.upper())


def _run_yes_no(win, event, core, cfg, question: str, correct_present,
                txt_color) -> dict:
    """Show a yes/no question and collect a response (correct_present may be None)."""
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
    key, rt = event.waitKeys(keyList=[yes_key, no_key, quit_key],
                             timeStamped=clock)[0]
    if key == quit_key:
        return {"response": None, "rt_s": None, "correct": None, "aborted": True}
    response = "yes" if key == yes_key else "no"
    correct = (None if correct_present is None
               else (response == "yes") == bool(correct_present))
    return {"response": response, "rt_s": round(rt, 4), "correct": correct,
            "aborted": False}


def _probe_audio(win, event, core, cfg, entry, words_file, block_limit,
                 min_word_len, rng, txt_color) -> dict:
    """
    Yes/no 'was WORD spoken?' probe for attend-audio. The target is chosen at
    trial time from words actually heard within `block_limit` seconds, so it is
    valid even when only the first part of a longer clip was played.
    """
    words = []
    try:
        with open(words_file, encoding="utf-8") as f:
            words = json.load(f).get("words", [])
    except Exception:
        pass
    p = cc.make_trial_audio_probe(words, entry.get("transcript", ""),
                                  min_word_len, rng, window_end_s=block_limit)
    if not p:
        return {"modality": "Audio", "probe_id": None, "question": None,
                "correct_answer": None, "response": None, "rt_s": None,
                "correct": None}
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
        p = vprobes[0]
        correct_present = None
        cc.log("WARNING: no video probe has 'correct' set; logging UNSCORED. "
               "Fill visual.video.probes[].correct.")
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
    """Random reversal onset times (s from onset), intervals ~U[min_iv, max_iv]."""
    times: list[float] = []
    t = rng.uniform(min_iv, max_iv)
    while t < duration:
        times.append(round(t, 4))
        t += rng.uniform(min_iv, max_iv)
    return times


def _collect_number(win, event, core, cfg, prompt: str, txt_color):
    """On-screen numeric entry. Returns (value|None, rt|None, aborted)."""
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
            entered = (entered + key.replace("num_", ""))[:3]


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
def _start_webcam(cfg, subject, tag, win, visual, core, txt_color):
    """Start the webcam recorder subprocess (session-level). -> (ctrl|None, info)."""
    wcfg = cfg.get("webcam", {}) or {}
    info = {"enabled": bool(wcfg.get("enabled")), "file": None, "opened": False,
            "frames": None, "duration_s": None, "start_wallclock": None}
    if not wcfg.get("enabled"):
        return None, info

    from cma_webcam import WebcamController
    wdir = cc.abspath(wcfg.get("dir", "webcam"))
    os.makedirs(wdir, exist_ok=True)
    out_path = os.path.join(wdir, f"{subject}_{tag}.{wcfg.get('container', 'mp4')}")
    res = wcfg.get("resolution") or None
    ctrl = WebcamController(
        out_path, device=int(wcfg.get("device", 0)),
        fps=float(wcfg.get("fps", 0) or 0),
        resolution=tuple(res) if res else None,
        fourcc=wcfg.get("fourcc", "mp4v"),
        max_seconds=float(wcfg.get("max_seconds", 3600)))
    info["file"] = os.path.relpath(out_path, cc.BASE_DIR)

    visual.TextStim(win, text="Preparing camera…", color=txt_color,
                    height=0.05).draw()
    win.flip()

    opened = ctrl.start()
    info["opened"] = opened
    info["start_wallclock"] = ctrl.start_wallclock
    if opened:
        cc.log(f"Webcam recording (session) -> {out_path}")
        return ctrl, info

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
# Behaviour logging
# ===========================================================================
def _append_behavior_csv(paths: cc.Paths, record: dict) -> None:
    """Append one trial row to behavior/<subject>.csv (header on first write)."""
    os.makedirs(paths.behavior_dir, exist_ok=True)
    csv_path = os.path.join(paths.behavior_dir, f"{record['subject']}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            w.writeheader()
        w.writerow(record)


def _write_session_json(paths, subject, session_ts, seed, cfg, webcam_info,
                        records) -> None:
    """Write one JSON per session with all trial records + session metadata."""
    os.makedirs(paths.behavior_dir, exist_ok=True)
    out = os.path.join(paths.behavior_dir, f"{subject}_{session_ts}_session.json")
    scored = [r for r in records if r["probe_correct"] is not None]
    n_ok = sum(1 for r in scored if r["probe_correct"])
    payload = {
        "subject": subject, "session_timestamp": session_ts, "seed": seed,
        "n_trials_run": len(records), "n_scored": len(scored),
        "n_correct": n_ok,
        "accuracy": (round(n_ok / len(scored), 4) if scored else None),
        "visual_mode": cfg["visual"].get("mode"),
        "block_seconds": cfg["experiment"]["block_seconds"],
        "webcam": webcam_info,
        "trials": records,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    cc.log(f"Session log -> {os.path.join(paths.behavior_dir, subject + '.csv')}")
    cc.log(f"            -> {out}")


# ===========================================================================
# Orchestration
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Run a cross-modal attention session.")
    ap.add_argument("--subject", default="sub01", help="Participant id.")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--trials", type=int, default=None,
                    help="Override experiment.n_trials.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for stimulus/probe/cut randomisation.")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    if args.trials is not None:
        cfg["experiment"]["n_trials"] = args.trials
    paths = cc.Paths.from_config(cfg).ensure()
    manifest = cc.read_manifest(paths.manifest)

    seed = args.seed if args.seed is not None else _dt.datetime.now().microsecond
    rng = random.Random(seed)
    cc.log(f"Subject {args.subject}, seed {seed}, "
           f"{cfg['experiment'].get('n_trials')} trials.")

    run_session(cfg, paths, manifest, args.subject, rng, seed)


if __name__ == "__main__":
    main()
