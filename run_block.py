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


def _jittered_delay(base_s: float, jcfg: dict, rng: random.Random):
    """
    Delay (gap) length for a trial: base + Gaussian jitter (std_ms), clamped to
    +/- max_ms. Returns (delay_seconds, jitter_seconds). Never negative.
    """
    if not jcfg or not jcfg.get("enabled"):
        return base_s, 0.0
    std = float(jcfg.get("std_ms", 150)) / 1000.0
    mx = float(jcfg.get("max_ms", 300)) / 1000.0
    j = max(-mx, min(mx, rng.gauss(0.0, std)))
    return max(0.0, base_s + j), j


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


def _make_trigger(win, visual, cfg):
    """
    Build the photodiode trigger square, or None if disabled.

    Positioned in HEIGHT units using only the aspect RATIO, so it is robust to
    Retina/HiDPI displays (where win.size in pixels and the pixel coordinate
    system don't match, which pushes a pix-unit corner square off-screen).
    Height units are isotropic, so equal width/height is a true square.
    """
    tcfg = cfg.get("trigger", {}) or {}
    if not tcfg.get("enabled"):
        return None
    aspect = float(win.size[0]) / float(win.size[1])       # ratio -> Retina-safe
    side = (float(tcfg["size_h"]) if tcfg.get("size_h") is not None
            else float(tcfg.get("size_px", 100)) / float(win.size[1]))
    half = side / 2.0
    xr, xl = aspect / 2.0 - half, -aspect / 2.0 + half
    yt, yb = 0.5 - half, -0.5 + half
    pos = {"bottom-right": (xr, yb), "bottom-left": (xl, yb),
           "top-right": (xr, yt), "top-left": (xl, yt)}.get(
               tcfg.get("corner", "bottom-right"), (xr, yb))
    return visual.Rect(win, width=side, height=side, units="height", pos=pos,
                       fillColor=tcfg.get("color", "white"),
                       lineColor=tcfg.get("color", "white"))


def _test_trigger(cfg: dict) -> None:
    """Open the window and blink ONLY the trigger square, to check placement."""
    from psychopy import visual, core, event, monitors
    exp = cfg["experiment"]
    win = visual.Window(fullscr=bool(exp["fullscreen"]), color=exp["bg_color"],
                        units="height", monitor=monitors.Monitor("expMonitor"),
                        allowGUI=False, winType="pyglet")
    win.mouseVisible = False
    trig = _make_trigger(win, visual, cfg)
    info = visual.TextStim(
        win, color=exp["text_color"], height=0.04, wrapWidth=1.4,
        text=("Trigger test — a white square should blink in the "
              f"{cfg.get('trigger', {}).get('corner', 'bottom-right')} corner.\n"
              f"win.size = {tuple(win.size)}\n\nPress any key to exit."))
    clock = core.Clock()
    while not event.getKeys():
        info.draw()
        if trig is not None and int(clock.getTime() * 2) % 2 == 0:   # ~1 Hz blink
            trig.draw()
        win.flip()
    win.close()
    core.quit()


# ===========================================================================
# Session
# ===========================================================================
def run_session(cfg: dict, paths: cc.Paths, manifest: dict, subject: str,
                rng: random.Random, seed: int, session: str = "") -> None:
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
    start_on_space = bool(exp.get("start_on_space", True))
    gap_base = float(exp["gap_seconds"])
    jitter_cfg = exp.get("jitter", {}) or {}
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
    jit = (f"+/-jitter(std {jitter_cfg.get('std_ms')}ms, max {jitter_cfg.get('max_ms')}ms)"
           if jitter_cfg.get("enabled") else "no jitter")
    cc.log(f"Refresh ~{refresh_hz:.2f} Hz. Session: {n_trials} trials of "
           f"{block_cap:.0f}s ({visual_mode}); base delay {gap_base:.2f}s {jit}.")

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

    # Focus cue: a small "attend to" line above a big "AUDIO"/"VISUAL" word.
    cue_word_h = float(exp.get("cue_word_height", 0.20))
    cue_prefix_h = float(exp.get("cue_prefix_height", 0.05))
    cue_word = visual.TextStim(win, text="", color=txt_color, height=cue_word_h,
                               bold=True, pos=(0, 0), alignText="center")
    cue_prefix = visual.TextStim(
        win, text=exp.get("cue_prefix_text", "attend to"), color=txt_color,
        height=cue_prefix_h, pos=(0, (cue_word_h + cue_prefix_h) * 0.6),
        alignText="center")
    cue_space = visual.TextStim(win, text="Press SPACE to start", color=txt_color,
                                height=cue_prefix_h * 0.85,
                                pos=(0, -(cue_word_h * 0.5 + cue_prefix_h * 1.5)))

    # --- Photodiode trigger square (shown for the whole gap) ----------------
    trigger = _make_trigger(win, visual, cfg)
    if trigger is not None:
        cc.log(f"Trigger square at {cfg['trigger'].get('corner', 'bottom-right')}"
               f"; on during the gap (rising edge = gap onset, falling = AV onset).")

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

    # --- Tetris self-playing visual stimulus (reused ImageStim per frame) ----
    TetrisGame = None
    tetris_stim = tetris_fix = None
    tcfg = vis.get("tetris", {}) or {}
    if visual_mode == "tetris":
        from tetris import TetrisGame       # local import (pulls only numpy)
        cols_t, rows_t = int(tcfg.get("cols", 10)), int(tcfg.get("rows", 20))
        board_h = float(tcfg.get("board_height", 0.5))
        board_w = board_h * (cols_t / rows_t)
        cell_px = int(tcfg.get("render_cell_px", 14))
        init_img = TetrisGame(cols=cols_t, rows=rows_t, seed=0).to_image(cell_px)
        # PsychoPy maps numpy row 0 to the bottom, so flipVert keeps the stack down.
        tetris_stim = visual.ImageStim(
            win, image=init_img, size=(board_w, board_h), units="height",
            interpolate=False, flipVert=bool(tcfg.get("flip_vertical", True)))
        if tcfg.get("show_fixation", True):
            tetris_fix = visual.Circle(win, radius=0.006, units="height",
                                       fillColor=txt_color, lineColor=txt_color,
                                       pos=(0, 0))
        cc.log(f"Tetris {cols_t}x{rows_t}, board {board_w:.2f}x{board_h:.2f}h, "
               f"speed {tcfg.get('speed_s_per_step', 0.045)}s/step.")

    global_clock = core.Clock()

    # --- Session webcam (ONE recording for the whole session) ---------------
    session_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    webcam, webcam_info = _start_webcam(cfg, subject, f"{session_ts}_session",
                                        win, visual, core, txt_color)

    # --- Session intro / fixation instruction (once) ------------------------
    visual.TextStim(
        win, color=txt_color, height=0.045, wrapWidth=1.5, alignText="center",
        text=("Keep your eyes on the central fixation dot throughout.\n\n"
              "Before each trial you will be told which stream to attend:\n"
              f"AUDIO — {_attend_hint('Audio', visual_mode).lower()}\n"
              f"VISUAL — {_attend_hint('Visual', visual_mode).lower()}\n\n"
              "Press SPACE to begin, or ESC to quit.")).draw()
    win.flip()
    event.clearEvents()
    if "escape" in (event.waitKeys(keyList=["space", "escape"]) or []):
        if webcam is not None:
            webcam.stop()
        win.close()
        core.quit()
        return

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
        game = None
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
        elif visual_mode == "tetris":
            # Deterministic per-trial game the subject PLAYS with the arrow keys
            # (seed = base + master seed + trial). on_top_out="reset" keeps the
            # stimulus running if they top out mid-trial.
            game_seed = int(tcfg.get("seed_base", 0)) + seed + trial_idx
            game = TetrisGame(
                cols=int(tcfg.get("cols", 10)), rows=int(tcfg.get("rows", 20)),
                seed=game_seed, mode="player", on_top_out="reset",
                start_level=int(tcfg.get("start_level", 0)),
                gravity_scale=float(tcfg.get("gravity_scale", 1.0)),
                lock_delay_ticks=int(tcfg.get("lock_delay_ticks", 30)),
                flash_s=float(tcfg.get("flash_s", 0.18)))
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
        tetris_clears = 0
        tetris_max_height = 0
        game_record_file = None
        instruction_seconds = None
        delay_seconds = None
        delay_frames = None
        jitter_s = 0.0
        gap_onset_wallclock = None
        av_onset_global = onset_wall = av_onset_wallclock = None
        block_duration = 0.0
        probe = {"modality": None, "probe_id": None, "question": None,
                 "correct_answer": None, "response": None, "rt_s": None,
                 "correct": None}

        # STEP 1 — focus: auto = cue the assigned stream; manual = subject chooses
        # (its actual on-screen length is measured for the log).
        counter.text = f"Trial {trial_idx} / {n_trials}"
        if focus_mode == "auto":
            attended = assigned_focus
            choice_key = "auto"
            cue_word.text = attended.upper()      # big "AUDIO" / "VISUAL"
            cue_prefix.draw()                     # small "attend to" above it
            cue_word.draw()
            counter.draw()
            if start_on_space:
                cue_space.draw()
            win.flip()
            instr_clock = core.Clock()
            event.clearEvents()
            if start_on_space:
                got = event.waitKeys(keyList=["space", "escape"])  # self-paced
            else:
                got = event.waitKeys(maxWait=cue_seconds, keyList=["escape"])
            instruction_seconds = instr_clock.getTime()
            if got and "escape" in got:
                aborted = True
        else:
            instruction.draw()
            counter.draw()
            win.flip()
            instr_clock = core.Clock()
            event.clearEvents()
            keys = event.waitKeys(keyList=["left", "right", "escape"])
            instruction_seconds = instr_clock.getTime()
            if "escape" in keys:
                aborted = True
            else:
                choice_key = keys[0]
                attended = "Audio" if choice_key == "left" else "Visual"

        if not aborted:
            # STEP 2 — jittered fixation delay, frame-counted (trigger square on)
            delay_s, jitter_s = _jittered_delay(gap_base, jitter_cfg, rng)
            delay_frames = max(1, int(round(delay_s * refresh_hz)))
            delay_seconds = round(delay_frames / refresh_hz, 4)
            for gi in range(delay_frames):
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
            elif visual_mode == "tetris":
                # The subject PLAYS the game: poll the control keys each frame, feed
                # them to the engine, advance, and draw. Record the exact display
                # frame times (relative to onset) so the played game can be
                # reconstructed frame-for-frame afterwards.
                cell_px = int(tcfg.get("render_cell_px", 14))
                keymap = {"left": "left", "right": "right", "up": "cw", "z": "ccw",
                          "down": "soft", "space": "hard", "c": "hold"}
                prev_t = 0.0
                frame_times: list[float] = []
                while block_clock.getTime() < btarget:
                    t = block_clock.getTime()
                    for k in event.getKeys(keyList=list(keymap) + ["escape"]):
                        if k == "escape":
                            aborted = True
                            break
                        game.press(keymap[k])          # buffered; applied on next tick
                    if aborted:
                        break
                    game.update(t - prev_t)
                    prev_t = t
                    tetris_stim.image = game.to_image(cell_px)
                    tetris_stim.draw()
                    if tetris_fix is not None:
                        tetris_fix.draw()
                    win.flip()
                    frame_times.append(round(block_clock.getTime(), 5))
                tetris_clears = game.line_clear_events
                tetris_max_height = game.max_stack_height
                # The engine's record() holds the seed + every keystroke by tick +
                # the full stats -> the played game replays bit-for-bit. Add the
                # on-screen render params and display frame times for reconstruction.
                game_record_file = _save_game_record(
                    paths, subject, session_ts, trial_idx, {
                        **game.record(),
                        "render_cell_px": cell_px,
                        "refresh_hz": round(refresh_hz, 3),
                        "block_duration_s": round(block_clock.getTime(), 4),
                        "n_display_frames": len(frame_times),
                        "frame_times_s": frame_times,
                    })
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
                elif visual_mode == "tetris":
                    # Scored against the peak stack height (highest number of rows).
                    probe = _probe_count(win, event, core, cfg, tetris_max_height,
                                         txt_color, tcfg.get("probe_question"))
                elif visual_mode == "gabor":
                    probe = _probe_count(win, event, core, cfg, gabor_switches,
                                         txt_color,
                                         vis["gabor"].get("probe_question"))
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
            "session": session,
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
            "visual_stim": ({"gabor": "gabor", "tetris": "tetris"}.get(
                visual_mode, os.path.relpath(video_file, cc.BASE_DIR))),
            "video_start_s": (round(vid_offset, 3) if visual_mode == "video"
                              else None),
            "gabor_switches": gabor_switches if visual_mode == "gabor" else None,
            "tetris_clears": tetris_clears if visual_mode == "tetris" else None,
            "tetris_max_height": (tetris_max_height if visual_mode == "tetris"
                                  else None),
            "game_record_file": (os.path.relpath(game_record_file, cc.BASE_DIR)
                                 if game_record_file else None),
            "word_boundary_file": os.path.relpath(words_file, cc.BASE_DIR),
            "instruction_seconds": (round(instruction_seconds, 4)
                                    if instruction_seconds is not None else None),
            "delay_base_s": gap_base,
            "delay_jitter_ms": round(jitter_s * 1000.0, 1),
            "delay_seconds": delay_seconds,
            "delay_frames": delay_frames,
            "gap_onset_wall_clock": gap_onset_wallclock,
            "av_onset_global_s": av_onset_global,
            "av_onset_wall_clock": onset_wall,
            "audio_duration_s": round(audio_duration, 3),
            "block_target_s": round(btarget, 3),
            "block_duration_s": round(block_duration, 3),
            "refresh_hz": round(refresh_hz, 3),
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

    _write_session_json(paths, subject, session_ts, seed, cfg, webcam_info,
                        records, session)
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
    return {"left": "LEFT", "right": "RIGHT",
            "up": "UP", "down": "DOWN"}.get(key, key.upper())


def _attend_hint(attended: str, visual_mode: str) -> str:
    """Short task reminder shown on the auto cue."""
    if attended == "Audio":
        return "Listen for the words."
    if visual_mode == "tetris":
        return "Track the tallest the stack gets."
    if visual_mode == "gabor":
        return "Count the direction reversals."
    return ""


def _run_yes_no(win, event, core, cfg, question: str, correct_present,
                txt_color) -> dict:
    """Show a yes/no question and collect a response (correct_present may be None)."""
    from psychopy import visual
    keys_cfg = cfg["probe"]["keys"]
    yes_key, no_key = keys_cfg["yes"], keys_cfg["no"]
    quit_key = cfg["probe"].get("quit_key", "escape")
    header = cfg["probe"].get("instruction", "")

    # Lay the options out so their on-screen SIDE matches their key: the option
    # bound to the left key sits on the left. With no=left / yes=right this reads
    # "NO (<-)      YES (->)".
    _side = {"left": 0, "right": 2}
    opts = sorted([("YES", yes_key), ("NO", no_key)],
                  key=lambda o: _side.get(o[1], 1))
    labels = "        ".join(f"{name} ({_key_label(k)})" for name, k in opts)
    prompt = f"{header}\n\n{question}\n\n{labels}"
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


def _probe_count(win, event, core, cfg, true_count: int, txt_color,
                 question=None) -> dict:
    """Numeric count probe (Gabor reversals or Tetris row clears)."""
    if not question:
        question = "How many events did you count?"
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
    """
    Append one trial row to behavior/<subject>.csv (header on first write).

    Guard against silent corruption: if an existing file has a DIFFERENT column
    schema (e.g. a stale CSV from another code version), never append into it —
    write to a fresh timestamped file instead and warn.
    """
    os.makedirs(paths.behavior_dir, exist_ok=True)
    fields = list(record.keys())
    csv_path = os.path.join(paths.behavior_dir, f"{record['subject']}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != fields:
            alt = os.path.join(paths.behavior_dir,
                               f"{record['subject']}_{record.get('timestamp', 'x')}.csv")
            cc.log(f"WARNING: {os.path.basename(csv_path)} has a different column "
                   f"schema; writing this row to {os.path.basename(alt)} to avoid "
                   f"corrupting it.")
            csv_path = alt
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(record)


def _save_game_record(paths: cc.Paths, subject: str, session_ts: str,
                      trial_idx: int, meta: dict) -> str:
    """
    Save one Tetris trial's reconstruction record (seed + params + duration +
    display frame times). The game is deterministic, so reconstruct_tetris.py can
    render a faithful mp4 from this afterwards. Returns the file path.
    """
    os.makedirs(paths.games_dir, exist_ok=True)
    out = os.path.join(paths.games_dir,
                       f"{subject}_{session_ts}_t{trial_idx:03d}.json")
    payload = {"subject": subject, "session_timestamp": session_ts,
               "trial": trial_idx, **meta}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def _write_session_json(paths, subject, session_ts, seed, cfg, webcam_info,
                        records, session="") -> None:
    """Write one JSON per session with all trial records + session metadata."""
    os.makedirs(paths.behavior_dir, exist_ok=True)
    out = os.path.join(paths.behavior_dir, f"{subject}_{session_ts}_session.json")
    scored = [r for r in records if r["probe_correct"] is not None]
    n_ok = sum(1 for r in scored if r["probe_correct"])
    payload = {
        "subject": subject, "session": session,
        "session_timestamp": session_ts, "seed": seed,
        "n_trials_run": len(records), "n_scored": len(scored),
        "n_correct": n_ok,
        "accuracy": (round(n_ok / len(scored), 4) if scored else None),
        "visual_mode": cfg["visual"].get("mode"),
        "block_seconds": cfg["experiment"]["block_seconds"],
        "config": cfg,                       # full config snapshot for provenance
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
    ap.add_argument("--test-trigger", action="store_true",
                    help="Blink only the photodiode square to check placement, "
                         "then exit (no trials).")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    if args.test_trigger:
        _test_trigger(cfg)
        return
    if args.trials is not None:
        cfg["experiment"]["n_trials"] = args.trials
    paths = cc.Paths.from_config(cfg).ensure()
    manifest = cc.read_manifest(paths.manifest)

    # Fixed master seed from config (reproducible session), unless overridden.
    seed = (args.seed if args.seed is not None
            else int(cfg["experiment"].get("seed", 0)))
    rng = random.Random(seed)
    cc.log(f"Subject {args.subject}, seed {seed}, "
           f"{cfg['experiment'].get('n_trials')} trials.")

    run_session(cfg, paths, manifest, args.subject, rng, seed)


if __name__ == "__main__":
    main()
