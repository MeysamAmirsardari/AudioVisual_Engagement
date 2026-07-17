#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_block.py — LOAD-MODULATION variant of the audio-visual experiment.

Difference from the parent experiment: there is NO attention instruction. On every
trial the subject simply PLAYS a Tetris game whose DIFFICULTY varies trial-to-trial
(very easy … super hard, balanced and randomly ordered). A speech clip plays in the
background throughout, and after the block a comprehension probe about the AUDIO is
asked on EVERY trial. Difficulty is thus an implicit cognitive-load manipulation on
the visual/motor task, and audio-probe accuracy (+ the beep reaction time) index how
much spare capacity was left for the audio.

Per trial
---------
  1. Ready       — fixation + trial counter (no attention cue).
  2. Gap         — fixation for a jittered, frame-counted delay; the photodiode
                   square is ON (rising edge = gap onset, falling edge = AV onset).
  3. Block       — a 25 s Tetris game the subject plays; if they top out the game
                   RESTARTS and play continues. A background speech clip plays via a
                   low-jitter LSL audio engine (see lsl_audio.py).
  4. Beep        — a short tone right after the block; the subject responds as fast
                   as possible (an extra vigilance/arousal cue). A photodiode pulse
                   and a DAC-timestamped LSL marker tag its onset.
  5. Audio probe — yes/no "was WORD spoken?" about the clip (kept every trial).

Synchronisation (belt and braces): a photodiode square (hardware visual timing), an
LSL marker stream (discrete events), and an LSL audio stream (the waveform, DAC-
timestamped) — plus a local backup JSONL of every marker. Everything is logged: one
CSV row + a per-session JSON (full config snapshot) + a Tetris reconstruction record
per trial + the LSL event log.

Run:  python run.py --subject P01 --session 01     (preferred; organises outputs)
      python run_block.py --subject sub01 --test-trigger   (photodiode placement)
      python run_block.py --test-audio                     (LSL audio self-test)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import random
import sys
import time

import cma_common as cc


# ===========================================================================
# Trial planning helpers
# ===========================================================================
def _assign_audio(items: list[dict], n_trials: int, rng: random.Random) -> list[dict]:
    """Balanced audio-clip assignment: reshuffle+concatenate the library so each clip
    is used about equally and none repeats until all have appeared."""
    plan: list[dict] = []
    while len(plan) < n_trials:
        block = items[:]
        rng.shuffle(block)
        plan.extend(block)
    return plan[:n_trials]


def _assign_difficulty(levels: list[dict], n_trials: int,
                       rng: random.Random) -> list[dict]:
    """Balanced difficulty assignment across the named levels, randomly ordered
    (same reshuffle+concatenate scheme, so tiers are as balanced as n_trials allows)."""
    plan: list[dict] = []
    while len(plan) < n_trials:
        block = levels[:]
        rng.shuffle(block)
        plan.extend(block)
    return plan[:n_trials]


def _resolve(paths: cc.Paths, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(paths.stim_dir, rel)


def _jittered_delay(base_s: float, jcfg: dict, rng: random.Random):
    """Gap length = base + Gaussian jitter (std_ms), clamped to +/- max_ms. Never < 0.
    Returns (delay_seconds, jitter_seconds)."""
    if not jcfg or not jcfg.get("enabled"):
        return base_s, 0.0
    std = float(jcfg.get("std_ms", 150)) / 1000.0
    mx = float(jcfg.get("max_ms", 300)) / 1000.0
    j = max(-mx, min(mx, rng.gauss(0.0, std)))
    return max(0.0, base_s + j), j


def _make_trigger(win, visual, cfg):
    """Photodiode trigger square (or None). Positioned in HEIGHT units using only the
    aspect RATIO, so it is robust to Retina/HiDPI displays."""
    tcfg = cfg.get("trigger", {}) or {}
    if not tcfg.get("enabled"):
        return None
    aspect = float(win.size[0]) / float(win.size[1])
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
        if trig is not None and int(clock.getTime() * 2) % 2 == 0:
            trig.draw()
        win.flip()
    win.close()
    core.quit()


def _test_audio(cfg: dict) -> None:
    """Open the LSL audio engine and play a couple of beeps, printing timing — a
    quick check that the device + LSL streams work before a real session."""
    eng = _open_audio(cfg, backup_path=None)
    if eng is None:
        print("Audio engine could not be opened.")
        return
    print("stream_info:", json.dumps(eng.stream_info(), indent=2))
    for f in (700, 1000):
        eng.play(eng.make_beep(freq=f, dur=0.12))
        onset = eng.wait_onset(timeout=1.0)
        print(f"beep {f} Hz onset (LSL) = {onset}")
        time.sleep(0.5)
    eng.close()
    print("audio self-test done.")


# ===========================================================================
# LSL audio engine
# ===========================================================================
def _open_audio(cfg: dict, backup_path):
    """Create the low-jitter LSL audio engine, or None if disabled/unavailable."""
    lcfg = cfg.get("lsl", {}) or {}
    if not lcfg.get("enabled", True):
        cc.log("LSL audio disabled in config.")
        return None
    try:
        from lsl_audio import LSLAudioEngine
        dev = lcfg.get("device")
        eng = LSLAudioEngine(
            device=(None if dev in (None, "", "default") else dev),
            blocksize=int(lcfg.get("blocksize", 256)),
            latency=lcfg.get("latency", "low"),
            audio_name=lcfg.get("audio_stream_name", "ExpAudio"),
            marker_name=lcfg.get("marker_stream_name", "ExpAudioMarkers"),
            source_id=lcfg.get("source_id", "load_experiment_audio"),
            backup_path=backup_path, amp=float(lcfg.get("amp", 1.0)))
        cc.log(f"LSL audio engine: {eng.samplerate} Hz, block {eng.blocksize}, "
               f"offset(LSL-PA) {eng._offset:+.3f}s, streams "
               f"'{eng.audio_stream_name}'/'{eng.marker_stream_name}'.")
        return eng
    except Exception as exc:
        cc.log(f"ERROR opening LSL audio engine ({type(exc).__name__}: {exc}).")
        if (cfg.get("lsl", {}) or {}).get("required", True):
            raise
        return None


# ===========================================================================
# Session
# ===========================================================================
def run_session(cfg: dict, paths: cc.Paths, manifest: dict, subject: str,
                rng: random.Random, seed: int, session: str = "") -> None:
    exp = cfg["experiment"]
    vis = cfg["visual"]
    tcfg = vis.get("tetris", {}) or {}
    n_trials = int(exp.get("n_trials", 1))
    block_cap = float(exp["block_seconds"])
    iti = float(exp.get("inter_trial_interval_s", 0.8))
    ready_seconds = float(exp.get("ready_seconds", 1.0))
    start_on_space = bool(exp.get("start_on_space", False))
    gap_base = float(exp["gap_seconds"])
    jitter_cfg = exp.get("jitter", {}) or {}
    min_word_len = int(cfg["probe"]["audio"]["min_word_len"])
    bcfg = cfg.get("beep", {}) or {}

    audio_items = manifest.get("audio", [])
    if not audio_items:
        raise RuntimeError("No audio stimuli in the library. Run:  python build_stimuli.py")

    # Difficulty levels (very easy … super hard), balanced + shuffled per session.
    levels = list(cfg["difficulty"]["levels"])
    audio_plan = _assign_audio(audio_items, n_trials, rng)
    diff_plan = _assign_difficulty(levels, n_trials, rng)

    # --- Backup LSL event log + audio engine (opened ONCE) ------------------
    os.makedirs(paths.behavior_dir, exist_ok=True)
    session_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    lsl_backup = os.path.join(paths.behavior_dir, f"{subject}_{session_ts}_lsl_events.jsonl")
    eng = _open_audio(cfg, backup_path=lsl_backup)
    if eng is not None:
        eng.marker("session_start", extra={"subject": subject, "session": session,
                                           "seed": seed, "n_trials": n_trials})

    # --- Window -------------------------------------------------------------
    from psychopy import visual, core, event, logging, monitors
    logging.console.setLevel(logging.WARNING)
    mon = monitors.Monitor("expMonitor")
    win = visual.Window(fullscr=bool(exp["fullscreen"]), color=exp["bg_color"],
                        units="height", monitor=mon, allowGUI=False, winType="pyglet")
    win.mouseVisible = False
    txt_color = exp["text_color"]

    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                       nWarmUpFrames=10, threshold=1)
    refresh_hz = measured if measured else float(exp["fallback_refresh_hz"])
    cc.log(f"Refresh ~{refresh_hz:.2f} Hz. Session: {n_trials} trials of "
           f"{block_cap:.0f}s Tetris (load-modulation, {len(levels)} difficulty tiers); "
           f"base gap {gap_base:.2f}s.")

    fixation = visual.TextStim(win, text="+", color=txt_color, height=0.12)
    counter = visual.TextStim(win, text="", color=txt_color, height=0.03, pos=(0, -0.42))
    ready_msg = visual.TextStim(win, text="", color=txt_color, height=0.045,
                                wrapWidth=1.5, alignText="center")
    trigger = _make_trigger(win, visual, cfg)
    if trigger is not None:
        cc.log(f"Photodiode square at {cfg['trigger'].get('corner', 'bottom-right')} "
               "(on during gap; pulsed at the beep).")

    # --- Tetris stimulus (reused ImageStim per frame) -----------------------
    from tetris import TetrisGame
    cols_t, rows_t = int(tcfg.get("cols", 10)), int(tcfg.get("rows", 20))
    board_h = float(tcfg.get("board_height", 0.5))
    board_w = board_h * (cols_t / rows_t)
    cell_px = int(tcfg.get("render_cell_px", 14))
    init_img = TetrisGame(cols=cols_t, rows=rows_t, seed=0).to_image(cell_px)
    tetris_stim = visual.ImageStim(win, image=init_img, size=(board_w, board_h),
                                   units="height", interpolate=False,
                                   flipVert=bool(tcfg.get("flip_vertical", True)))
    tetris_fix = (visual.Circle(win, radius=0.006, units="height", fillColor=txt_color,
                                lineColor=txt_color, pos=(0, 0))
                  if tcfg.get("show_fixation", True) else None)
    # Subject-facing HUD: SCORE (top-left) + countdown TIMER (top-right), flanking the
    # centred board. Re-rendered only when the shown value changes (cheap).
    hud_score = visual.TextStim(win, text="", color=txt_color, height=0.05,
                                pos=(-0.46, 0.42), alignText="left", anchorHoriz="left")
    hud_time = visual.TextStim(win, text="", color=txt_color, height=0.05,
                               pos=(0.46, 0.42), alignText="right", anchorHoriz="right")
    # NB: SPACE is deliberately NOT a game control (no hard drop) — it is reserved
    # for the post-trial beep response (and starting the session).
    keymap = {"left": "left", "right": "right", "up": "cw", "z": "ccw",
              "down": "soft", "c": "hold"}

    global_clock = core.Clock()
    webcam, webcam_info = _start_webcam(cfg, subject, f"{session_ts}_session",
                                        win, visual, core, txt_color)

    # --- Intro --------------------------------------------------------------
    visual.TextStim(
        win, color=txt_color, height=0.045, wrapWidth=1.5, alignText="center",
        text=("You will PLAY Tetris for a series of short games of varying difficulty.\n"
              "Play as well as you can — if you top out, the game restarts automatically.\n\n"
              "A voice plays in the background; after each game you'll answer ONE quick\n"
              "yes/no question about it, and respond to a short beep.\n\n"
              "Controls: left/right move · up/Z rotate · down soft drop · C hold.\n\n"
              "Press SPACE to begin, or ESC to quit.")).draw()
    win.flip()
    event.clearEvents()
    if "escape" in (event.waitKeys(keyList=["space", "escape"]) or []):
        _shutdown(eng, webcam, win, core)
        return

    beep_buf = (eng.make_beep(freq=float(bcfg.get("freq_hz", 1000)),
                              dur=float(bcfg.get("duration_s", 0.1)),
                              amp=float(bcfg.get("amp", 0.5)))
                if (eng is not None and bcfg.get("enabled", True)) else None)

    # -----------------------------------------------------------------------
    def run_one_trial(trial_idx: int, entry: dict, difficulty: dict) -> dict:
        audio_wav = _resolve(paths, entry["audio_path"])
        words_file = _resolve(paths, entry["words_json"])
        audio_duration = float(entry.get("duration_s")
                               or cc.audio_duration_seconds(audio_wav))
        btarget = audio_duration          # trial length == this clip's audio length
        audio_buf = eng.load(audio_wav) if eng is not None else None
        game = TetrisGame(
            cols=cols_t, rows=rows_t, seed=int(tcfg.get("seed_base", 0)) + seed + trial_idx,
            mode="player", on_top_out="reset",
            start_level=int(difficulty.get("start_level", 0)),
            gravity_scale=float(difficulty.get("gravity_scale", 1.0)),
            lock_delay_ticks=int(difficulty.get("lock_delay_ticks",
                                                tcfg.get("lock_delay_ticks", 30))),
            flash_s=float(tcfg.get("flash_s", 0.18)))

        aborted = False
        rec_extra = {"trial": trial_idx, "difficulty": difficulty.get("name"),
                     "audio_stim_id": entry["id"]}
        L = {k: None for k in ("gap_onset_lsl", "av_onset_lsl", "audio_onset_lsl",
                               "block_end_lsl", "beep_onset_lsl", "beep_response_lsl")}
        gap_onset_wall = av_onset_wall = onset_wall_iso = None
        delay_seconds = delay_frames = None
        jitter_s = 0.0
        block_duration = 0.0
        tetris_clears = tetris_max_height = n_resets = 0
        game_record_file = None
        beep_key = beep_rt = None
        probe = {"modality": None, "probe_id": None, "question": None,
                 "correct_answer": None, "response": None, "rt_s": None, "correct": None}

        # STEP 1 — ready (no attention cue) -----------------------------------
        counter.text = f"Trial {trial_idx} / {n_trials}"
        ready_msg.text = "Get ready…"
        ready_msg.draw(); counter.draw(); win.flip()
        if eng is not None:
            eng.marker("trial_start", extra=rec_extra)
        event.clearEvents()
        if start_on_space:
            if "escape" in (event.waitKeys(keyList=["space", "escape"]) or []):
                aborted = True
        else:
            if event.waitKeys(maxWait=ready_seconds, keyList=["escape"]):
                aborted = True

        if not aborted:
            # STEP 2 — jittered gap, photodiode square ON --------------------
            delay_s, jitter_s = _jittered_delay(gap_base, jitter_cfg, rng)
            delay_frames = max(1, int(round(delay_s * refresh_hz)))
            delay_seconds = round(delay_frames / refresh_hz, 4)
            for gi in range(delay_frames):
                fixation.draw()
                if trigger is not None:
                    trigger.draw()
                win.flip()
                if gi == 0:
                    gap_onset_wall = time.time()
                    if eng is not None:
                        L["gap_onset_lsl"] = eng.marker("gap_onset", extra=rec_extra)

            # STEP 3 — AV onset: start audio + game, drop the square ----------
            if eng is not None and audio_buf is not None:
                eng.play(audio_buf)
            tetris_stim.image = game.to_image(cell_px)
            tetris_stim.draw()
            if tetris_fix is not None:
                tetris_fix.draw()
            win.flip()                                          # <- AV ONSET (square off)
            block_clock = core.Clock(); block_clock.reset()
            av_onset_wall = time.time()
            onset_wall_iso = _dt.datetime.now().isoformat(timespec="milliseconds")
            if eng is not None:
                L["av_onset_lsl"] = eng.marker("av_onset", extra=rec_extra)
                L["audio_onset_lsl"] = eng.wait_onset(timeout=0.5)
                eng.marker("audio_onset", timestamp=L["audio_onset_lsl"],
                           extra={**rec_extra, "audio_path": entry["audio_path"]})

            # STEP 3b — the subject plays for `btarget` seconds --------------
            prev_t = 0.0
            frame_times: list[float] = []
            last_resets = 0
            shown_score = shown_rem = None
            while block_clock.getTime() < btarget:
                t = block_clock.getTime()
                for k in event.getKeys(keyList=list(keymap) + ["escape"]):
                    if k == "escape":
                        aborted = True
                        break
                    game.press(keymap[k])
                if aborted:
                    break
                game.update(t - prev_t)
                prev_t = t
                if game.resets != last_resets:                  # topped out -> restarted
                    last_resets = game.resets
                    if eng is not None:
                        eng.marker("game_reset", extra={**rec_extra, "resets": last_resets,
                                                        "t_block_s": round(t, 4)})
                if game.score != shown_score:                   # HUD: score
                    shown_score = game.score
                    hud_score.text = f"Score {shown_score}"
                rem = int(math.ceil(max(0.0, btarget - t)))     # HUD: countdown timer
                if rem != shown_rem:
                    shown_rem = rem
                    hud_time.text = f"Time {rem}s"
                tetris_stim.image = game.to_image(cell_px)
                tetris_stim.draw()
                if tetris_fix is not None:
                    tetris_fix.draw()
                hud_score.draw()
                hud_time.draw()
                win.flip()
                frame_times.append(round(block_clock.getTime(), 5))
            block_duration = block_clock.getTime()
            tetris_clears = game.line_clear_events
            tetris_max_height = game.max_stack_height
            n_resets = game.resets
            if eng is not None:
                eng.stop(fade=True)                              # fade out the clip
                L["block_end_lsl"] = eng.marker("block_end", extra={
                    **rec_extra, "block_duration_s": round(block_duration, 4),
                    "resets": n_resets, "clears": tetris_clears})

            game_record_file = _save_game_record(paths, subject, session_ts, trial_idx, {
                **game.record(), "difficulty": difficulty,
                "render_cell_px": cell_px, "refresh_hz": round(refresh_hz, 3),
                "block_duration_s": round(block_duration, 4),
                "n_display_frames": len(frame_times), "frame_times_s": frame_times})

            # STEP 4 — beep + speeded response -------------------------------
            if not aborted and eng is not None and beep_buf is not None:
                beep_key, beep_rt = _beep_response(win, event, core, eng, beep_buf,
                                                   fixation, trigger, bcfg, refresh_hz,
                                                   rec_extra, L)
                if beep_key == "escape":
                    aborted = True

            # STEP 5 — audio comprehension probe (every trial) ---------------
            if not aborted:
                if eng is not None:
                    eng.marker("probe_onset", extra=rec_extra)
                probe = _probe_audio(win, event, core, cfg, entry, words_file,
                                     btarget, min_word_len, rng, txt_color)
                if eng is not None:
                    eng.marker("probe_response", extra={
                        **rec_extra, "response": probe.get("response"),
                        "correct": probe.get("correct"), "rt_s": probe.get("rt_s")})
                if probe.get("aborted"):
                    aborted = True

        return {
            "subject": subject, "session": session, "trial": trial_idx,
            "n_trials": n_trials,
            "timestamp": _dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "difficulty_name": difficulty.get("name"),
            "start_level": difficulty.get("start_level"),
            "gravity_scale": difficulty.get("gravity_scale"),
            "lock_delay_ticks": difficulty.get("lock_delay_ticks",
                                               tcfg.get("lock_delay_ticks", 30)),
            "audio_stim_id": entry["id"], "audio_source": entry.get("source"),
            "speaker": entry.get("speaker"),
            "visual_mode": "tetris_play",
            "tetris_clears": tetris_clears, "tetris_max_height": tetris_max_height,
            "tetris_resets": n_resets,
            "game_record_file": (os.path.relpath(game_record_file, cc.BASE_DIR)
                                 if game_record_file else None),
            "word_boundary_file": os.path.relpath(words_file, cc.BASE_DIR),
            "delay_base_s": gap_base, "delay_jitter_ms": round(jitter_s * 1000.0, 1),
            "delay_seconds": delay_seconds, "delay_frames": delay_frames,
            "gap_onset_wall_clock": gap_onset_wall,
            "av_onset_wall_clock": onset_wall_iso,
            "webcam_av_onset_wallclock": av_onset_wall,
            "gap_onset_lsl": L["gap_onset_lsl"], "av_onset_lsl": L["av_onset_lsl"],
            "audio_onset_lsl": L["audio_onset_lsl"], "block_end_lsl": L["block_end_lsl"],
            "beep_onset_lsl": L["beep_onset_lsl"], "beep_response_lsl": L["beep_response_lsl"],
            "beep_key": beep_key, "beep_rt_s": beep_rt,
            "audio_duration_s": round(audio_duration, 3),
            "block_target_s": round(btarget, 3),
            "block_duration_s": round(block_duration, 3),
            "refresh_hz": round(refresh_hz, 3),
            "probe_modality": probe["modality"], "probe_id": probe["probe_id"],
            "probe_question": probe["question"],
            "probe_correct_answer": probe["correct_answer"],
            "probe_response": probe["response"], "probe_rt_s": probe["rt_s"],
            "probe_correct": probe["correct"],
            "webcam_enabled": webcam_info["enabled"], "webcam_file": webcam_info["file"],
            "webcam_opened": webcam_info["opened"], "aborted": aborted,
        }

    # --- Trial loop ---------------------------------------------------------
    records: list[dict] = []
    n_correct = n_scored = 0
    for i in range(n_trials):
        rec = run_one_trial(i + 1, audio_plan[i], diff_plan[i])
        _append_behavior_csv(paths, rec)
        records.append(rec)
        if rec["probe_correct"] is not None:
            n_scored += 1
            n_correct += int(bool(rec["probe_correct"]))
        cc.log(f"Trial {i + 1}/{n_trials}: difficulty {rec['difficulty_name']}, "
               f"audio {rec['audio_stim_id']}, resets {rec['tetris_resets']} "
               f"-> probe {rec['probe_correct']}, beep RT {rec['beep_rt_s']}.")
        if rec["aborted"]:
            cc.log("Session aborted by user.")
            break
        if i < n_trials - 1:
            fixation.draw(); win.flip()
            core.wait(iti)

    # --- Teardown -----------------------------------------------------------
    if eng is not None:
        eng.marker("session_end", extra={"n_trials_run": len(records)})
    if webcam is not None:
        st = webcam.stop() or {}
        webcam_info.update(opened=st.get("opened", webcam_info["opened"]),
                           frames=st.get("frames"), duration_s=st.get("duration_s"))
        cc.log(f"Webcam stopped: {st.get('frames')} frames -> {webcam_info['file']}")

    lsl_info = eng.stream_info() if eng is not None else None
    _write_session_json(paths, subject, session_ts, seed, cfg, webcam_info, records,
                        session, lsl_info, os.path.relpath(lsl_backup, cc.BASE_DIR))
    acc = f"{n_correct}/{n_scored}" if n_scored else "n/a"
    cc.log(f"Session complete: {len(records)} trial(s), audio-probe accuracy {acc}.")

    visual.TextStim(win, text="Session complete.\nThank you.",
                    color=txt_color, height=0.06).draw()
    win.flip(); core.wait(1.5)
    _shutdown(eng, webcam, win, core)


def _shutdown(eng, webcam, win, core):
    try:
        if eng is not None:
            eng.close()
    except Exception:
        pass
    try:
        if webcam is not None:
            webcam.stop()
    except Exception:
        pass
    win.close()
    core.quit()


# ===========================================================================
# Beep response
# ===========================================================================
def _beep_response(win, event, core, eng, beep_buf, fixation, trigger, bcfg,
                   refresh_hz, rec_extra, L):
    """Play the beep (DAC-timestamped + photodiode pulse) and collect a speeded key
    press. RT is measured from a clock reset at the beep-onset flip. Returns (key, rt)
    — key is None if no response, 'escape' if the subject quit."""
    keys = list(bcfg.get("response_keys", ["space"]))
    window_s = float(bcfg.get("response_window_s", 1.5))
    pulse_frames = max(1, int(round(float(bcfg.get("photodiode_pulse_s", 0.05)) * refresh_hz)))

    eng.play(beep_buf)
    rt_clock = core.Clock()
    for pf in range(pulse_frames):                              # photodiode pulse @ beep
        fixation.draw()
        if trigger is not None:
            trigger.draw()
        win.flip()
        if pf == 0:
            rt_clock.reset()
            L["beep_onset_lsl"] = eng.wait_onset(timeout=0.4)
            eng.marker("beep_onset", timestamp=L["beep_onset_lsl"], extra=rec_extra)
    fixation.draw(); win.flip()                                 # square off

    event.clearEvents()
    got = event.waitKeys(maxWait=window_s, keyList=keys + ["escape"], timeStamped=rt_clock)
    if not got:
        return None, None
    key, rt = got[0]
    if key == "escape":
        return "escape", None
    L["beep_response_lsl"] = eng.marker(
        "beep_response", extra={**rec_extra, "key": key, "rt_s": round(rt, 4)})
    return key, round(rt, 4)


# ===========================================================================
# Probe presentation (audio yes/no — kept every trial)
# ===========================================================================
def _key_label(key: str) -> str:
    return {"left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN"}.get(key, key.upper())


def _run_yes_no(win, event, core, cfg, question: str, correct_present, txt_color) -> dict:
    from psychopy import visual
    keys_cfg = cfg["probe"]["keys"]
    yes_key, no_key = keys_cfg["yes"], keys_cfg["no"]
    quit_key = cfg["probe"].get("quit_key", "escape")
    header = cfg["probe"].get("instruction", "")
    _side = {"left": 0, "right": 2}
    opts = sorted([("YES", yes_key), ("NO", no_key)], key=lambda o: _side.get(o[1], 1))
    labels = "        ".join(f"{name} ({_key_label(k)})" for name, k in opts)
    prompt = f"{header}\n\n{question}\n\n{labels}"
    visual.TextStim(win, text=prompt, color=txt_color, height=0.05, wrapWidth=1.5,
                    alignText="center").draw()
    win.flip()
    event.clearEvents()
    clock = core.Clock()
    key, rt = event.waitKeys(keyList=[yes_key, no_key, quit_key], timeStamped=clock)[0]
    if key == quit_key:
        return {"response": None, "rt_s": None, "correct": None, "aborted": True}
    response = "yes" if key == yes_key else "no"
    correct = (None if correct_present is None
               else (response == "yes") == bool(correct_present))
    return {"response": response, "rt_s": round(rt, 4), "correct": correct, "aborted": False}


def _probe_audio(win, event, core, cfg, entry, words_file, block_limit, min_word_len,
                 rng, txt_color) -> dict:
    """Yes/no 'was WORD spoken?' probe; target chosen from words heard within
    `block_limit` s so it is valid even though only the first part of the clip played."""
    words = []
    try:
        with open(words_file, encoding="utf-8") as f:
            words = json.load(f).get("words", [])
    except Exception:
        pass
    p = cc.make_trial_audio_probe(words, entry.get("transcript", ""), min_word_len, rng,
                                  window_end_s=block_limit)
    if not p:
        return {"modality": "Audio", "probe_id": None, "question": None,
                "correct_answer": None, "response": None, "rt_s": None, "correct": None}
    res = _run_yes_no(win, event, core, cfg, p["question"], p["present"], txt_color)
    return {"modality": "Audio", "probe_id": p["target_word"], "question": p["question"],
            "correct_answer": "yes" if p["present"] else "no",
            "response": res["response"], "rt_s": res["rt_s"], "correct": res["correct"],
            "aborted": res.get("aborted", False)}


# ===========================================================================
# Webcam (session-level subprocess; copied verbatim from the parent experiment)
# ===========================================================================
def _start_webcam(cfg, subject, tag, win, visual, core, txt_color):
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
    ctrl = WebcamController(out_path, device=int(wcfg.get("device", 0)),
                            fps=float(wcfg.get("fps", 0) or 0),
                            resolution=tuple(res) if res else None,
                            fourcc=wcfg.get("fourcc", "mp4v"),
                            max_seconds=float(wcfg.get("max_seconds", 3600)))
    info["file"] = os.path.relpath(out_path, cc.BASE_DIR)
    visual.TextStim(win, text="Preparing camera…", color=txt_color, height=0.05).draw()
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
        win.close(); core.quit()
        raise RuntimeError(f"Webcam is required but failed to start: {msg}")
    cc.log(f"WARNING: webcam not recording ({msg}); continuing without it.")
    return None, info


# ===========================================================================
# Logging
# ===========================================================================
def _append_behavior_csv(paths: cc.Paths, record: dict) -> None:
    os.makedirs(paths.behavior_dir, exist_ok=True)
    fields = list(record.keys())
    csv_path = os.path.join(paths.behavior_dir, f"{record['subject']}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != fields:
            alt = os.path.join(paths.behavior_dir,
                               f"{record['subject']}_{record.get('timestamp', 'x')}.csv")
            cc.log(f"WARNING: {os.path.basename(csv_path)} has a different column schema; "
                   f"writing this row to {os.path.basename(alt)} to avoid corrupting it.")
            csv_path = alt
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(record)


def _save_game_record(paths: cc.Paths, subject: str, session_ts: str, trial_idx: int,
                      meta: dict) -> str:
    os.makedirs(paths.games_dir, exist_ok=True)
    out = os.path.join(paths.games_dir, f"{subject}_{session_ts}_t{trial_idx:03d}.json")
    payload = {"subject": subject, "session_timestamp": session_ts, "trial": trial_idx, **meta}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def _write_session_json(paths, subject, session_ts, seed, cfg, webcam_info, records,
                        session="", lsl_info=None, lsl_backup=None) -> None:
    os.makedirs(paths.behavior_dir, exist_ok=True)
    out = os.path.join(paths.behavior_dir, f"{subject}_{session_ts}_session.json")
    scored = [r for r in records if r["probe_correct"] is not None]
    n_ok = sum(1 for r in scored if r["probe_correct"])
    payload = {
        "subject": subject, "session": session, "session_timestamp": session_ts,
        "seed": seed, "experiment": "load_modulation",
        "n_trials_run": len(records), "n_scored": len(scored), "n_correct": n_ok,
        "accuracy": (round(n_ok / len(scored), 4) if scored else None),
        "block_seconds": cfg["experiment"]["block_seconds"],
        "difficulty_levels": cfg["difficulty"]["levels"],
        "lsl": lsl_info, "lsl_backup_log": lsl_backup,
        "config": cfg, "webcam": webcam_info, "trials": records,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    cc.log(f"Session log -> {os.path.join(paths.behavior_dir, subject + '.csv')}")
    cc.log(f"            -> {out}")


# ===========================================================================
# Orchestration
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Run a load-modulation Tetris+audio session.")
    ap.add_argument("--subject", default="sub01")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--trials", type=int, default=None, help="Override experiment.n_trials.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--test-trigger", action="store_true",
                    help="Blink only the photodiode square, then exit.")
    ap.add_argument("--test-audio", action="store_true",
                    help="Open the LSL audio engine, play test beeps, then exit.")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    if args.test_trigger:
        _test_trigger(cfg); return
    if args.test_audio:
        _test_audio(cfg); return
    if args.trials is not None:
        cfg["experiment"]["n_trials"] = args.trials
    paths = cc.Paths.from_config(cfg).ensure()
    manifest = cc.read_manifest(paths.manifest)
    seed = (args.seed if args.seed is not None else int(cfg["experiment"].get("seed", 0)))
    rng = random.Random(seed)
    cc.log(f"Subject {args.subject}, seed {seed}, {cfg['experiment'].get('n_trials')} trials.")
    run_session(cfg, paths, manifest, args.subject, rng, seed)


if __name__ == "__main__":
    main()
