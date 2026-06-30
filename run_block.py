#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Modal Attention EEG Experiment -- Single Block Runner
===========================================================

This self-contained script runs ONE experimental block of a cross-modal
(audio vs. visual) attention paradigm in PsychoPy.

It solves a "missing asset" problem entirely with local compute on Apple
Silicon:

    * The VISUAL stimulus already exists  ->  stim_files/tetris.mp4
    * The AUDITORY stimulus does NOT exist yet. So, BEFORE the window opens,
      the script:
          1. Synthesises a continuous English speech track from a hardcoded
             paragraph (gTTS by default; offline pyttsx3 fallback).
          2. Runs a LOCAL Whisper model (faster-whisper) to extract strict,
             millisecond-level word-onset/offset timestamps.
          3. Writes those boundaries to both .json and .csv.

Experimental flow (executed in this exact order):
    1. Instruction screen -> wait for LEFT (Audio) or RIGHT (Visual).
    2. Buffer gap         -> fixation cross for EXACTLY 1.000 s (frame-counted).
    3. Audiovisual block  -> tetris.mp4 (muted) + generated speech, started
                             synchronously and run concurrently.
    4. Data logging       -> choice, AV-onset timestamp, word-boundary path.

Author: generated for an Apple Silicon (M-series) / macOS target.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Everything tunable lives here so the experimental logic below stays clean.

# Resolve paths relative to THIS file so the script can be launched from
# anywhere (e.g. an IDE "Run" button or `python run_block.py`).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "stim_files", "tetris.mp4")

# Generated assets (audio + boundaries) live here; created on demand.
GEN_DIR = os.path.join(BASE_DIR, "generated")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Standardised auditory-stimulus artefacts.
SPEECH_WAV = os.path.join(GEN_DIR, "speech.wav")          # what PsychoPy plays
BOUNDARY_JSON = os.path.join(GEN_DIR, "word_boundaries.json")
BOUNDARY_CSV = os.path.join(GEN_DIR, "word_boundaries.csv")

# --- Text to be spoken -------------------------------------------------------
# Hardcoded English paragraph. Keep it continuous and clearly articulated so
# the forced-aligner can find crisp word boundaries.
SPEECH_TEXT = (
    "Attention is the quiet gatekeeper of perception. "
    "At every moment your senses deliver far more information than the mind can hold, "
    "so the brain must choose what to keep and what to let slip away. "
    "When you listen closely to a single voice in a crowded room, "
    "the other sounds do not vanish, they simply fade into the background. "
    "The same is true for vision, where a single moving shape can capture your focus "
    "while everything around it dissolves into a soft and unremarkable blur. "
    "In this experiment you will hold two streams in mind at once, "
    "one for the ear and one for the eye, "
    "and your task is simply to decide, before the block begins, "
    "which of the two will receive your full attention."
)

# --- TTS engine selection ----------------------------------------------------
# "gtts"    -> Google Text-to-Speech. Natural prosody, needs an internet
#              connection, returns MP3 (decoded locally to WAV).
# "pyttsx3" -> Fully OFFLINE macOS system voice (NSSpeechSynthesizer).
TTS_ENGINE = "gtts"

# --- Whisper (forced alignment) ---------------------------------------------
# faster-whisper model size. "small.en" is a good speed/accuracy trade-off for
# English on Apple Silicon CPU; use "base.en" for more speed, "medium.en" for
# more accuracy. CTranslate2 has no Metal backend, so we run on CPU with int8.
WHISPER_MODEL = "small.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# --- Display / timing --------------------------------------------------------
FULLSCREEN = True
BG_COLOR = "black"
TEXT_COLOR = "white"
GAP_SECONDS = 1.0          # The "buffer gap" duration (frame-counted).
BLOCK_SECONDS = 30.0       # Hard cap on audiovisual playback: BOTH the video
                           # and the audio are stopped at exactly this point,
                           # regardless of their underlying file durations.
FALLBACK_REFRESH = 60.0    # Hz, used only if PsychoPy can't measure the monitor.

INSTRUCTION_TEXT = (
    "Decide your attentional focus:\n\n"
    "Press the LEFT arrow for Audio,\n"
    "or the RIGHT arrow for Visual."
)


# ===========================================================================
# STAGE 1 -- AUDIO SYNTHESIS
# ===========================================================================
def _log(msg: str) -> None:
    """Tiny timestamped progress logger for the preprocessing phase."""
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _normalise_to_wav(src_path: str, dst_wav: str) -> float:
    """
    Decode an arbitrary audio file (MP3/AIFF/WAV) to a mono float WAV that both
    PsychoPy's audio backend and faster-whisper can consume reliably.

    Returns the audio duration in seconds.

    No system `ffmpeg` binary is required: modern `soundfile` (libsndfile >=1.1)
    decodes MP3 directly; `librosa` is used only as a fallback.
    """
    import numpy as np
    import soundfile as sf

    data = None
    sr = None
    try:
        # Primary path: libsndfile via soundfile (handles WAV/AIFF/FLAC/MP3).
        data, sr = sf.read(src_path, dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        _log(f"soundfile could not read '{os.path.basename(src_path)}' "
             f"({exc}); falling back to librosa.")
        import librosa
        data, sr = librosa.load(src_path, sr=None, mono=True)

    # Collapse to mono if the decoder returned stereo.
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)

    # Light peak-normalisation so playback level is consistent regardless of TTS.
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        data = (data / peak) * 0.95

    sf.write(dst_wav, data.astype("float32"), int(sr), subtype="PCM_16")
    duration = float(len(data)) / float(sr) if sr else 0.0
    return duration


def generate_speech_audio(text: str, out_wav: str, engine: str = "gtts") -> float:
    """
    Synthesize `text` to a standardized mono WAV at `out_wav`.

    Parameters
    ----------
    text   : the paragraph to speak.
    out_wav: destination .wav path.
    engine : "gtts" (online, natural) or "pyttsx3" (offline, system voice).

    Returns
    -------
    The duration of the generated audio in seconds.
    """
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    _log(f"Generating speech with engine='{engine}' "
         f"({len(text.split())} words)...")

    if engine == "gtts":
        # gTTS writes MP3; we then decode it locally to a normalised WAV.
        from gtts import gTTS
        tmp_mp3 = os.path.splitext(out_wav)[0] + "_raw.mp3"
        gTTS(text=text, lang="en", slow=False).save(tmp_mp3)
        _log("gTTS MP3 received; decoding to WAV...")
        duration = _normalise_to_wav(tmp_mp3, out_wav)

    elif engine == "pyttsx3":
        # Fully offline. The macOS 'nsss' driver typically writes AIFF; we then
        # normalise to WAV. runAndWait() must be called to flush the file.
        import pyttsx3
        tmp_aiff = os.path.splitext(out_wav)[0] + "_raw.aiff"
        eng = pyttsx3.init()
        eng.setProperty("rate", 175)  # words per minute
        eng.save_to_file(text, tmp_aiff)
        eng.runAndWait()
        eng.stop()
        if not (os.path.exists(tmp_aiff) and os.path.getsize(tmp_aiff) > 0):
            raise RuntimeError(
                "pyttsx3 produced no audio. Try TTS_ENGINE='gtts' instead."
            )
        _log("pyttsx3 audio received; normalising to WAV...")
        duration = _normalise_to_wav(tmp_aiff, out_wav)

    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")

    _log(f"Speech WAV written: {out_wav}  ({duration:.2f} s)")
    return duration


# ===========================================================================
# STAGE 2 -- FORCED ALIGNMENT (WORD BOUNDARIES)
# ===========================================================================
def extract_word_boundaries(wav_path: str,
                            json_path: str,
                            csv_path: str,
                            model_size: str = WHISPER_MODEL) -> list[dict]:
    """
    Run a LOCAL Whisper model with word-level timestamps over `wav_path` and
    write strict word boundaries to both JSON and CSV.

    All times are in SECONDS relative to the start of the audio file, which --
    because audio onset == audiovisual onset in this paradigm -- is identical
    to the time relative to the start of tetris.mp4 playback.

    Returns the list of boundary dicts.
    """
    from faster_whisper import WhisperModel

    _log(f"Loading faster-whisper model '{model_size}' "
         f"(device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})...")
    model = WhisperModel(model_size, device=WHISPER_DEVICE,
                         compute_type=WHISPER_COMPUTE)

    _log("Transcribing with word-level timestamps...")
    # word_timestamps=True asks Whisper for per-word start/end via its
    # cross-attention alignment. We pin the language to English for stability.
    segments, info = model.transcribe(
        wav_path,
        language="en",
        word_timestamps=True,
        beam_size=5,
    )

    boundaries: list[dict] = []
    for seg in segments:
        # Each segment carries a list of Word objects when word_timestamps=True.
        for w in (seg.words or []):
            word = w.word.strip()
            if not word:
                continue
            start_s = float(w.start)
            end_s = float(w.end)
            boundaries.append({
                "word": word,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "start_ms": int(round(start_s * 1000.0)),
                "end_ms": int(round(end_s * 1000.0)),
                "duration_ms": int(round((end_s - start_s) * 1000.0)),
                "confidence": round(float(getattr(w, "probability", 0.0) or 0.0), 3),
            })

    if not boundaries:
        raise RuntimeError(
            "Whisper returned no words. Check that the audio is non-empty."
        )

    # --- Write JSON ---------------------------------------------------------
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    payload = {
        "audio_file": os.path.relpath(wav_path, BASE_DIR),
        "language": getattr(info, "language", "en"),
        "model": model_size,
        "n_words": len(boundaries),
        "note": "All times are seconds/ms relative to audio start == AV onset.",
        "words": boundaries,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # --- Write CSV ----------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "start_s", "end_s",
                        "start_ms", "end_ms", "duration_ms", "confidence"],
        )
        writer.writeheader()
        writer.writerows(boundaries)

    _log(f"Extracted {len(boundaries)} word boundaries.")
    _log(f"  JSON -> {json_path}")
    _log(f"  CSV  -> {csv_path}")
    return boundaries


# ===========================================================================
# STAGE 3 -- THE EXPERIMENT (PsychoPy)
# ===========================================================================
def run_experiment(subject: str,
                   speech_wav: str,
                   boundary_json: str,
                   audio_duration: float) -> None:
    """
    Open the PsychoPy window and run the single block:
        instruction -> 1.0 s fixation gap -> synchronous audiovisual playback.

    PsychoPy is imported INSIDE this function so that all heavy preprocessing
    (and its console output) completes before any window/graphics context opens.
    """
    # --- Select the precise PTB audio backend BEFORE importing sound --------
    # The psychtoolbox ('ptb') backend gives low-latency, schedulable audio,
    # which is what lets us align audio onset to an exact video frame flip.
    from psychopy import prefs
    prefs.hardware["audioLib"] = ["ptb", "sounddevice", "pyo"]

    from psychopy import visual, core, event, logging, monitors
    from psychopy.sound import Sound
    from psychopy.constants import FINISHED

    logging.console.setLevel(logging.WARNING)

    # --- Window -------------------------------------------------------------
    # A named monitor keeps PsychoPy from warning about an unknown display.
    mon = monitors.Monitor("expMonitor")
    win = visual.Window(
        fullscr=FULLSCREEN,
        color=BG_COLOR,
        units="height",        # resolution-independent sizing
        monitor=mon,
        allowGUI=False,
        winType="pyglet",
    )
    win.mouseVisible = False

    # --- Measure the true refresh rate for frame-accurate gap timing --------
    # getActualFrameRate() returns None if it can't get a stable estimate.
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                       nWarmUpFrames=10, threshold=1)
    refresh_hz = measured if measured else FALLBACK_REFRESH
    gap_frames = int(round(GAP_SECONDS * refresh_hz))
    _log(f"Refresh rate ~{refresh_hz:.2f} Hz -> "
         f"{gap_frames} frames for the {GAP_SECONDS:.3f} s gap.")

    # --- Reusable visual components -----------------------------------------
    instruction = visual.TextStim(win, text=INSTRUCTION_TEXT, color=TEXT_COLOR,
                                  height=0.05, wrapWidth=1.4, alignText="center")
    fixation = visual.TextStim(win, text="+", color=TEXT_COLOR, height=0.12)

    # --- Pre-load the auditory stimulus into the audio buffer ---------------
    # Building the Sound object now avoids first-play latency during the trial.
    snd = Sound(speech_wav, stereo=True, hamming=False)

    # --- Pre-load the (muted) movie -----------------------------------------
    # noAudio=True strips the original soundtrack; volume=0 is belt-and-braces.
    movie = visual.MovieStim(
        win,
        VIDEO_PATH,
        noAudio=True,
        volume=0.0,
        loop=False,
        # Scale to fit the height-based canvas while preserving aspect ratio.
        size=None,
    )

    global_clock = core.Clock()

    # ---------------------------------------------------------------
    # FLOW STEP 1 -- Instruction screen, wait for LEFT / RIGHT
    # ---------------------------------------------------------------
    instruction.draw()
    win.flip()
    event.clearEvents()
    keys = event.waitKeys(keyList=["left", "right", "escape"])
    if "escape" in keys:
        _abort(win, core)
    choice_key = keys[0]
    choice_label = "Audio" if choice_key == "left" else "Visual"
    _log(f"Subject chose: {choice_key.upper()} ({choice_label})")

    # ---------------------------------------------------------------
    # FLOW STEP 2 -- The 1.0 s buffer gap (frame-counted = frame-accurate)
    # ---------------------------------------------------------------
    for _ in range(gap_frames):
        fixation.draw()
        win.flip()
    # Keep the fixation on screen for the final frame's duration as well.

    # ---------------------------------------------------------------
    # FLOW STEP 3 -- Synchronous audiovisual playback
    # ---------------------------------------------------------------
    # We schedule the audio to begin at the exact moment of the next window
    # flip (the flip that reveals the movie's first frame). PsychoPy exposes
    # the predicted next-flip time in the PTB time-base specifically for this.
    av_onset_global = None
    onset_wall_clock = _dt.datetime.now().isoformat(timespec="milliseconds")

    try:
        when_ptb = win.getFutureFlipTime(clock="ptb")  # next flip, ptb timebase
        snd.play(when=when_ptb)                          # scheduled, sample-accurate
    except Exception as exc:
        # If scheduling isn't supported, fall back to immediate play. The first
        # video frame still appears on the same flip, so jitter stays sub-frame.
        _log(f"Scheduled audio unavailable ({exc}); starting audio immediately.")
        snd.play()

    movie.play()
    block_clock = core.Clock()       # measures elapsed playback from AV onset
    flip_t = win.flip()              # <- AV onset: first movie frame is shown here
    block_clock.reset()              # t=0 is pinned to the onset flip
    av_onset_global = global_clock.getTime()
    onset_wall_clock = _dt.datetime.now().isoformat(timespec="milliseconds")
    _log(f"AUDIOVISUAL ONSET at t={av_onset_global:.4f}s "
         f"(win flip clock={flip_t:.4f}); capping block at {BLOCK_SECONDS:.1f}s.")

    # --- Concurrent, capped draw loop ---------------------------------------
    # The audio plays in the background; we keep pumping movie frames to the
    # window. The loop ends at the FIRST of: the BLOCK_SECONDS cap, the video
    # finishing, or an escape press. We stop on the first flip at/after the cap
    # so the on-screen video and the audio are truncated together at ~30 s.
    aborted = False
    while movie.status != FINISHED and block_clock.getTime() < BLOCK_SECONDS:
        movie.draw()
        win.flip()
        if event.getKeys(keyList=["escape"]):
            aborted = True
            break

    block_duration = block_clock.getTime()

    # --- Tidy up media: stop BOTH streams at the cap ------------------------
    try:
        snd.stop()
    except Exception:
        pass
    try:
        movie.stop()
    except Exception:
        pass
    _log(f"Block ended at {block_duration:.3f}s "
         f"({'aborted' if aborted else 'cap/finished'}).")

    # ---------------------------------------------------------------
    # FLOW STEP 4 -- Data logging
    # ---------------------------------------------------------------
    _write_trial_log(
        subject=subject,
        choice_key=choice_key,
        choice_label=choice_label,
        av_onset_global=av_onset_global,
        onset_wall_clock=onset_wall_clock,
        boundary_json=boundary_json,
        speech_wav=speech_wav,
        audio_duration=audio_duration,
        refresh_hz=refresh_hz,
        gap_frames=gap_frames,
        block_seconds_cap=BLOCK_SECONDS,
        block_duration_s=block_duration,
        aborted=aborted,
    )

    # --- Goodbye ------------------------------------------------------------
    bye = visual.TextStim(win, text="Block complete.\nThank you.",
                          color=TEXT_COLOR, height=0.06)
    bye.draw()
    win.flip()
    core.wait(1.5)
    win.close()
    core.quit()


def _abort(win, core):
    """Clean shutdown on escape during instructions."""
    _log("Aborted by user.")
    win.close()
    core.quit()
    sys.exit(0)


def _write_trial_log(**row) -> None:
    """
    Append a single-row trial record to data/<subject>_block.csv and also drop a
    standalone JSON for this run. Captures the four required fields plus context.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    subject = row["subject"]
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    record = {
        "subject": subject,
        "timestamp": ts,
        # --- the four explicitly-required fields ---
        "choice_key": row["choice_key"],                 # 'left' / 'right'
        "choice_label": row["choice_label"],             # 'Audio' / 'Visual'
        "av_onset_global_s": round(row["av_onset_global"], 4)
        if row["av_onset_global"] is not None else None,
        "av_onset_wall_clock": row["onset_wall_clock"],
        "word_boundary_file": os.path.relpath(row["boundary_json"], BASE_DIR),
        # --- helpful context ---
        "video_file": os.path.relpath(VIDEO_PATH, BASE_DIR),
        "audio_file": os.path.relpath(row["speech_wav"], BASE_DIR),
        "audio_duration_s": round(row["audio_duration"], 3),
        "refresh_hz": round(row["refresh_hz"], 3),
        "gap_frames": row["gap_frames"],
        "gap_seconds_target": GAP_SECONDS,
        "block_seconds_cap": row["block_seconds_cap"],
        "block_duration_s": round(row["block_duration_s"], 3),
        "aborted": row["aborted"],
    }

    # Per-run JSON.
    json_path = os.path.join(DATA_DIR, f"{subject}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # Cumulative CSV (one row per block).
    csv_path = os.path.join(DATA_DIR, f"{subject}_block.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    _log(f"Trial logged -> {csv_path}")
    _log(f"             -> {json_path}")


# ===========================================================================
# ORCHESTRATION
# ===========================================================================
def preprocess() -> float:
    """
    Run the two preprocessing stages (audio synthesis + forced alignment) that
    must complete BEFORE the PsychoPy window opens. Returns audio duration (s).
    """
    print("=" * 70)
    print("PRE-FLIGHT: building the missing auditory asset locally")
    print("=" * 70)

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Visual stimulus not found: {VIDEO_PATH}")

    # Stage 1: synthesise speech -> WAV.
    audio_duration = generate_speech_audio(SPEECH_TEXT, SPEECH_WAV,
                                           engine=TTS_ENGINE)

    # Stage 2: local Whisper forced alignment -> word boundaries.
    extract_word_boundaries(SPEECH_WAV, BOUNDARY_JSON, BOUNDARY_CSV,
                            model_size=WHISPER_MODEL)

    print("=" * 70)
    print("PRE-FLIGHT COMPLETE. Opening PsychoPy window next...")
    print("=" * 70)
    return audio_duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal attention block.")
    parser.add_argument("--subject", default="sub01",
                        help="Subject/participant identifier (default: sub01).")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Reuse existing generated audio + boundaries.")
    args = parser.parse_args()

    # --- Preprocessing (loud, before any graphics) --------------------------
    if args.skip_preprocess and os.path.exists(SPEECH_WAV) \
            and os.path.exists(BOUNDARY_JSON):
        _log("Skipping preprocessing; reusing existing assets.")
        # Recover duration from the existing WAV.
        import soundfile as sf
        info = sf.info(SPEECH_WAV)
        audio_duration = info.frames / float(info.samplerate)
    else:
        audio_duration = preprocess()

    # --- Run the block ------------------------------------------------------
    run_experiment(
        subject=args.subject,
        speech_wav=SPEECH_WAV,
        boundary_json=BOUNDARY_JSON,
        audio_duration=audio_duration,
    )


if __name__ == "__main__":
    main()
