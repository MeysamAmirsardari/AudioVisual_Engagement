# cross_modal_attention

A cross-modal (audio **vs.** visual) selective-attention paradigm in PsychoPy,
designed for EEG. A subject chooses which stream to attend, watches a muted
gameplay video while a continuous speech clip plays, and then answers a yes/no
attention-check about the **attended** stream.

The stimulus library is **prepared once** and reused every session — no
audio/alignment is regenerated at run time.

## Layout

| File | Purpose |
|------|---------|
| `config.yaml` | All parameters (paths, timing, dataset, probes). Edit this, not the code. |
| `build_stimuli.py` | **Run once.** Downloads real speech + word onsets into `stims/` and writes `stims/manifest.json`. |
| `run_block.py` | Runs one experimental block from the prepared library; logs to `behavior/`. |
| `cma_common.py` | Shared config / manifest / path helpers. |
| `cma_webcam.py` | Separate-process webcam recorder (used by `run_block.py`). |
| `tetris/` | Self-playing black-and-white Tetris (the default visual stimulus). |
| `build_video_segments.py` | (Only for `visual.mode: video`) detects the game segments of the recorded clip. |
| `stims/` | Prepared audio (`*.wav` + `*.words.json/csv`) + `manifest.json`. *(git-ignored)* |
| `behavior/` | Per-subject behavioural logs (`<subject>.csv` + per-block JSON). *(git-ignored)* |
| `webcam/` | Per-block webcam recordings + frame-timestamp sidecars. *(git-ignored)* |
| `stim_files/tetris.mp4` | The visual stimulus. |

## Install

```bash
source .venv/bin/activate            # Python 3.11
python -m pip install -r requirements.txt
# (or pin exact versions with requirements.lock.txt)
```

## 1. Build the stimulus library (once)

```bash
python build_stimuli.py              # builds build.n_clips (default 10) clips
python build_stimuli.py --add 5      # append 5 more later
python build_stimuli.py --list       # show what's prepared
python build_stimuli.py --source tts # alternative: local gTTS + faster-whisper
```

**Audio source:** [`gilkeyio/librispeech-alignments`](https://huggingface.co/datasets/gilkeyio/librispeech-alignments)
— real LibriSpeech audiobook speech with Montreal-Forced-Aligner **word onsets**
(CC-BY-4.0). Same-speaker utterances are concatenated to ~30 s clips; word onsets
are time-shifted accordingly. Each clip gets an auto-generated yes/no word probe
(one true-target and one false-target variant).

## 1b. Detect the usable video segments (once, video mode)

```bash
python build_video_segments.py            # writes visual.video.segments_file
python build_video_segments.py --preview  # also print each segment
```

The tetris clip has non-game (colour) sections that must not be shown. Gameplay
is black-and-white, so this classifies frames by colour saturation and writes the
contiguous grayscale intervals long enough to hold a block. Random cuts are then
drawn ONLY from those segments (currently ~74 % of the clip; colour menus/intros
are skipped). Re-run it if you change the video or `visual.video.filter`.

## 2. Run a session

```bash
python run_block.py --subject sub01              # experiment.n_trials trials
python run_block.py --subject sub01 --trials 10  # override the trial count
python run_block.py --subject sub01 --seed 42    # reproducible clips/cuts/probes
```

One session runs `experiment.n_trials` (default **60**) trials back-to-back.
Per trial: **focus cue/choice** → **jittered fixation delay** (base
`gap_seconds`, default 2 s, + Gaussian jitter) → synchronous visual stimulus +
speech clip (PTB flip-scheduled onset, `experiment.block_seconds` long, default
**20 s**) → attention probe about the **attended** stream → an inter-trial
fixation. Audio clips are assigned in a balanced shuffle. Results: one row per
trial in `behavior/<subject>.csv` plus a per-session JSON.

**Timing jitter (`experiment.jitter`):** the pre-stimulus delay is jittered per
trial to decorrelate onsets for EEG — Gaussian `std_ms` (default 150), clamped to
±`max_ms` (default 300). The log records the actual `delay_seconds`,
`delay_jitter_ms`, `delay_frames`, and the `instruction_seconds` (measured length
of the cue/choice screen).

**Focus assignment (`experiment.focus_mode`):**
- **`auto` (default)** — the experiment assigns focus, **balanced 50/50 and
  shuffled** (30 Audio / 30 Visual for 60 trials), and cues the subject
  ("Attend to the AUDIO") for `cue_seconds`.
- **`manual`** — the subject chooses each trial (← Audio / → Visual).

**Photodiode trigger (`trigger`):** a white square in the bottom-right corner is
shown for the whole 1 s gap — its **rising edge = gap onset**, **falling edge =
audiovisual onset** — for hardware-precise EEG timing. Size/corner configurable.

## Visual stimulus (`visual.mode`)

- **`tetris` (default)** — a **self-playing, black-and-white Tetris** (in
  `tetris/`), watched passively. Deterministic (seeded per trial), grayscale
  (no colour confound), compact and **centred with a fixation dot** to limit eye
  movements. Cleared rows briefly **flash**; attend-visual is to **count those
  flashes**, scored automatically against the known count (`tetris_clears`). No
  recorded video, no non-game sections, no random-cut machinery. Tune board
  size/speed under `visual.tetris`. (Preview: `python tetris/tetris_game.py
  --save-frames 6` writes frames to `/tmp`.)
- **`gabor`** — a small, centred Gabor patch that rotates and randomly **reverses
  direction**; attend-visual counts the reversals (numeric probe, auto-scored).
- **`video`** — the pre-recorded `tetris.mp4` with random cuts from the
  black-and-white game segments (needs `build_video_segments.py`; attend-visual
  trials are logged unscored — see the caveat below).

All three keep the eyes centred; the auto cue reminds the subject of the task
("Count the row clears" / "reversals" / "Listen for the words").

The auditory task (either mode) is a yes/no "was *WORD* spoken?" probe whose
target is chosen at trial time from words actually heard within the played
window — so it stays valid even though only the first `block_seconds` of each
~30 s clip is played.

## Webcam (optional)

If `webcam.enabled` is true in `config.yaml`, each block records the participant
to `webcam/<subject>_<ts>_b<block>.mp4` in a **separate process** (so it never
disturbs the draw-loop timing). Alongside the video it writes:
`<name>.frames.csv` — a wall-clock timestamp per frame — and `<name>.status.json`.
The behaviour log stores `webcam_av_onset_wallclock`, so you can find the exact
frame at audiovisual onset for EEG alignment.

macOS: grant Camera access to your terminal/Python under **System Settings →
Privacy & Security → Camera** (you'll be prompted on first run). If the camera
can't open and `webcam.required` is false, the block runs without recording.

## ⚠️ Visual attention-check in `video` mode with random cuts

With `visual.video.random_start: true`, every trial shows a *different* random
cut, so a single pre-filled answer in `visual.video.probes[].correct` can't be
correct for all of them. As shipped, **attend-visual trials are logged
UNSCORED** (`probe_correct = null`) in this mode — attend-audio trials are scored
normally. Options if you need a scored visual check:

- switch to `visual.mode: gabor` (reversal count, auto-scored, also fixes the
  eye-movement confound); or
- turn off random cuts (`random_start: false`) and fill `probes[].correct` for
  the fixed cut; or
- ask for a per-cut visual probe (needs a way to know the ground truth per cut).
