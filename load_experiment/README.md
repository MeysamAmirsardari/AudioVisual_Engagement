# Load-modulation audio-visual experiment

A variant of the cross-modal attention experiment. **There is no attention
instruction.** On every trial the subject simply *plays* a Tetris game whose
**difficulty varies trial-to-trial** (very easy → super hard, balanced and randomly
ordered). A speech clip plays in the background throughout; after each block the
subject answers **one yes/no audio-comprehension question** and responds to a **short
beep**. Game difficulty is an implicit cognitive-load manipulation; audio-probe
accuracy and beep reaction time index the spare capacity left for the audio.

This directory is self-contained (its own `run_block.py`, `config.yaml`, `tetris/`,
`cma_common.py`, `cma_webcam.py`, `reconstruct_tetris.py`, `lsl_audio.py`) and reuses
the **shared speech library** in `../stims` (built once by the parent project's
`build_stimuli.py`).

## Trial structure (40 s block)

1. **Ready** — fixation + trial counter (auto-advances after `ready_seconds`; no cue).
2. **Gap** — jittered, frame-counted fixation; the **photodiode square is ON** (rising
   edge = gap onset, falling edge = AV onset).
3. **Block (40 s)** — the subject plays Tetris (with an on-screen SCORE + countdown
   TIMER); **topping out restarts the game** and
   play continues. A background speech clip plays via the low-jitter LSL audio engine.
4. **Beep** — a short tone; the subject responds as fast as possible (SPACE). Onset is
   tagged by a DAC-timestamped LSL marker **and** a photodiode-square pulse.
5. **Audio probe** — yes/no "was WORD spoken?" about the clip (every trial).

## Low-jitter audio + LSL (the important part)

`lsl_audio.py` is a single **PortAudio (sounddevice) callback engine** that *both* plays
the audio and streams it to LSL:

- Playback runs in PortAudio's real-time thread, **decoupled from the video loop** — a
  dropped frame never perturbs audio timing.
- Every block is timestamped with the hardware **DAC output time**
  (`outputBufferDacTime`) mapped to the LSL clock, so recorded timestamps reflect *when
  sound actually leaves the card* — bypassing sound-card buffering latency/jitter.
- Each clip is **resampled once** to the device's native rate (no on-the-fly resampling
  jitter).
- LSL pushing happens on a **worker thread**, so it can never stall/glitch the audio.

Validated offline: the LSL audio stream's first-sample timestamp matches the discrete
onset marker to **~0.1 ms**.

Three redundant synchronisation paths + a backup log:

| Path | Stream / file | What it times |
|---|---|---|
| Hardware visual | photodiode square | gap onset, AV onset, beep pulse |
| LSL markers | `ExpAudioMarkers` | every discrete event (JSON) |
| LSL audio | `ExpAudio` | the waveform, DAC-timestamped |
| Local backup | `behavior/<subj>_<ts>_lsl_events.jsonl` | every marker, even if no recorder runs |

> For **absolute** hardware audio-onset timing you can additionally split the audio
> output into a trigger box / the amplifier's aux channel. The DAC-timestamped LSL
> stream is the best software-only approach and is what most EEG rigs use.

## Recording setup (do this before each session)

1. Start your **EEG LSL stream** (amplifier's LSL app).
2. Open **LabRecorder**, tick: your EEG stream, **`ExpAudio`**, **`ExpAudioMarkers`**
   (and any others). Start recording **before** launching the session so no onset
   blocks are missed.
3. Tape the **photodiode** over the configured corner (default bottom-right;
   `--test-trigger` to check placement).

## Running

```bash
# from the project root (script dir is auto-added to the path):
python load_experiment/run_block.py --test-audio        # LSL audio engine + streams self-test
python load_experiment/run_block.py --test-trigger      # photodiode square placement
python load_experiment/run_block.py --subject test --trials 3   # quick dry run (no GUI)

# real session (collects metadata, organises outputs under load_experiment/data/<subj>/<sess>/):
python load_experiment/run.py --subject P01 --session 01
```

Requirements (already in the project `.venv`): `psychopy`, `pylsl`, `sounddevice`,
`soundfile`, `scipy`, `numpy`, `pyyaml` (+ `opencv-python` for the webcam).

## Outputs (per session, under `data/<subject>/<session>/`)

- `behavior/<subject>.csv` — one row per trial (difficulty, all onset times incl. LSL,
  beep key + RT, audio-probe result, Tetris stats).
- `behavior/<subject>_<ts>_session.json` — every trial + **full config snapshot** + LSL
  stream metadata + difficulty tiers.
- `behavior/<subject>_<ts>_lsl_events.jsonl` — backup of every LSL marker.
- `games/<subject>_<ts>_tNNN.json` — deterministic Tetris record (seed + keystrokes +
  difficulty + display frame times) → `reconstruct_tetris.py` renders a faithful mp4.
- `webcam/…` — optional face video with a per-frame timestamp sidecar.

## Configuration (`config.yaml`)

Key blocks: `difficulty.levels` (the tiers), `lsl` (device/blocksize/latency/stream
names), `beep` (freq/duration/response), `probe.audio`, `visual.tetris`, `trigger`
(photodiode), `webcam`. `experiment.block_seconds` = 40, `experiment.seed` fixes the
whole session (audio order, difficulty order, gap jitter, every Tetris seed).
