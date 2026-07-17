#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lsl_audio.py — low-jitter audio playback + LSL streaming for the load experiment.

A SINGLE PortAudio (sounddevice) callback engine that BOTH plays audio to the sound
card AND streams the exact waveform to LSL. The trick for minimum jitter and correct
synchronisation:

  * Playback runs in PortAudio's own real-time audio thread (a callback), decoupled
    from the PsychoPy draw loop — so dropped video frames never perturb audio timing.
  * Every audio block is timestamped with the hardware DAC output time
    (`outputBufferDacTime`, the moment the first sample physically leaves the card),
    mapped to the LSL clock via a measured offset. The recorded LSL timestamps thus
    reflect WHEN SOUND ACTUALLY PLAYS, bypassing the sound card's buffering latency
    and its jitter — you recover true onset times post-hoc even if there is a fixed
    output delay.
  * The clip is resampled ONCE to the device's native rate, so CoreAudio never has to
    resample on the fly (a source of variable latency).
  * LSL pushing happens on a separate worker thread fed by a lock-free queue, so a
    slow push can never stall — or glitch — the audio callback.

Three streams for redundant synchronisation:
  1. an LSL "Audio" stream (the mono waveform, regular rate, DAC-timestamped),
  2. an LSL "Markers" stream (discrete JSON events: onsets, beep, responses, …),
  3. a local backup event log (JSON lines) so event timing survives even if no LSL
     recorder (e.g. LabRecorder) is running.

The engine is opened ONCE per session (no per-trial device opening) and each clip is
played by swapping the active buffer, so there is no per-trial device-setup jitter.
"""

from __future__ import annotations

import json
import queue
import threading

import numpy as np

DEFAULT_BLOCK = 256          # frames/callback; ~5.8 ms @ 44.1 kHz (low latency)


class LSLAudioEngine:
    def __init__(self, device=None, blocksize=DEFAULT_BLOCK, latency="low",
                 audio_name="ExpAudio", marker_name="ExpAudioMarkers",
                 source_id="load_experiment_audio", backup_path=None,
                 out_channels=None, amp=1.0):
        import sounddevice as sd
        from pylsl import StreamInfo, StreamOutlet, local_clock

        self._sd = sd
        self._local_clock = local_clock
        dev = (sd.query_devices(kind="output") if device is None
               else sd.query_devices(device, "output"))
        self.samplerate = int(dev["default_samplerate"])
        self.device = device
        self.out_channels = int(out_channels or min(2, int(dev["max_output_channels"])))
        self.blocksize = int(blocksize)
        self.amp = float(amp)

        # ---- LSL outlets ---------------------------------------------------
        ainfo = StreamInfo(audio_name, "Audio", 1, self.samplerate, "float32", source_id)
        chans = ainfo.desc().append_child("channels").append_child("channel")
        chans.append_child_value("label", "audio_mono")
        chans.append_child_value("unit", "normalized")
        ainfo.desc().append_child_value("device", str(dev.get("name", "")))
        self.audio_outlet = StreamOutlet(ainfo, chunk_size=self.blocksize, max_buffered=360)
        minfo = StreamInfo(marker_name, "Markers", 1, 0, "string", source_id + "_mrk")
        self.marker_outlet = StreamOutlet(minfo)
        self.audio_stream_name, self.marker_stream_name = audio_name, marker_name
        self.source_id = source_id

        # ---- playback state (guarded by _lock) -----------------------------
        self._lock = threading.Lock()
        self._buf = None
        self._pos = 0
        self._playing = False
        self._fade_out = False
        self._onset_evt = threading.Event()
        self.last_onset_lsl = None
        self.last_onset_dac = None
        self._clip_cache = {}
        self._offset = 0.0                                    # LSL - PortAudio stream time

        # ---- LSL push worker ----------------------------------------------
        self._q: queue.Queue = queue.Queue(maxsize=2048)
        self._stop_worker = False
        self._events = []
        self._backup_path = backup_path
        if backup_path:
            open(backup_path, "w").close()                    # truncate
        self._worker = threading.Thread(target=self._push_worker, daemon=True)
        self._worker.start()

        # ---- audio stream (kept open for the whole session) ----------------
        self.stream = sd.OutputStream(
            samplerate=self.samplerate, channels=self.out_channels, dtype="float32",
            blocksize=self.blocksize, device=device, latency=latency,
            callback=self._callback)
        self.stream.start()
        self.measure_offset()

    # -----------------------------------------------------------------------
    # clock mapping: PortAudio stream time  ->  LSL local_clock
    # -----------------------------------------------------------------------
    def measure_offset(self, n=15):
        offs = []
        for _ in range(n):
            t_lsl = self._local_clock()
            t_pa = self.stream.time
            offs.append(t_lsl - t_pa)
        self._offset = float(np.median(offs))
        return self._offset

    def dac_to_lsl(self, dac):
        return float(dac) + self._offset

    # -----------------------------------------------------------------------
    # real-time audio callback (must never raise or block)
    # -----------------------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        try:
            outdata.fill(0.0)
            with self._lock:
                playing = self._playing and self._buf is not None
                if playing:
                    start = self._pos
                    chunk = self._buf[start:start + frames]
                    n = len(chunk)
                    fade = self._fade_out
                    self._pos += n
                    ended = self._pos >= len(self._buf)
                else:
                    start = n = 0
                    chunk = None
                    fade = ended = False
            if playing and n > 0:
                block = chunk * self.amp
                if fade:                                       # ramp to zero, then stop
                    block = block * np.linspace(1.0, 0.0, n, dtype=np.float32)
                for c in range(self.out_channels):
                    outdata[:n, c] = block
                dac0 = float(time_info.outputBufferDacTime) or float(time_info.currentTime)
                ts_first = self.dac_to_lsl(dac0)
                if start == 0:                                 # first block of the clip
                    self.last_onset_dac = dac0
                    self.last_onset_lsl = ts_first
                    self._onset_evt.set()
                ts_last = ts_first + (n - 1) / self.samplerate
                try:
                    self._q.put_nowait((ts_last, np.array(block, dtype=np.float32)))
                except queue.Full:
                    pass
                if ended or fade:
                    with self._lock:
                        self._playing = False
                        self._fade_out = False
        except Exception:
            pass                                               # never propagate out of RT

    # -----------------------------------------------------------------------
    # LSL push worker (non-realtime): stamps each block at its true DAC time
    # -----------------------------------------------------------------------
    def _push_worker(self):
        while not self._stop_worker:
            try:
                ts_last, block = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self.audio_outlet.push_chunk(block.reshape(-1, 1).tolist(),
                                             timestamp=float(ts_last))
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # clip loading (resampled once to the device rate) + edge ramps
    # -----------------------------------------------------------------------
    def load(self, path):
        if path in self._clip_cache:
            return self._clip_cache[path]
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != self.samplerate:
            x = self._resample(x, sr, self.samplerate)
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._edge_ramp(x, ms=5)
        self._clip_cache[path] = x
        return x

    @staticmethod
    def _resample(x, sr_in, sr_out):
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)

    def _edge_ramp(self, x, ms=5):
        n = min(len(x) // 2, int(self.samplerate * ms / 1000))
        if n > 0:
            r = np.linspace(0.0, 1.0, n, dtype=np.float32)
            x[:n] *= r
            x[-n:] *= r[::-1]

    # -----------------------------------------------------------------------
    # playback
    # -----------------------------------------------------------------------
    def play(self, samples):
        """Start playing `samples` (mono float32 @ self.samplerate). Non-blocking.
        Use wait_onset() to get the hardware onset time in the LSL clock."""
        self._onset_evt.clear()
        self.last_onset_lsl = self.last_onset_dac = None
        with self._lock:
            self._buf = np.ascontiguousarray(samples, dtype=np.float32)
            self._pos = 0
            self._playing = True
            self._fade_out = False

    def wait_onset(self, timeout=1.0):
        """Block until the first sample has been handed to the DAC; return its LSL time."""
        return self.last_onset_lsl if self._onset_evt.wait(timeout) else None

    def stop(self, fade=True):
        with self._lock:
            if self._playing:
                if fade:
                    self._fade_out = True
                else:
                    self._playing = False

    def is_playing(self):
        with self._lock:
            return self._playing

    def make_beep(self, freq=1000.0, dur=0.1, amp=0.5):
        n = int(self.samplerate * dur)
        t = np.arange(n) / self.samplerate
        tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        self._edge_ramp(tone, ms=5)
        return tone

    # -----------------------------------------------------------------------
    # markers + provenance
    # -----------------------------------------------------------------------
    def marker(self, label, timestamp=None, extra=None):
        """Push a JSON event on the marker stream (and the backup log). Returns its ts."""
        ts = self._local_clock() if timestamp is None else float(timestamp)
        payload = {"label": label, "lsl_time": ts}
        if extra:
            payload.update(extra)
        s = json.dumps(payload, ensure_ascii=False)
        try:
            self.marker_outlet.push_sample([s], timestamp=ts)
        except Exception:
            pass
        self._events.append(payload)
        if self._backup_path:
            try:
                with open(self._backup_path, "a", encoding="utf-8") as f:
                    f.write(s + "\n")
            except Exception:
                pass
        return ts

    def now(self):
        return self._local_clock()

    def stream_info(self):
        return {"audio_stream": self.audio_stream_name,
                "marker_stream": self.marker_stream_name, "source_id": self.source_id,
                "samplerate": self.samplerate, "out_channels": self.out_channels,
                "blocksize": self.blocksize, "device": str(self.device),
                "clock_offset_lsl_minus_pa_s": self._offset,
                "output_latency_s": float(getattr(self.stream, "latency", 0.0) or 0.0)}

    def close(self):
        self.stop(fade=False)
        self._stop_worker = True
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass
