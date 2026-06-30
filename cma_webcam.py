#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cma_webcam.py — record the participant webcam during a block.

It runs as a SEPARATE PROCESS from the PsychoPy experiment, on purpose:
  * camera capture + encoding never contends with the timing-critical draw loop
    (no dropped flips), and
  * OpenCV's bundled media libraries can't collide with PsychoPy's movie backend
    (they live in different processes).

OpenCV (`cv2`) is imported ONLY inside the subprocess entry point, so importing
this module in the experiment process does not load cv2 there.

Outputs (next to the chosen --out path):
    <out>               the video         (e.g. webcam/sub01_..._b1.mp4)
    <out>.frames.csv    per-frame WALL-CLOCK timestamps  ->  aligns to AV onset
    <out>.status.json   {opened, width, height, fps, frames, duration_s, error}

Standalone use:
    python cma_webcam.py --out webcam/test.mp4 --device 0 --fps 0 \
                         --width 640 --height 480 --max-seconds 300
The recorder stops on SIGINT/SIGTERM (the parent's WebcamController.stop()) or
when --max-seconds elapses, finalising the video cleanly either way.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

# Set by the signal handlers; checked by the capture loop.
_STOP = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


# ===========================================================================
# Subprocess entry point (imports cv2; runs in its own process)
# ===========================================================================
def record_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Webcam recorder subprocess.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0.0, help="0 = camera's rate")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fourcc", default="mp4v")
    ap.add_argument("--max-seconds", type=float, default=600.0)
    args = ap.parse_args(argv)

    # Stop cleanly when the parent signals us.
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    status_path = args.out + ".status.json"
    frames_path = args.out + ".frames.csv"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def write_status(d: dict) -> None:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    try:
        import cv2
    except Exception as e:  # pragma: no cover
        write_status({"opened": False, "error": f"cv2 import failed: {e}"})
        return 2

    cap = cv2.VideoCapture(args.device)             # AVFoundation on macOS
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        write_status({"opened": False,
                      "error": f"cannot open camera device {args.device}"})
        cap.release()
        return 3

    ok, frame = cap.read()
    if not ok or frame is None:
        write_status({"opened": False,
                      "error": "camera opened but returned no frame"})
        cap.release()
        return 4

    h, w = frame.shape[:2]
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps > 0 else (cam_fps if cam_fps and cam_fps > 1 else 30.0)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*args.fourcc),
                             fps, (w, h))
    if not writer.isOpened():
        write_status({"opened": False, "error": "VideoWriter failed to open "
                      f"(fourcc={args.fourcc})"})
        cap.release()
        return 5

    # Signal "camera is up" so the parent can proceed (and detect failures).
    write_status({"opened": True, "width": w, "height": h, "fps": fps,
                  "device": args.device, "fourcc": args.fourcc})

    # Capture loop. We timestamp every frame with the wall clock (time.time),
    # which is comparable across processes, so the experiment process can later
    # map AV onset (also a time.time stamp) to the nearest webcam frame.
    times: list[float] = []
    writer.write(frame)
    times.append(time.time())
    deadline = times[0] + args.max_seconds
    while not _STOP and time.time() < deadline:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        times.append(time.time())

    writer.release()
    cap.release()

    with open(frames_path, "w", encoding="utf-8") as f:
        f.write("frame_index,t_wallclock,t_rel_s\n")
        t0 = times[0] if times else 0.0
        for i, t in enumerate(times):
            f.write(f"{i},{t:.6f},{t - t0:.6f}\n")

    write_status({"opened": True, "width": w, "height": h, "fps": fps,
                  "device": args.device, "fourcc": args.fourcc,
                  "frames": len(times),
                  "duration_s": round(times[-1] - times[0], 3) if len(times) > 1 else 0.0,
                  "first_frame_wallclock": times[0] if times else None})
    return 0


# ===========================================================================
# Parent-side controller (no cv2 import; safe to use in the experiment process)
# ===========================================================================
class WebcamController:
    """Start/stop a webcam recorder subprocess and read back its status."""

    def __init__(self, out_path: str, device: int = 0, fps: float = 0.0,
                 resolution=None, fourcc: str = "mp4v", max_seconds: float = 600.0):
        self.out_path = out_path
        self.device = device
        self.fps = fps
        self.resolution = resolution          # (w, h) or None
        self.fourcc = fourcc
        self.max_seconds = max_seconds
        self.proc = None
        self.status_path = out_path + ".status.json"
        self.frames_path = out_path + ".frames.csv"
        self.start_wallclock = None

    def start(self, open_timeout: float = 5.0) -> bool:
        """Spawn the recorder; block until the camera opens or fails. -> opened?"""
        import subprocess
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        for p in (self.status_path,):
            try:
                os.remove(p)
            except OSError:
                pass

        w, h = (self.resolution or (0, 0))
        cmd = [sys.executable, os.path.abspath(__file__),
               "--out", self.out_path, "--device", str(self.device),
               "--fps", str(self.fps), "--width", str(w), "--height", str(h),
               "--fourcc", self.fourcc, "--max-seconds", str(self.max_seconds)]
        self.proc = subprocess.Popen(cmd)
        self.start_wallclock = time.time()

        t0 = time.time()
        while time.time() - t0 < open_timeout:
            st = self.read_status()
            if st is not None:
                return bool(st.get("opened"))
            if self.proc.poll() is not None:      # exited before writing status
                return False
            time.sleep(0.05)
        return False

    def read_status(self):
        if os.path.exists(self.status_path):
            try:
                with open(self.status_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def stop(self, timeout: float = 5.0):
        """Signal the recorder to finish, wait for a clean finalise, read status."""
        if self.proc is None:
            return self.read_status()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout)
            except Exception:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except Exception:
                    self.proc.kill()
        return self.read_status()


if __name__ == "__main__":
    sys.exit(record_main())
