"""Read the BrainVision recording and the photodiode edge markers.

The 63 data channels are recorded as bare physical numbers (electrode 2 is the
online reference and is absent). The photodiode square's luminance transitions
are logged by the amplifier as ``S 15`` / ``S 14`` marker pairs; we keep the
``S 15`` sample of each pair as the edge time.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime

import mne
import numpy as np

from .utils import LOG


@dataclass
class Markers:
    """Photodiode edges parsed from the .vmrk marker file."""
    edge_samples: np.ndarray      # sample index (@ raw sfreq) of every S 15 edge
    rec_start_unix: float         # absolute POSIX time of sample 0 (New Segment)
    sfreq: float


def _resilient_vhdr(vhdr_path: str) -> str:
    """Return a .vhdr whose DataFile/MarkerFile actually resolve.

    The recording was anonymised by renaming the .eeg/.vmrk files, but the header
    still points at the original names. If those are missing we build a patched
    header in a temp dir (symlinking the real .eeg/.vmrk in place of copying the
    ~0.5 GB data), leaving the user's data directory untouched.
    """
    d = os.path.dirname(os.path.abspath(vhdr_path))
    text = open(vhdr_path, encoding="utf-8").read()
    dataf = re.search(r"DataFile=(.+)", text).group(1).strip()
    markf = re.search(r"MarkerFile=(.+)", text).group(1).strip()
    if os.path.exists(os.path.join(d, dataf)) and os.path.exists(os.path.join(d, markf)):
        return vhdr_path                                 # header already valid

    real_eeg = next((f for f in os.listdir(d) if f.endswith(".eeg")), None)
    real_vmrk = next((f for f in os.listdir(d) if f.endswith(".vmrk")), None)
    if not (real_eeg and real_vmrk):
        return vhdr_path                                 # let MNE raise clearly
    tmp = tempfile.mkdtemp(prefix="cma_bv_")
    for real, want in [(real_eeg, dataf), (real_vmrk, markf)]:
        link = os.path.join(tmp, want)
        try:
            os.symlink(os.path.join(d, real), link)
        except (OSError, NotImplementedError):
            import shutil; shutil.copy2(os.path.join(d, real), link)
    patched = os.path.join(tmp, os.path.basename(vhdr_path))
    open(patched, "w", encoding="utf-8").write(text)
    LOG.info("Header referenced missing %s/%s; patched via temp symlinks.",
             dataf, markf)
    return patched


def load_raw(vhdr_path: str) -> mne.io.BaseRaw:
    """Load the continuous BrainVision file and mark every channel as EEG."""
    raw = mne.io.read_raw_brainvision(_resilient_vhdr(vhdr_path),
                                      preload=True, verbose="ERROR")
    # All 63 channels are scalp EEG; there is no dedicated EOG/ECG channel.
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    LOG.info("Loaded %s: %d ch, %.1f s @ %.0f Hz",
             vhdr_path.split("/")[-1], len(raw.ch_names),
             raw.times[-1], raw.info["sfreq"])
    return raw


def _parse_new_segment_unix(vmrk_text: str) -> float | None:
    """Absolute POSIX time of the recording start (New Segment timestamp)."""
    m = re.search(r"New Segment,,\d+,\d+,\d+,(\d{20})", vmrk_text)
    if not m:
        return None
    ts = m.group(1)                       # YYYYMMDDHHMMSS + 6 microsecond digits
    dt = datetime.strptime(ts[:14] + ts[14:20], "%Y%m%d%H%M%S%f")
    return dt.timestamp()                 # interpreted in local tz (as recorded)


def load_markers(vmrk_path: str, sfreq: float,
                 edge_code: str = "S 15") -> Markers:
    """Parse photodiode edge samples and the recording-start wall clock.

    ``edge_code`` is matched against the marker *description* (e.g. ``"S 15"``).
    """
    text = open(vmrk_path, "r", encoding="utf-8", errors="replace").read()
    want = edge_code.split("/")[-1].strip()          # tolerate "Stimulus/S 15"
    samples = []
    for line in text.splitlines():
        m = re.match(r"Mk\d+=Stimulus,([^,]*),(\d+),", line)
        if m and m.group(1).strip() == want:
            samples.append(int(m.group(2)))
    edges = np.asarray(sorted(samples), dtype=int)
    rec_start = _parse_new_segment_unix(text)
    if rec_start is None:
        raise RuntimeError(f"No 'New Segment' timestamp in {vmrk_path}")
    LOG.info("Parsed %d photodiode edges ('%s'); recording start %s",
             len(edges), want,
             datetime.fromtimestamp(rec_start).isoformat(timespec="milliseconds"))
    return Markers(edge_samples=edges, rec_start_unix=rec_start, sfreq=sfreq)
