#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stimuli.py — prepare the auditory stimulus library ONCE, then reuse it.

This script solves the "regenerate every run" problem: instead of synthesising
audio + word boundaries at the start of every session, you run this builder a
single time. It downloads a real, already-labelled speech dataset (LibriSpeech
with Montreal-Forced-Aligner WORD onsets), assembles ~30 s clips, and writes:

    stims/audio/<id>.wav            the audio stimulus (16 kHz mono)
    stims/audio/<id>.words.json     word onsets (start/end, ms + s)
    stims/audio/<id>.words.csv      same, as CSV
    stims/manifest.json             index of every prepared clip + its probes

Usage
-----
    python build_stimuli.py                 # build N clips (config: build.n_clips)
    python build_stimuli.py --add 5         # APPEND 5 more clips to the library
    python build_stimuli.py --source tts    # use local gTTS + Whisper instead
    python build_stimuli.py --list          # show what's already prepared

All knobs live in config.yaml (the `build:` block).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random

import numpy as np

import cma_common as cc

# A small bank of common, concrete content words used as FALSE probe targets
# (a word that is plausibly speakable but, when chosen, is absent from a given
# transcript). Kept generic so it rarely collides with LibriSpeech vocabulary.
COMMON_WORDS = [
    "anchor", "purple", "engine", "harbor", "candle", "ladder", "silver",
    "meadow", "rocket", "pencil", "garden", "bottle", "thunder", "marble",
    "velvet", "copper", "lantern", "pillow", "saddle", "feather", "orchard",
    "cabinet", "blanket", "compass", "diamond", "kettle", "mirror", "ribbon",
    "tunnel", "wagon", "whistle", "balloon", "biscuit", "cottage", "dolphin",
    "glacier", "hammer", "jacket", "kingdom", "magnet",
]

# Function words excluded when picking TRUE probe targets from a transcript.
STOPWORDS = {
    "the", "and", "that", "with", "this", "from", "they", "have", "were",
    "their", "what", "when", "your", "which", "them", "then", "than", "into",
    "been", "more", "some", "such", "only", "would", "could", "should", "about",
    "there", "these", "those", "here", "very", "just", "also", "upon", "shall",
    "will", "they", "him", "her", "his", "its", "our", "are", "was", "for",
    "not", "but", "you", "had", "has",
}

# Sentence bank used only by the TTS fallback to compose varied paragraphs.
SENTENCE_BANK = [
    "Attention is the quiet gatekeeper of perception.",
    "The brain keeps only a fraction of what the senses deliver.",
    "A single voice can be followed through a crowded and noisy room.",
    "The other sounds do not vanish, they fade gently into the background.",
    "A moving shape can capture the eye while its surroundings blur.",
    "Memory favours the things to which we choose to attend.",
    "The harbour was calm under a wide and silver morning sky.",
    "Travellers spoke of distant mountains capped with early snow.",
    "The old clock in the hallway counted the hours without complaint.",
    "Rain tapped the window like a patient and familiar visitor.",
    "She traced the river on the map with a careful finger.",
    "The lantern threw long shadows across the wooden floor.",
    "Far away a train whistled and then the night was still.",
    "He gathered the letters and tied them with a faded ribbon.",
    "The garden smelled of rain, of earth, and of green growing things.",
    "Every choice of focus is also a choice of what to ignore.",
]


# ===========================================================================
# Probe construction (yes/no "was WORD spoken?")
# ===========================================================================
def _content_words(transcript: str, min_len: int) -> list[str]:
    """Lower-cased content words from a transcript suitable as probe targets."""
    toks = [t.strip(".,!?;:\"'()").lower() for t in transcript.split()]
    return [
        t for t in toks
        if t.isalpha() and len(t) >= min_len
        and t not in STOPWORDS and t != "unk"
    ]


def make_audio_probes(transcript: str, min_len: int, rng: random.Random) -> list[dict]:
    """
    Build the two yes/no probe variants for a clip:
      * a TRUE-target probe  — a content word that DOES occur in the transcript.
      * a FALSE-target probe — a common word that does NOT occur in it.
    The runner picks one per trial so the correct answer stays balanced.
    """
    present_pool = _content_words(transcript, min_len)
    transcript_set = set(_content_words(transcript, 1))  # any-length, for membership

    probes: list[dict] = []
    if present_pool:
        tw = rng.choice(sorted(set(present_pool)))
        probes.append({"target_word": tw, "present": True,
                       "question": f'Was the word "{tw}" spoken?'})

    absent_pool = [w for w in COMMON_WORDS
                   if w not in transcript_set and len(w) >= min_len]
    if absent_pool:
        fw = rng.choice(absent_pool)
        probes.append({"target_word": fw, "present": False,
                       "question": f'Was the word "{fw}" spoken?'})
    return probes


# ===========================================================================
# Writing a prepared clip to disk
# ===========================================================================
def _write_word_files(words: list[dict], json_path: str, csv_path: str,
                      transcript: str, source: str) -> None:
    """Persist merged word onsets to JSON and CSV."""
    payload = {
        "source": source,
        "transcript": transcript,
        "n_words": len(words),
        "note": "Times are seconds/ms relative to audio start == AV onset.",
        "words": words,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word", "start_s", "end_s",
                                          "start_ms", "end_ms", "duration_ms",
                                          "is_unk"])
        w.writeheader()
        w.writerows(words)


def _save_clip(clip_id: str, audio: np.ndarray, sr: int, words: list[dict],
               transcript: str, source: str, paths: cc.Paths,
               extra: dict, min_word_len: int, rng: random.Random) -> dict:
    """
    Normalise + write a clip's wav and word files, and return its manifest entry.
    Audio paths in the entry are stored RELATIVE to the stim directory.
    """
    import soundfile as sf

    # Peak-normalise for a consistent presentation level across clips.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = (audio / peak) * 0.95

    wav_rel = os.path.join("audio", f"{clip_id}.wav")
    json_rel = os.path.join("audio", f"{clip_id}.words.json")
    csv_rel = os.path.join("audio", f"{clip_id}.words.csv")
    wav_abs = os.path.join(paths.stim_dir, wav_rel)
    sf.write(wav_abs, audio.astype("float32"), int(sr), subtype="PCM_16")
    _write_word_files(words,
                      os.path.join(paths.stim_dir, json_rel),
                      os.path.join(paths.stim_dir, csv_rel),
                      transcript, source)

    entry = {
        "id": clip_id,
        "source": source,
        "audio_path": wav_rel,
        "words_json": json_rel,
        "words_csv": csv_rel,
        "transcript": transcript,
        "duration_s": round(len(audio) / float(sr), 3),
        "sample_rate": int(sr),
        "n_words": len(words),
        "probes": make_audio_probes(transcript, min_word_len, rng),
    }
    entry.update(extra)
    return entry


# ===========================================================================
# Source 1 — LibriSpeech with MFA word onsets (preferred)
# ===========================================================================
def _merge_utterances(utts: list[dict], gap_s: float, sr: int,
                      max_duration_s: float) -> tuple[np.ndarray, list[dict], str]:
    """
    Concatenate several utterances (each: {array, words, transcript}) into one
    continuous clip, offsetting word onsets by the cumulative time and inserting
    `gap_s` of silence between utterances. Trim to `max_duration_s` if needed.
    """
    gap = np.zeros(int(round(gap_s * sr)), dtype="float32")
    chunks: list[np.ndarray] = []
    words: list[dict] = []
    transcripts: list[str] = []
    offset = 0.0  # seconds already accumulated

    for i, u in enumerate(utts):
        arr = u["array"].astype("float32")
        for w in u["words"]:
            start = float(w["start"]) + offset
            end = float(w["end"]) + offset
            tok = (w["word"] or "").strip()
            words.append({
                "word": tok,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "start_ms": int(round(start * 1000)),
                "end_ms": int(round(end * 1000)),
                "duration_ms": int(round((end - start) * 1000)),
                "is_unk": tok == "<unk>",
            })
        transcripts.append(u["transcript"].strip())
        chunks.append(arr)
        offset += len(arr) / sr
        if i < len(utts) - 1:
            chunks.append(gap)
            offset += len(gap) / sr

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")

    # Enforce the per-clip maximum duration.
    max_samples = int(round(max_duration_s * sr))
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        cutoff = max_duration_s
        words = [w for w in words if w["end_s"] <= cutoff]

    return audio, words, " ".join(transcripts)


def build_librispeech(cfg: dict, paths: cc.Paths, n_needed: int,
                      start_index: int, rng: random.Random) -> list[dict]:
    """Stream the dataset and assemble `n_needed` ~target-length clips."""
    from datasets import load_dataset, Audio

    b = cfg["build"]
    ls = b["librispeech"]
    target = float(b["target_duration_s"])
    max_dur = float(b["max_duration_s"])
    min_word_len = int(cfg["probe"]["audio"]["min_word_len"])
    same_speaker = bool(ls.get("same_speaker_per_clip", True))
    gap_s = float(ls.get("inter_utterance_gap_s", 0.3))
    scan_limit = int(ls.get("scan_limit", 4000))
    split = ls["split"]
    source_tag = f"librispeech:{split}"

    cc.log(f"Streaming '{ls['dataset']}' split='{split}' "
           f"(target ~{target:.0f}s/clip, need {n_needed})...")
    ds = load_dataset(ls["dataset"], split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes -> we decode

    import soundfile as sf

    # Buffer utterances per speaker until one reaches the target duration.
    buffers: dict[str, dict] = {}     # speaker -> {"utts": [...], "dur": float, "sex": ..}
    used_speakers: set[str] = set()
    entries: list[dict] = []
    idx = start_index
    scanned = 0

    for ex in ds:
        scanned += 1
        if scanned > scan_limit:
            cc.log(f"Reached scan_limit={scan_limit}; stopping search.")
            break

        speaker = ex["id"].split("-")[0]
        group = speaker if same_speaker else "_all"
        if same_speaker and speaker in used_speakers:
            continue  # keep speakers distinct across clips

        data, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        buf = buffers.setdefault(group, {"utts": [], "dur": 0.0, "sex": ex["sex"]})
        buf["utts"].append({"array": data, "words": ex["words"],
                            "transcript": ex["transcript"], "id": ex["id"]})
        buf["dur"] += len(data) / sr

        if buf["dur"] >= target:
            audio, words, transcript = _merge_utterances(
                buf["utts"], gap_s, sr, max_dur)
            idx += 1
            clip_id = f"ls_{split}_{idx:02d}"
            extra = {
                "speaker": speaker if same_speaker else "mixed",
                "speaker_sex": buf["sex"] if same_speaker else "mixed",
                "component_utterances": [u["id"] for u in buf["utts"]],
            }
            entry = _save_clip(clip_id, audio, sr, words, transcript,
                               source_tag, paths, extra, min_word_len, rng)
            entries.append(entry)
            cc.log(f"  + {clip_id}: {entry['duration_s']:.1f}s, "
                   f"{entry['n_words']} words, speaker {speaker} "
                   f"({len(buf['utts'])} utts) -> probes "
                   f"{[p['target_word'] for p in entry['probes']]}")
            used_speakers.add(speaker)
            del buffers[group]
            if len(entries) >= n_needed:
                break

    if len(entries) < n_needed:
        cc.log(f"WARNING: only assembled {len(entries)}/{n_needed} clips "
               f"(scanned {scanned} utterances). Try a larger scan_limit or a "
               f"different split.")
    return entries


# ===========================================================================
# Source 2 — local gTTS + faster-whisper (offline-ish fallback)
# ===========================================================================
def _compose_text(rng: random.Random, target_words: int) -> str:
    """Sample sentences into a paragraph of roughly `target_words` words."""
    pool = SENTENCE_BANK[:]
    rng.shuffle(pool)
    out, count = [], 0
    for s in pool:
        out.append(s)
        count += len(s.split())
        if count >= target_words:
            break
    return " ".join(out)


def build_tts(cfg: dict, paths: cc.Paths, n_needed: int,
              start_index: int, rng: random.Random) -> list[dict]:
    """Synthesize clips locally (gTTS) and align words with faster-whisper."""
    from gtts import gTTS
    from faster_whisper import WhisperModel
    import soundfile as sf
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            module=r"faster_whisper\.feature_extractor")

    b = cfg["build"]
    tcfg = b["tts"]
    min_word_len = int(cfg["probe"]["audio"]["min_word_len"])
    target = float(b["target_duration_s"])
    texts = list(tcfg.get("texts") or [])
    # ~2.6 words/second of speech is a reasonable rate estimate.
    target_words = int(target * 2.6)

    cc.log(f"Loading faster-whisper '{tcfg['whisper_model']}' for alignment...")
    model = WhisperModel(tcfg["whisper_model"], device="cpu", compute_type="int8")

    entries: list[dict] = []
    idx = start_index
    for k in range(n_needed):
        text = texts[k] if k < len(texts) else _compose_text(rng, target_words)
        cc.log(f"Synthesizing clip {k + 1}/{n_needed} ({len(text.split())} words)...")
        tmp_mp3 = os.path.join(paths.audio_dir, f"_tmp_{k}.mp3")
        gTTS(text=text, lang="en", slow=False).save(tmp_mp3)
        data, sr = sf.read(tmp_mp3, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        os.remove(tmp_mp3)

        # Align with word timestamps (faster-whisper reads from a temp wav).
        tmp_wav = os.path.join(paths.audio_dir, f"_tmp_align_{k}.wav")
        sf.write(tmp_wav, data.astype("float32"), int(sr), subtype="PCM_16")
        seg_iter, _ = model.transcribe(tmp_wav, language="en",
                                       word_timestamps=True, beam_size=5)
        words = []
        for seg in seg_iter:
            for w in (seg.words or []):
                tok = w.word.strip()
                if not tok:
                    continue
                words.append({
                    "word": tok,
                    "start_s": round(float(w.start), 3),
                    "end_s": round(float(w.end), 3),
                    "start_ms": int(round(float(w.start) * 1000)),
                    "end_ms": int(round(float(w.end) * 1000)),
                    "duration_ms": int(round((float(w.end) - float(w.start)) * 1000)),
                    "is_unk": False,
                })
        os.remove(tmp_wav)

        idx += 1
        clip_id = f"tts_{idx:02d}"
        entry = _save_clip(clip_id, data, sr, words, text, f"tts:{tcfg['engine']}",
                           paths, {"speaker": "tts", "speaker_sex": "na"},
                           min_word_len, rng)
        entries.append(entry)
        cc.log(f"  + {clip_id}: {entry['duration_s']:.1f}s, {entry['n_words']} words")
    return entries


# ===========================================================================
# Orchestration
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the auditory stimulus library.")
    ap.add_argument("--config", default=cc.DEFAULT_CONFIG)
    ap.add_argument("--add", type=int, default=0,
                    help="Append this many NEW clips to the existing library.")
    ap.add_argument("--source", choices=["librispeech", "tts"], default=None,
                    help="Override build.source from the config.")
    ap.add_argument("--n", type=int, default=None,
                    help="Override build.n_clips (ignored when --add is used).")
    ap.add_argument("--list", action="store_true",
                    help="List the already-prepared clips and exit.")
    args = ap.parse_args()

    cfg = cc.load_config(args.config)
    paths = cc.Paths.from_config(cfg).ensure()
    manifest = cc.read_manifest(paths.manifest)

    if args.list:
        items = manifest.get("audio", [])
        cc.log(f"{len(items)} prepared audio clip(s) in {paths.manifest}:")
        for e in items:
            cc.log(f"  {e['id']:18s} {e['duration_s']:5.1f}s  "
                   f"{e['n_words']:3d} words  [{e['source']}]")
        return

    if args.source:
        cfg["build"]["source"] = args.source
    source = cfg["build"]["source"]

    existing = manifest.get("audio", [])
    existing_ids = {e["id"] for e in existing}
    # Continue clip numbering from the current maximum so --add never collides.
    start_index = 0
    for e in existing:
        tail = e["id"].rsplit("_", 1)[-1]
        if tail.isdigit():
            start_index = max(start_index, int(tail))

    n_needed = args.add if args.add > 0 else (args.n or int(cfg["build"]["n_clips"]))
    rng = random.Random(int(cfg["build"]["seed"]) + start_index)

    print("=" * 72)
    print(f"BUILDING STIMULUS LIBRARY — source='{source}', "
          f"{'adding' if args.add else 'target'} {n_needed} clip(s)")
    print("=" * 72)

    if source == "librispeech":
        new_entries = build_librispeech(cfg, paths, n_needed, start_index, rng)
    elif source == "tts":
        new_entries = build_tts(cfg, paths, n_needed, start_index, rng)
    else:
        raise ValueError(f"Unknown build.source: {source!r}")

    # Merge (dedupe by id) and persist.
    merged = existing + [e for e in new_entries if e["id"] not in existing_ids]
    manifest["audio"] = merged
    cc.write_manifest(paths.manifest, manifest)

    print("=" * 72)
    cc.log(f"Done. Library now has {len(merged)} clip(s). Manifest: {paths.manifest}")
    print("=" * 72)


if __name__ == "__main__":
    main()
