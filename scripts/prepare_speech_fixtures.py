#!/usr/bin/env python3
"""Rebuild the small attributed speech fixtures from the pinned public archive.

The full corpus stays in an explicitly supplied external cache directory. Only
selected PCM audio and its provenance are written to the fixture directory.
Archive members are read as streams; no paths from the archive are extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ARCHIVE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
ARCHIVE_SHA256 = "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
ARCHIVE_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
MEMBER_PATTERN = re.compile(r"LibriSpeech/test-clean/(\d+)/(\d+)/(\d+)-(\d+)-(\d+)\.flac")
RATE = 16_000
SAMPLES = RATE * 10


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(archive: Path, output: Path) -> dict:
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the pinned OpenSLR test-clean archive")
    fixtures = []
    with tarfile.open(archive, "r:gz") as corpus:
        members = {}
        for member in corpus.getmembers():
            match = MEMBER_PATTERN.fullmatch(member.name)
            if not match:
                continue
            speaker, chapter, utterance_speaker, utterance_chapter, utterance = (
                int(part) for part in match.groups()
            )
            if (speaker, chapter) != (utterance_speaker, utterance_chapter):
                raise ValueError("Mismatched source identity")
            if not member.isfile() or member.size > 64 * 1024 * 1024:
                raise ValueError("Invalid FLAC member")
            key = (speaker, chapter, utterance)
            if key in members:
                raise ValueError("Duplicate utterance identity")
            members[key] = member
        speakers = sorted({key[0] for key in members})[:10]
        if len(speakers) != 10:
            raise ValueError("The corpus must provide at least ten speakers")
        output.mkdir(parents=True, exist_ok=True)
        for speaker in speakers:
            pieces, segments = [], []
            remaining = SAMPLES
            for key in sorted(key for key in members if key[0] == speaker):
                member = members[key]
                stream = corpus.extractfile(member)
                if stream is None:
                    raise ValueError("Missing audio member")
                source_bytes = stream.read()
                pcm, rate = sf.read(io.BytesIO(source_bytes), dtype="int16")
                if rate != RATE or pcm.ndim != 1:
                    raise ValueError("Expected mono 16 kHz source audio")
                take = min(remaining, len(pcm))
                pieces.append(pcm[:take])
                segments.append({
                    "archive_member": member.name,
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "source_total_samples": len(pcm),
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": take,
                    "output_start_sample": SAMPLES - remaining,
                    "output_end_sample_exclusive": SAMPLES - remaining + take,
                })
                remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise ValueError(f"Speaker {speaker} has less than ten seconds")
            joined = np.concatenate(pieces)
            filename = f"speaker-{speaker:04d}.wav"
            destination = output / filename
            sf.write(destination, joined, RATE, subtype="PCM_16", format="WAV")
            fixtures.append({
                "speaker_id": speaker,
                "file": filename,
                "sha256": sha256(destination),
                "pcm_s16le_sha256": hashlib.sha256(joined.astype("<i2").tobytes()).hexdigest(),
                "samples": len(joined),
                "segments": segments,
            })
    manifest = {
        "schema_version": 1,
        "dataset": "LibriSpeech test-clean / OpenSLR SLR12",
        "creators": ["Vassil Panayotov", "Guoguo Chen", "Daniel Povey", "Sanjeev Khudanpur"],
        "source_page": "https://www.openslr.org/12/",
        "source_archive_url": ARCHIVE_URL,
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_archive_md5": ARCHIVE_MD5,
        "source_checksum_url": "https://www.openslr.org/resources/12/md5sum.txt",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "sample_rate": RATE,
        "channels": 1,
        "encoding": "WAV PCM signed 16-bit little-endian",
        "selection": "First ten speakers by numeric ID; numeric (speaker, chapter, utterance) "
        "ordering; concatenate each speaker's utterances and keep the first 160000 samples.",
        "modifications": ["FLAC decoded losslessly to PCM16", "Selected utterances concatenated",
                          "Cropped to ten seconds", "Encoded as WAV without source metadata"],
        "test_transform": "Scale each decoded fixture by target_peak / max(abs(audio)); "
        "target_peak in [0.10, 0.30, 0.55]; prepend one second of zero-valued PCM; "
        "pad the final partial frame with zeros.",
        "transcripts_included": False,
        "fixtures": fixtures,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing test-clean.tar.gz")
    parser.add_argument("--cache-dir", type=Path, help="External directory for corpus download")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/speech"))
    args = parser.parse_args()
    if args.archive is None:
        if args.cache_dir is None:
            parser.error("Supply --archive or an external --cache-dir")
        repo = Path(__file__).resolve().parents[1]
        cache = args.cache_dir.resolve()
        if cache == repo or repo in cache.parents:
            parser.error("Keep the full corpus outside this repository")
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "test-clean.tar.gz"
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            try:
                with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as source:
                    with temporary.open("wb") as target:
                        while block := source.read(1024 * 1024):
                            target.write(block)
                if sha256(temporary) != ARCHIVE_SHA256:
                    raise ValueError("Downloaded archive checksum mismatch")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
    else:
        archive = args.archive
    manifest = prepare(archive, args.output)
    print(json.dumps({"fixtures": len(manifest["fixtures"]), "total_seconds": 100}))


if __name__ == "__main__":
    main()
