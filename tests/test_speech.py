"""Acceptance on the explicitly selected, attributed normal-speech inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tactical_speech_enhancement.config import Settings
from tactical_speech_enhancement.guard import InputGuard

FIXTURES = Path(__file__).parent / "fixtures" / "speech"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
SPEAKERS = [61, 121, 237, 260, 672, 908, 1089, 1188, 1221, 1284]


def test_selected_corpus_provenance_and_audio_integrity():
    assert MANIFEST["license"] == "CC-BY-4.0"
    assert [item["speaker_id"] for item in MANIFEST["fixtures"]] == SPEAKERS
    assert MANIFEST["transcripts_included"] is False
    for item in MANIFEST["fixtures"]:
        path = FIXTURES / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        audio, rate = sf.read(path, dtype="int16")
        assert rate == 16000 and audio.shape == (160000,)
        assert hashlib.sha256(audio.astype("<i2").tobytes()).hexdigest() == (
            item["pcm_s16le_sha256"]
        )
        cursor = 0
        previous_id = None
        for segment in item["segments"]:
            assert segment["output_start_sample"] == cursor
            assert segment["source_start_sample"] == 0
            cursor += segment["source_end_sample_exclusive"]
            assert segment["output_end_sample_exclusive"] == cursor
            assert segment["source_end_sample_exclusive"] <= segment["source_total_samples"]
            numeric_id = tuple(int(part) for part in Path(segment["archive_member"]).stem.split("-"))
            assert numeric_id[0] == item["speaker_id"]
            assert previous_id is None or previous_id < numeric_id
            previous_id = numeric_id
        assert cursor == 160000


@pytest.mark.parametrize("target_peak", [0.10, 0.30, 0.55])
@pytest.mark.parametrize("item", MANIFEST["fixtures"], ids=lambda item: f"speaker-{item['speaker_id']}")
def test_normal_speech_has_no_guard_reentry_or_additional_blocked_samples(item, target_peak):
    audio, rate = sf.read(FIXTURES / item["file"], dtype="float64")
    audio *= target_peak / np.max(np.abs(audio))
    # Exactly one second of silence, including its half-frame at the boundary.
    signal = np.concatenate((np.zeros(rate), audio))
    frame_size = Settings().frame_samples
    signal = np.pad(signal, (0, -len(signal) % frame_size))
    guard = InputGuard(Settings())
    blocked_samples = 0
    for start in range(0, len(signal), frame_size):
        decision = guard.inspect(signal[start:start + frame_size])
        assert decision.valid and not decision.unsafe
        first = max(rate - start, 0)
        last = min(rate + len(audio) - start, frame_size)
        if first < last:
            blocked_samples += np.count_nonzero(decision.gains[first:last] < 1)
    assert guard.summary()["reentries"] == 0
    assert guard.summary()["anomalies"] == 0
    assert blocked_samples == 0
