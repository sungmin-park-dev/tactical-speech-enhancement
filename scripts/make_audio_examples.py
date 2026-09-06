"""Generate reproducible listening examples with the same offline path as `tse process`."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from tactical_speech_enhancement.artifacts import default_model_path
from tactical_speech_enhancement.cli import process_file
from tactical_speech_enhancement.config import Settings
from tactical_speech_enhancement.guard import InputGuard

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260906


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pcm(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = np.clip(np.rint(samples * 32768), -32768, 32767).astype(np.int16)
    sf.write(path, pcm, rate, subtype="PCM_16")


def check_output(source: Path, output: Path, settings: Settings) -> dict:
    samples, rate = sf.read(source, dtype="float32")
    result, result_rate = sf.read(output, dtype="int16")
    assert rate == result_rate == settings.sample_rate
    assert result.shape == samples.shape and np.isfinite(result).all()
    peak = int(np.max(np.abs(result.astype(np.int32))))
    assert peak <= math.floor(32768 * 10 ** (settings.ceiling_dbfs / 20))
    guard = InputGuard(settings)
    muted_samples = 0
    for start in range(0, len(samples), settings.frame_samples):
        piece = samples[start : start + settings.frame_samples]
        frame = np.zeros(settings.frame_samples, dtype=np.float32)
        frame[: len(piece)] = piece
        decision = guard.inspect(frame)
        muted = decision.gains[: len(piece)] == 0
        assert np.all(result[start : start + len(piece)][muted] == 0)
        muted_samples += int(np.count_nonzero(muted))
    assert guard.gain == 1 and np.any(result[rate:])
    return {
        "length_and_rate_match": True,
        "output_peak_pcm16": peak,
        "output_peak_dbfs": 20 * math.log10(peak / 32768),
        "digital_ceiling_passed": True,
        "guard_zero_gain_samples_checked": muted_samples,
        "guard_zero_gain_leaked_samples": 0,
        "guard_final_gain": guard.gain,
        "guard_anomalous_frames": guard.anomalies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples/audio")
    parser.add_argument("--mp3", action="store_true", help="also encode listening copies with FFmpeg")
    args = parser.parse_args()
    if args.mp3 and shutil.which("ffmpeg") is None:
        parser.error("--mp3 requires FFmpeg with the libmp3lame encoder")
    settings = Settings()
    rate = settings.sample_rate
    source = ROOT / "tests/fixtures/speech/speaker-0061.wav"
    provenance = json.loads((source.parent / "manifest.json").read_text())
    expected = next(item for item in provenance["fixtures"] if item["file"] == source.name)
    if sha256(source) != expected["sha256"]:
        raise ValueError("Source fixture checksum mismatch")
    speech, source_rate = sf.read(source, dtype="float64")
    if source_rate != rate or speech.shape != (10 * rate,):
        raise ValueError("Expected the pinned ten-second mono speech fixture")
    speech *= 0.30 / np.max(np.abs(speech))
    rng = np.random.default_rng(SEED)
    timeline = np.arange(speech.size) / rate
    noise = rng.standard_normal(speech.size) + 0.5 * np.sin(2 * np.pi * 120 * timeline)
    noise *= np.sqrt(np.mean(speech**2)) / (10 ** (5 / 20) * np.sqrt(np.mean(noise**2)))
    # One second of digital silence completes startup protection before speech.
    clean = np.concatenate((np.zeros(rate), speech))
    noisy = np.concatenate((np.zeros(rate), speech + noise))
    assert np.max(np.abs(noisy)) < settings.recovery_threshold
    impulse = noisy.copy()
    events = [(4.0, 0.005, 0.95), (4.192, 0.005, 0.95), (7.0, 0.050, 0.999)]
    for start_s, duration_s, amplitude in events:
        start = round(start_s * rate)
        count = round(duration_s * rate)
        # Broadband synthetic bursts, including a repeat during gain recovery.
        pulse = np.clip(rng.standard_normal(count) * 2, -1, 1) * amplitude
        pulse[:2] = (amplitude, -amplitude)
        impulse[start : start + count] = pulse
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    for name, samples in (("clean-reference", clean), ("noise-input", noisy),
                          ("impulse-input", impulse)):
        write_pcm(destination / f"{name}.wav", samples, rate)
    cases = {}
    for stem in ("noise", "impulse"):
        input_path = destination / f"{stem}-input.wav"
        output_path = destination / f"{stem}-output.wav"
        report = process_file(input_path, output_path, args.model_path, settings)
        checks = check_output(input_path, output_path, settings)
        if stem == "noise":
            assert checks["guard_anomalous_frames"] == 0
        else:
            assert checks["guard_anomalous_frames"] > 0
        cases[stem] = {"processing": report, "checks": checks}
    names = ("clean-reference", "noise-input", "noise-output", "impulse-input", "impulse-output")
    files = []
    for name in names:
        wav = destination / f"{name}.wav"
        data, _ = sf.read(wav, dtype="int16")
        files.append({"file": wav.name, "sha256": sha256(wav), "samples": len(data)})
        if args.mp3:
            mp3 = wav.with_suffix(".mp3")
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(wav), "-map_metadata", "-1", "-c:a", "libmp3lame",
                "-b:a", "64k", "-id3v2_version", "0", "-write_id3v1", "0", str(mp3),
            ], check=True)
            files.append({"file": mp3.name, "sha256": sha256(mp3)})
    manifest = {
        "schema_version": 1,
        "source": {"file": source.relative_to(ROOT).as_posix(), "sha256": expected["sha256"],
                   "source_page": provenance["source_page"], "creators": provenance["creators"]},
        "license": "CC-BY-4.0",
        "sample_rate": rate,
        "duration_seconds": len(clean) / rate,
        "recipe": {"seed": SEED, "speech_peak": 0.30, "noise_snr_db": 5,
                   "noise": "Gaussian white noise plus a 120 Hz sine with relative amplitude 0.5",
                   "leading_silence_seconds": 1,
                   "pulse_shape": "clipped Gaussian bursts replacing input samples; first two at opposite rails",
                   "impulses": [{"start_seconds": t, "duration_seconds": d, "amplitude": a}
                                for t, d, a in events],
                   "normalization_after_processing": False},
        "mp3": {"included": args.mp3, "codec": "libmp3lame", "bitrate": "64k",
                "encoder_version": subprocess.check_output(
                    ["ffmpeg", "-version"], text=True
                ).splitlines()[0] if args.mp3 else None,
                "purpose": "listening only; lossless WAV is the sample-limit validation artifact"},
        "cases": cases,
        "files": files,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: case["checks"] for name, case in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
