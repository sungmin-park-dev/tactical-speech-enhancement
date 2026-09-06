from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tactical_speech_enhancement.artifacts import default_model_path
from tactical_speech_enhancement.cli import benchmark, load_settings, main, process_file
from tactical_speech_enhancement.config import Settings


def test_configuration_rejects_unknown_and_nonfinite_fields(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("peak_threshold = nan\n")
    with pytest.raises(ValueError):
        load_settings(config)
    config.write_text("unknown = 1\n")
    with pytest.raises(TypeError):
        load_settings(config)


def test_process_refuses_input_overwrite_before_opening_model(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="different"):
        process_file(path, path, tmp_path / "absent.onnx", Settings())
    assert path.read_bytes() == b"original"


def test_process_rejects_stereo_and_wrong_rate(tmp_path):
    for audio, rate in ((np.zeros((300, 2)), 16000), (np.zeros(300), 48000)):
        path = tmp_path / "input.wav"
        sf.write(path, audio, rate)
        with pytest.raises(ValueError, match="mono 16000"):
            process_file(path, tmp_path / "out.wav", tmp_path / "absent.onnx", Settings())


def test_real_model_file_cli_preserves_length_limits_output_and_reports_settings(tmp_path, capsys):
    source = Path(__file__).parent / "fixtures/speech/speaker-0061.wav"
    output, report_path = tmp_path / "out.wav", tmp_path / "report.json"
    assert (
        main(
            [
                "--model-path",
                str(default_model_path()),
                "process",
                str(source),
                str(output),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    samples, rate = sf.read(output, dtype="int16")
    assert rate == 16000 and len(samples) == 160000
    assert np.max(np.abs(samples.astype(np.int32))) <= int(32768 * 10 ** (-6 / 20))
    assert np.any(samples[16000:])
    report = json.loads(report_path.read_text())
    assert report["settings"]["jitter_target_frames"] == 4
    assert report["pipeline"]["worker_mode"] == "offline_synchronous"
    assert report["pipeline"]["worker_stopped"]
    assert report["pipeline"]["model_results_used"] > 100


def test_benchmark_is_bounded_and_labels_actual_environment():
    report = benchmark(default_model_path(), 10, 1, Settings(telemetry_capacity=4))
    assert report["recent_window_frames"] == 4
    assert report["frames"] == 10
    assert report["settings"]["model_deadline_ms"] == 16
    assert report["max_ms"] >= report["mean_ms"] >= 0


def test_report_cannot_overwrite_input(tmp_path):
    source = tmp_path / "in.wav"
    source.write_bytes(b"original")
    with pytest.raises(SystemExit) as error:
        main(["process", str(source), str(tmp_path / "out.wav"), "--report", str(source)])
    assert error.value.code == 2
    assert source.read_bytes() == b"original"
