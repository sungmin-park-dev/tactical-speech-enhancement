from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from tactical_speech_enhancement import artifacts


def test_corrupt_asset_rejected(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="size"):
        artifacts.verify(path)


def test_failed_fetch_preserves_existing_file_and_removes_partial(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"existing")
    monkeypatch.setattr(
        artifacts,
        "manifest",
        lambda: {
            "artifact": {
                "id": "test",
                "size_bytes": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
                "source_url": "https://example.invalid/model",
            }
        },
    )

    class Reply(io.BytesIO):
        url = "https://example.invalid/model"

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", lambda *a, **kw: Reply(b"oversize"))
    with pytest.raises(ValueError, match="exceeds"):
        artifacts.fetch(path, force=True)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_fetch_atomically_installs_only_verified_content(tmp_path, monkeypatch):
    data = b"abc"
    monkeypatch.setattr(
        artifacts,
        "manifest",
        lambda: {
            "artifact": {
                "id": "test",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_url": "https://example.invalid/model",
            }
        },
    )

    class Reply(io.BytesIO):
        url = "https://example.invalid/model"

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", lambda *a, **kw: Reply(data))
    path = tmp_path / "model.onnx"
    assert artifacts.fetch(path)["sha256"] == hashlib.sha256(data).hexdigest()
    assert path.read_bytes() == data


def test_env_model_path_overrides_user_cache(monkeypatch):
    monkeypatch.setenv("TSE_MODEL_PATH", "/tmp/test-model.onnx")
    assert artifacts.default_model_path() == Path("/tmp/test-model.onnx")
