"""Integrity-pinned model download independent of audio hardware."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path


def manifest() -> dict:
    return json.loads(
        files("tactical_speech_enhancement").joinpath("model_manifest.json").read_text()
    )


def default_model_path() -> Path:
    override = os.environ.get("TSE_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "tactical-speech-enhancement" / "gtcrn_simple.onnx"


def verify(path: Path) -> dict:
    spec = manifest()["artifact"]
    if path.stat().st_size != spec["size_bytes"]:
        raise ValueError("Model size does not match the pinned manifest")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != spec["sha256"]:
        raise ValueError("Model SHA-256 does not match the pinned manifest")
    return {"model": spec["id"], "size_bytes": spec["size_bytes"], "sha256": digest}


def fetch(path: Path, *, force: bool = False, timeout: float = 30.0) -> dict:
    if path.exists() and not force:
        return verify(path)
    spec = manifest()["artifact"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".gtcrn-", delete=False) as out:
            temporary = Path(out.name)
            request = urllib.request.Request(
                spec["source_url"], headers={"User-Agent": "tse-model-fetch/1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("Model download redirected away from HTTPS")
                count = 0
                while chunk := response.read(64 * 1024):
                    count += len(chunk)
                    if count > spec["size_bytes"]:
                        raise ValueError("Model download exceeds the pinned size")
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        result = verify(temporary)
        os.replace(temporary, path)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
