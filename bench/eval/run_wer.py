"""WER with faster-whisper (base.en, int8, greedy) against LibriSpeech transcripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from metrics import normalise_text

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
TS = SP / "eval/testset"
OUT = SP / "eval/out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="wer")
    args = ap.parse_args()
    meta = json.loads((TS / "meta.json").read_text())
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads,
                         download_root=str(SP / "weights/whisper"))
    out_path = SP / f"eval/{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["system"], r["cond"], r["uid"]))
    with open(out_path, "a") as f:
        for clip in meta["clips"]:
            cond, uid = clip["cond"], clip["uid"]
            if cond == "c10_echo_only" or (args.conds and cond not in args.conds):
                continue
            for system in args.systems:
                if (system, cond, uid) in done:
                    continue
                p = OUT / system / cond / f"{uid}.wav"
                if not p.exists():
                    continue
                y, fs = sf.read(p, dtype="float32")
                peak = np.abs(y).max()
                if peak > 0:
                    y = y * (0.5 / peak)       # level-normalise for ASR only
                segs, _ = model.transcribe(y, language="en", beam_size=1, vad_filter=False,
                                           condition_on_previous_text=False)
                hyp = normalise_text(" ".join(s.text for s in segs))
                ref = normalise_text(clip["text"])
                m = jiwer.process_words(ref, hyp)
                f.write(json.dumps({"system": system, "cond": cond, "uid": uid, "hyp": hyp,
                                    "ref_words": len(ref.split()), "errors": m.substitutions + m.deletions + m.insertions,
                                    "wer": m.wer}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
