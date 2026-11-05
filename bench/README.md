# Benchmark and measurement tools

Tools used to evaluate this project without changing the pipeline code:
a synthetic stress test set, objective metrics with paired bootstrap
confidence intervals, a real-time two-peer harness, and a measurement kit
for Raspberry Pi 5. Results produced with these tools are **software and
synthetic-noise results**, not battlefield or device performance.

| Folder | Purpose |
|---|---|
| `eval/` | Test set (`build_testset.py`), systems under test (`systems.py`), metrics (`metrics.py`: SI-SDR, PESQ-WB, ESTOI, speech-loss ratio, event windows, ERLE), runners (`run_quality.py`, `run_wer.py`), aggregation (`analyze.py`, `event_windows.py`, `make_q2_tables.py`) |
| `rt/` | Real-time harness: the unmodified `tse peer` path with a fake audio device (`fake_sd.py`, ppm clock offset, xrun flags), UDP impairment proxy (`udp_proxy.py`), per-thread CPU / queue / latency telemetry (`peer_harness.py`, `run_condition.py`), and a discrete-event model of buffer policies (`naive_sim.py`) |
| `rp5/` | Scripts to run on the Raspberry Pi 5; procedure in `rp5/README_RP5.md` (Korean) |

## Setup

```sh
python -m pip install -e '.[audio,dev,bench]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # only for Demucs baselines
tse model fetch
bench/fetch_assets.sh            # LibriSpeech test-clean, optional baselines
```

Assets go to `$TSE_BENCH_DIR` (default `bench/work/`, git-ignored). Nothing
downloaded is redistributed here. Licenses: LibriSpeech CC BY 4.0, DTLN MIT,
facebookresearch/denoiser **CC BY-NC 4.0** (non-commercial; used only as a
comparison baseline).

## Typical runs

```sh
python bench/eval/build_testset.py                       # 12 conditions x 24 utterances
python bench/eval/run_quality.py --systems S0_noisy S1_gtcrn_only S2_current_pipeline --workers 4
python bench/eval/run_wer.py --systems S0_noisy S1_gtcrn_only S2_current_pipeline
python bench/eval/analyze.py                             # summary.json, diffs.json, tables.md
python bench/rt/run_condition.py --outdir runs/drift --duration 600 --ppm-a 250 --ppm-b -250 \
  --input speech.wav --model-path ~/.cache/tactical-speech-enhancement/gtcrn_simple.onnx
python bench/rt/naive_sim.py
```

The real-time harness measures the software path from the capture callback
to the peer's playout callback; device buffers, the radio link and the
acoustic path are not included.
