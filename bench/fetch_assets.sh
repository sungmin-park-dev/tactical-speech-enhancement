#!/usr/bin/env bash
# Download the external assets used by the benchmark into $TSE_BENCH_DIR (default bench/work).
# None of these are redistributed by this repository; check each license before use.
#   LibriSpeech test-clean  CC BY 4.0           https://www.openslr.org/12/
#   facebookresearch/denoiser code + weights    CC BY-NC 4.0 (non-commercial; optional baseline)
#   DTLN ONNX models                            MIT
set -euo pipefail
W="${TSE_BENCH_DIR:-$(cd "$(dirname "$0")" && pwd)/work}"
mkdir -p "$W/data" "$W/ext/fb_denoiser/denoiser" "$W/ext/dtln" "$W/weights"
if [ ! -d "$W/data/ls/LibriSpeech/test-clean" ]; then
  curl -fL -o "$W/data/test-clean.tar.gz" https://www.openslr.org/resources/12/test-clean.tar.gz
  echo "32fa31d27d2e1cad72775fee3f4849a9  $W/data/test-clean.tar.gz" | md5sum -c -
  mkdir -p "$W/data/ls" && tar -xzf "$W/data/test-clean.tar.gz" -C "$W/data/ls"
fi
for f in __init__.py demucs.py resample.py utils.py; do
  curl -fsSL -o "$W/ext/fb_denoiser/denoiser/$f" "https://raw.githubusercontent.com/facebookresearch/denoiser/main/denoiser/$f"
done
curl -fsSL -o "$W/ext/fb_denoiser/LICENSE" https://raw.githubusercontent.com/facebookresearch/denoiser/main/LICENSE
for f in dns48-11decc9d8e3f0998.th dns64-a7761ff99a7d5bb6.th master64-8a5dfb4bb92753dd.th; do
  [ -f "$W/weights/$f" ] || curl -fL -o "$W/weights/$f" "https://dl.fbaipublicfiles.com/adiyoss/denoiser/$f"
done
for f in model_1.onnx model_2.onnx LICENSE; do
  src="pretrained_model/$f"; [ "$f" = LICENSE ] && src=LICENSE
  curl -fsSL -o "$W/ext/dtln/$f" "https://raw.githubusercontent.com/breizhn/DTLN/master/$src"
done
echo "assets ready in $W"
