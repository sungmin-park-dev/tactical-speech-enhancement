# Third-party notices

Project code is distributed under the [MIT license](LICENSE). Third-party
materials retain the licenses and attribution below.

| Material | Source and attribution | License |
|---|---|---|
| GTCRN model and model-specific inference conventions | [GTCRN](https://github.com/Xiaobin-Rong/gtcrn), Copyright (c) 2024 Rong Xiaobin | [MIT text](licenses/GTCRN-MIT.txt) |
| ONNX Runtime dependency | [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime), Copyright (c) Microsoft Corporation | [MIT text](licenses/ONNX-Runtime-MIT.txt) |
| Ten cropped speech fixtures | [LibriSpeech / OpenSLR SLR12](https://www.openslr.org/12/), Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur; audiobook recordings originate from LibriVox | [CC BY 4.0 text](licenses/CC-BY-4.0.txt) |

The GTCRN ONNX file is downloaded from the
[sherpa-onnx speech-enhancement-models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models).
Its exact SHA-256 and model identity are pinned in the package manifest. This is
a model distribution source; the project does not bundle sherpa-onnx source code
or claim that its Apache-2.0 project license replaces GTCRN's MIT license.

The speech fixture modifications are lossless FLAC decoding, numeric utterance
selection, concatenation, cropping to ten seconds per speaker and WAV encoding.
[Fixture provenance](tests/fixtures/speech/manifest.json) records every selected
source segment and both source and output hashes. Tests additionally scale each
fixture to three digital peaks and prepend silence. No transcript files are
redistributed. Attribution does not imply endorsement by the original creators.

Other Python dependencies are installed from their distributions and retain
their own license notices. No third-party runtime binary is committed here.
