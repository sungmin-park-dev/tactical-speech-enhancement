# Public speech fixtures

`speech/` contains ten 10-second excerpts from **LibriSpeech test-clean** under
**CC BY 4.0**. These are public read-speech recordings, not project recordings.
Attribution and modification details are in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md),
and the complete license is [CC-BY-4.0.txt](../../licenses/CC-BY-4.0.txt).

Selection is deterministic: numeric speaker IDs, then numeric utterance IDs;
concatenate utterances and keep the first 160,000 mono samples per speaker.
`speech/manifest.json` pins the archive, source segments and generated WAV files.
The fixture files contain PCM audio only, without transcripts or source metadata.

Rebuild from the official archive stored **outside the repository**:

```sh
python scripts/prepare_speech_fixtures.py --archive /tmp/tse-corpus/test-clean.tar.gz
```

Or download the pinned archive into an external cache and rebuild:

```sh
python scripts/prepare_speech_fixtures.py --cache-dir /tmp/tse-corpus
```

The archive's SHA-256 is checked before any fixture is generated. This directory
contains only the 100-second selected subset; automated tests need no download.
