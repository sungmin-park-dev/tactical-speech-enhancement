# CODEX.md — Current Status

Claude/Codex should read this file first. This is the shortest reliable summary of the current repo state.

## Goal

BC + AC fusion speech enhancement for patent-support experiments.
Current patent-aligned assumption:

- AC can saturate in extreme noise
- BC is relatively more reliable
- saturation mask should describe AC corruption and drive BC-favoring fusion/restoration

## What Was Fixed

As of 2026-04-02, these core fixes are implemented:

1. `data/saturation.py`
- `soft_mask()` now returns `0.0` for zero clipping
- zero-clipping assert added in `__main__`

2. `data/pipeline.py`
- saturation moved from BC path to AC path
- current flow:

```text
clean -> BC simulator -> BC signal
clean -> noise mix -> AC mixed -> saturation -> AC signal + masks
```

3. `models/dpcrn_dual.py`
- CRM base spectrum changed from AC to BC

4. `models/fusion.py`
- fusion gate now uses true complex magnitude instead of element-wise `abs()`

## Verified Locally

Passed:

```bash
python -m data.saturation
python -m data.pipeline --config configs/data_config.yaml --env general --split train --n_train 2 --dry_run
python -m models.fusion
python -m models.dpcrn_dual
python train.py --config configs/train_config.yaml --data_config configs/data_config.yaml --epochs 1 --batch_size 2 --env general
```

Note:
- local train/pipeline checks were fallback or dummy-data level only
- this is code-validity confirmation, not real experiment evidence

## Colab / Drive Status

Old processed training data made before the bug fixes is invalid and should not be reused.

Current notebook storage root is now:

```text
/content/drive/MyDrive/Colab Notebooks/ARMY Projects/2026-tactical-speech-enhancement
```

Current archive paths:

```text
data archive:
/content/drive/MyDrive/Colab Notebooks/ARMY Projects/2026-tactical-speech-enhancement/data/processed_data.tar.gz

results archive:
/content/drive/MyDrive/Colab Notebooks/ARMY Projects/2026-tactical-speech-enhancement/results/
```

Important:
- `tactical-speech-data` is not referenced by the current notebooks
- previous notebook path `/content/drive/MyDrive/tactical-speech-enhancement` is no longer the active target

## Notebook Workflow

- `notebooks/download_to_drive.ipynb`
  - safe mode
  - does **not** auto-delete an existing Drive archive

- `notebooks/replace_processed_archive.ipynb`
  - one-time archive replacement notebook
  - deletes existing Drive archive and writes new one
  - requires manual `CONFIRM_REPLACE = True`

- `notebooks/train_colab.ipynb`
  - restores the processed archive from the new 2026 Drive root
  - intended for alpha search and full training

- `notebooks/evaluate_colab.ipynb`
  - also updated to use the new 2026 Drive root

## Current Practical Decision

- Keep current clean dataset choice for immediate alpha rerun
- Do not switch datasets before alpha search
- Re-synthesize processed data first, then rerun alpha search

## Main Remaining Gaps

- no real experiment results yet
- BWE is still not integrated into the actual forward restoration path
- train/evaluate pipeline is not yet fully wired for the final 100-condition matrix

## Next Recommended Steps

1. Commit current fixes and notebook path changes
2. In Colab, run `download_to_drive.ipynb` Step 0-2
3. If an old archive already exists at the new Drive path, run `replace_processed_archive.ipynb`
4. Run `train_colab.ipynb` for alpha search
5. After alpha selection, continue the 15-run DPCRN training set
