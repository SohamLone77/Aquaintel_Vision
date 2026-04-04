# Implementation Progress

## Completed

- [x] Step 1: Centralized runtime config
  - Added `config.yaml` as the single source of defaults.
  - Added `utils/config_loader.py` with validation and env override support (`DATA_PATH`).
  - Updated `train_unet.py` and `train_complete.py` to load config and fail fast with clear errors.

- [x] Step 2: Dataset validation gate
  - Added `scripts/validate_dataset.py`.
  - Integrated dataset validation in trainer startup before dataloader creation.
  - Removed silent dummy-data fallback from startup path.

- [x] Step 3: Deterministic dataset download/extract flow
  - Reworked `scripts/download_dataset.py` to avoid import-time side effects.
  - Added extraction normalization and post-extract validation.
  - Added archive/data failure messages that are explicit and deterministic.

- [x] Environment setup milestone
  - Created `.venv311` using Python 3.11.
  - Installed project dependencies from `requirements.txt`.
  - Verified TensorFlow import in `.venv311`.

## In Progress

- [x] Step 4: Model run registry metadata
  - Added `utils/model_registry.py`.
  - Integrated into `train_complete.py` after training ends.

## Next Actions (One by One)

1. Prepare dataset in `data/raw` and `data/reference`:
   - Option A: `./.venv311/Scripts/python.exe scripts/download_dataset.py`
   - Option B: place paired images manually, then run validator.

2. Validate dataset:
   - `./.venv311/Scripts/python.exe scripts/validate_dataset.py --strict-names`

3. Run training once data is available:
  - Metadata will be written automatically to `results/model_registry.json`.

4. Run 1-epoch smoke training:
   - `./.venv311/Scripts/python.exe -c "from train_unet import main; main({'epochs': 1, 'batch_size': 2})"`
  - Completed successfully with validated artifacts and metrics.

5. Run baseline training:
   - `./.venv311/Scripts/python.exe train_unet.py`
  - Completed: run `unet_20260403_0500`.

6. Baseline evaluation and report:
  - Evaluated `models/checkpoints/unet_20260403_0500_best.h5` on validation split.
  - Wrote report to `results/analysis/baseline_eval_unet_20260403_0500.md`.

7. Loss config wiring + full rerun:
  - Wired `loss_type` and `ssim_weight` into compile path in `train_complete.py`.
  - Completed full rerun: `unet_20260403_0811` (50 epochs).

8. Controlled short pilot (single variable change):
  - Changed `loss.ssim_weight` to `0.6` for a 10-epoch pilot.
  - Completed pilot run: `unet_20260403_0924`.
  - Wrote report to `results/analysis/pilot_eval_unet_20260403_0924.md`.

## Upcoming

1. Run strict A/B comparison at equal epoch budget (50 epochs):
  - Control: `ssim_weight=0.5` (run `unet_20260403_0811`)
  - Treatment: `ssim_weight=0.6` (run `unet_20260403_0938`)
  - Completed; report: `results/analysis/ab_eval_ssim_0p5_vs_0p6_20260403.md`.

2. Compare metrics and prediction quality from both A/B runs.
  - Completed: `0.5` remains winner on best validation metrics.

3. Lock in best loss configuration and proceed to next optimization axis.
  - Completed: default kept at `ssim_weight=0.5`.

## Next Optimization Axis

1. Run learning-rate A/B pilots (same epochs/data):
  - `learning_rate=1e-4` (control, `unet_20260403_0811`)
  - `learning_rate=5e-5` (treatment, `unet_20260403_1019`)
  - Completed; report: `results/analysis/ab_eval_lr_1e4_vs_5e5_20260403.md`.
  - Decision: keep `learning_rate=1e-4`.

2. If LR pilot is neutral, run batch-size A/B:
  - `batch_size=8` (control)
  - `batch_size=4` (treatment)
  - Completed; report: `results/analysis/ab_eval_batch_8_vs_4_20260403.md`.
  - Decision: keep `batch_size=8`.

3. Next axis (in progress): augmentation A/B.
  - Control: current augmentation settings.
  - Treatment: reduced augmentation intensity to check stability/generalization tradeoff.
  - Augmentation controls are now config-driven via `augmentation` section in `config.yaml`.
  - Pilot completed (10 epochs):
    - Control: `unet_augstd_20260403_pilot`
    - Treatment: `unet_auglight_20260403_pilot`
    - Report: `results/analysis/ab_eval_aug_standard_vs_light_pilot_20260403.md`
    - Pilot winner: `light`.

4. Next required validation:
  - Run strict 50-epoch augmentation A/B:
    - Control: `profile=standard`
    - Treatment: `profile=light`
  - Lock final augmentation profile only after equal-budget comparison.
