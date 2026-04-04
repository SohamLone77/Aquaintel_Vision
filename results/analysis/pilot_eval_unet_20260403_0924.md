# Pilot Evaluation: run `unet_20260403_0924`

## Experiment setup

- Goal: test configurable loss wiring with a controlled loss-weight change.
- Variable changed: `loss.ssim_weight` from `0.5` to `0.6`.
- Loss type: `combined`.
- Training length: `10` epochs (short pilot).
- Dataset/split: unchanged (`data/raw`, `data/reference`, validation split `0.2`).

## Pilot metrics (`unet_20260403_0924`)

- Best epoch: `9`
- Best `val_loss`: `0.109345`
- Best `val_mae`: `0.093316`
- Final epoch `val_loss`: `0.110209`
- Final epoch `val_mae`: `0.094260`
- Final epoch train `loss`: `0.103072`

## Reference runs

### Baseline full run (`unet_20260403_0500`, 50 epochs)

- Best epoch: `49`
- Best `val_loss`: `0.069774`
- Best `val_mae`: `0.076595`

### Re-run full run (`unet_20260403_0811`, 50 epochs)

- Best epoch: `49`
- Best `val_loss`: `0.066380`
- Best `val_mae`: `0.072987`

## Interpretation

- The pilot run converges smoothly in the first 10 epochs and shows stable train/validation tracking.
- Direct numeric comparison against 50-epoch runs is not apples-to-apples because this pilot stopped early.
- The primary objective of this pilot (verify configurable loss plumbing and run behavior) is successful.

## Recommended next controlled test

- Keep epochs at `50` for comparability.
- Run two back-to-back experiments with only one variable changed each time:
  1. `ssim_weight=0.5` (control)
  2. `ssim_weight=0.6` (treatment)
- Compare best `val_loss`, best `val_mae`, and sample predictions under identical epoch budget.
