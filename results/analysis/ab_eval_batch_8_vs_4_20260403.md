# A/B Evaluation: batch_size 8 vs 4 (2026-04-03)

## Setup

- Control run: unet_20260403_0811
- Treatment run: unet_20260403_1107
- Fixed settings across runs: 50 epochs, learning_rate 1e-4, loss combined, ssim_weight 0.5, same dataset and split.

## Metrics

### Control (batch_size=8, unet_20260403_0811)

- Best epoch: 49
- Best val_loss: 0.066380
- Best val_mae: 0.072987
- Final val_loss: 0.070340
- Final val_mae: 0.077854

### Treatment (batch_size=4, unet_20260403_1107)

- Best epoch: 49
- Best val_loss: 0.069955
- Best val_mae: 0.077466
- Final val_loss: 0.070303
- Final val_mae: 0.077846

## Delta (treatment - control)

- Best val_loss: +0.003574 (worse)
- Best val_mae: +0.004479 (worse)
- Final val_loss: -0.000037 (roughly neutral)
- Final val_mae: -0.000009 (roughly neutral)

## Decision

- Keep batch_size=8 as default.
- batch_size=4 did not improve best validation metrics under equal training budget.
