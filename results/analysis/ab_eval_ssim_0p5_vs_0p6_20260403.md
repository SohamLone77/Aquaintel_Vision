# A/B Evaluation: SSIM weight 0.5 vs 0.6 (50 epochs)

## Setup

- Objective: compare two loss-weight settings under the same epoch budget.
- Model/data/training settings: unchanged except `loss.ssim_weight`.
- Loss type: `combined`.
- Control run: `unet_20260403_0811` with `ssim_weight=0.5`.
- Treatment run: `unet_20260403_0938` with `ssim_weight=0.6`.

## Metrics summary

### Control (`unet_20260403_0811`, 50 epochs)

- Best epoch: `49`
- Best val_loss: `0.066380`
- Best val_mae: `0.072987`
- Final val_loss: `0.070340`
- Final val_mae: `0.077854`

### Treatment (`unet_20260403_0938`, 50 epochs)

- Best epoch: `50`
- Best val_loss: `0.081249`
- Best val_mae: `0.076623`
- Final val_loss: `0.081249`
- Final val_mae: `0.076623`

## Deltas (treatment minus control)

- Best val_loss: `+0.014869` (worse)
- Best val_mae: `+0.003636` (worse)
- Final val_loss: `+0.010909` (worse)
- Final val_mae: `-0.001232` (slightly better)

## Decision

- Keep `ssim_weight=0.5` as the default for now.
- The 0.6 treatment did not beat control on best validation MAE and showed notably worse validation loss trajectory.

## Notes

- Because `val_loss` is computed with a weighted combined loss, MAE should remain a primary comparison metric across weight changes.
- Next optimization should focus on learning-rate schedule/batch-size or alternative loss families with the same A/B protocol.
