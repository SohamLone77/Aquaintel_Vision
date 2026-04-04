# A/B Evaluation (Pilot): augmentation standard vs light (2026-04-03)

## Setup

- Control run: unet_augstd_20260403_pilot
- Treatment run: unet_auglight_20260403_pilot
- Fixed settings: epochs=10, batch_size=8, learning_rate=1e-4, loss=combined, ssim_weight=0.5, same dataset and split.

## Metrics

### Control (standard)

- Best val_loss: 0.095830
- Best val_mae: 0.091736
- Final val_loss: 0.095830
- Final val_mae: 0.091736

### Treatment (light)

- Best val_loss: 0.092662
- Best val_mae: 0.091079
- Final val_loss: 0.092662
- Final val_mae: 0.091079

## Delta (treatment - control)

- Best val_loss: -0.003168 (better)
- Best val_mae: -0.000657 (better)
- Final val_loss: -0.003168 (better)
- Final val_mae: -0.000657 (better)

## Pilot decision

- Light augmentation is better in this 10-epoch pilot.
- Next required step: run strict 50-epoch A/B (standard vs light) before locking final default.
