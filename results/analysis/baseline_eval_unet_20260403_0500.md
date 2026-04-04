# Baseline Evaluation Report

## Run
- Model name: unet_20260403_0500
- Checkpoint evaluated: models/checkpoints/unet_20260403_0500_best.h5
- Dataset split: 712 train / 178 validation
- Input size: 128x128

## Quantitative Results
- Best epoch: 49
- Best validation loss: 0.069774
- Best validation MAE: 0.076595
- Final train loss (epoch 50): 0.064930
- Final validation loss (epoch 50): 0.070764
- Final generalization gap (val_loss - loss): 0.005834

## Post-Training Checkpoint Evaluation
- Evaluated best checkpoint loss: 0.069774
- Evaluated best checkpoint MAE: 0.076595
- Validation steps: 23
- Validation samples: 178

## Qualitative Observations
Source image reviewed: results/training_plots/unet_20260403_0500_samples.png

- Sample 1: significant underwater color cast is reduced and contrast is improved, but prediction appears warmer than target.
- Sample 2: local structure and edges are improved; prediction has stronger contrast and darker regions than target.
- Sample 3: model removes green cast effectively; prediction tends to neutral/gray and may slightly under-preserve cyan tone.
- Sample 4: texture detail improves, though prediction is somewhat over-sharpened compared to target.

Overall qualitative summary:
- The model is clearly learning underwater enhancement and improves visibility.
- Remaining issue pattern is color-temperature bias and occasional over-contrast.

## Recommendation
Use this run as the baseline winner for the next experiment cycle.

Suggested next controlled experiment (single change):
1. Keep architecture and optimizer fixed.
2. Adjust loss to reduce warm over-bias (for example increase SSIM weight moderately or add a mild color-consistency term).
3. Run 5-10 epoch pilot first, then compare against this baseline before running full 50 epochs.
