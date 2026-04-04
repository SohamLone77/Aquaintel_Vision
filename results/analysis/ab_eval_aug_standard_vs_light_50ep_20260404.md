# Augmentation A/B Test Evaluation (STANDARD vs. LIGHT) - 50 Epochs

This document analyzes the 50-epoch A/B test comparing the `standard` augmentation profile against the `light` augmentation profile.

## Methodology

- **Control Run:** `unet_augstd_20260403_ab50` (Standard Augmentation)
- **Treatment Run:** `unet_20260404_0816` (Light Augmentation - *Attempted*)
- **Dataset:** 890 image pairs (80/20 train/val split)
- **Epochs:** 50
- **Loss:** Combined (SSIM weight: 0.5)

## Results Comparison

| Metric | Control (`standard`) | Treatment (`light` - actual: Placebo) |
|---|---|---|
| **Training Loss** | 0.0628 | 0.0645 |
| **Training MAE** | 0.0756 | 0.0768 |
| **Validation Loss** | 0.0716 | 0.0709 |
| **Validation MAE** | 0.0784 | 0.0772 |

## ⚠️ CRITICAL FINDING: The Placebo Effect

During evaluation, a critical configuration flaw was discovered that invalidates this specific A/B test run.

Although `config.yaml` was correctly updated to `profile: light` for the treatment run, the explicitly defined standard parameters (e.g., `flip_prob: 0.5`, `brightness_prob: 0.5`) were left in the YAML file. 

Due to the logic in `training/data_loader.py`:
```python
resolved = dict(profile_presets[profile])
resolved.update(cfg) # Overwrites preset with explicit YAML values
```
The explicit values in the YAML file overrode the `light` preset completely. As a result, **run `unet_20260404_0816` secretly used the exact same `standard` augmentation probabilities as the control run.** The minor differences in metrics seen above are purely due to stochastic noise between two identical training runs.

## Next Steps / User Action Required

To conduct a valid A/B test for `light` augmentation, we must remove the explicit probability overrides from `config.yaml` so the data loader can successfully fall back to the native `light` preset values defined in the code.
