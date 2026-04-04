# Sharpness + Resolution Fix TODO (Executed Step-by-Step)

- [x] 1. Update `losses/simple_losses.py` with edge/gradient/sharp losses.
- [x] 2. Create `data_loader_simple.py` for higher-resolution loading.
- [x] 3. Create `train_sharp.py`.
- [x] 4. Create `resume_sharp.py`.
- [x] 5. Create `sharpen_output.py`.
- [x] 6. Create `compare_results.py`.
- [x] 7. Create `quick_test_sharp.py`.
- [x] 8. Install dependencies and run scripts step-by-step.

## Notes
- Adapted paths to this workspace (`c:/Users/SOHAM/OneDrive/Desktop/AI`).
- Default data path in new scripts is `data` (contains `raw` and `reference`).
- Executed in practical smoke mode:
	- `resume_sharp.py --epochs 1`
	- `sharpen_output.py --max-images 3`
	- `compare_results.py --max-images 2`
	- `quick_test_sharp.py`
