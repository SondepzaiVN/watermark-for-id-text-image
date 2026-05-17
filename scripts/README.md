# Scripts Directory

This folder contains executable scripts for running experiments.

## Files

- `main.py`: primary entry point for batch processing host images.

## What `main.py` does

1. Loads host images and watermark input.
2. Optionally tunes alpha with Bayesian optimization.
3. Runs embedding and extraction under multiple attacks.
4. Saves images and Excel metrics to `results/`.
5. Extraction-only mode reads images from `data/host_images_extract` and uses key files in `results/`.

## Run

From project root:

```bash
python scripts/main.py
```
