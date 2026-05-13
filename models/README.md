# Models Directory

This folder contains the core watermarking implementation.

## Files

- `watermark_pipeline.py`: end-to-end pipeline for embedding, extraction, attack simulation, evaluation, and alpha optimization.

## Main Components

- ROI selection with SIFT + entropy ranking
- Embedding with SWT + DCT + QIM
- Optional affine correction during extraction
- Attack benchmark suite
- Metrics: PSNR, SSIM, NC, BER

## Extension Tips

- Keep algorithm logic in this folder.
- Keep runnable entry scripts in `scripts/`.
- If checkpoints are added later, document their source and purpose here.
