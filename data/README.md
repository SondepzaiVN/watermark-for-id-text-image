# Data Directory

This folder stores input data references for the watermarking pipeline.

## Structure

- `host_images/`: host images used as cover images for watermark embedding.
- `host_images_extract/`: images used for extraction-only mode.
- `watermark/`: watermark assets (for example `cict.png`).

## Notes

- Keep large datasets out of GitHub.
- Prefer adding download links and preparation steps in this file.
- Recommended input formats: `.png`, `.jpg`, `.bmp`.

## Expected Usage

The default script reads:

- Host images from `data/host_images`
- Watermark image from `data/watermark/cict.png`
