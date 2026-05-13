# An Adaptive Hybrid-Domain Watermarking Framework via Bayesian Learning for Content Authentication and Integrity Verification

Research-oriented implementation of an adaptive hybrid-domain watermarking framework with Bayesian learning for content authentication and integrity verification under diverse geometric and signal-processing attacks.

Core techniques used in the pipeline:

- SIFT keypoints for robust ROI localization
- Entropy-based ROI ranking
- SWT + DCT + QIM embedding
- Optional affine correction at extraction
- Bayesian optimization for alpha tuning

## 1. Repository Goal

This repository targets reproducible experiments for robust watermarking.

Primary goals:

- Evaluate robustness of watermark extraction under diverse attacks.
- Track imperceptibility and robustness metrics (PSNR, SSIM, NC, BER).
- Support image, text, and numeric ID watermark modes.
- Provide a baseline research pipeline that can be extended for new attacks, embedding rules, and optimization strategies.

## 2. Repository Structure

```text
.
|-- data/
|   |-- README.md
|   |-- host_images/                 # host images used as cover images
|   `-- watermark/                   # watermark assets (default: cict.png)
|-- models/
|   |-- README.md
|   `-- watermark_pipeline.py        # core algorithm + attack/eval flow
|-- notebooks/
|   |-- 01_data_preview.ipynb
|   |-- 02_embedding_steps.ipynb
|   |-- 03_attack_and_recovery.ipynb
|   `-- 04_metrics_and_charts.ipynb
|-- results/
|   |-- README.md
|   `-- ...                          # generated outputs
|-- scripts/
|   |-- README.md
|   `-- main.py                      # experiment entry script
|-- requirements.txt
`-- README.md
```

## 3. Environment Setup (Starting From Anaconda Base)

Recommended Python version: 3.10 to 3.12.

### 3.1 Create and activate a clean conda environment

```bash
conda create -n wm-pipeline python=3.11 -y
conda activate wm-pipeline
```

### 3.2 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Quick dependency check

```bash
python -c "import cv2, pywt, scipy, skimage, pandas, matplotlib, openpyxl, optuna; print('OK')"
```

### 3.4 Required packages (from requirements.txt)

- numpy
- opencv-contrib-python
- PyWavelets
- scipy
- scikit-image
- pandas
- matplotlib
- openpyxl
- optuna

## 4. Data Preparation

Place files as follows:

- Host images: data/host_images/
- Watermark image (default): data/watermark/cict.png

Input recommendations:

- Host images: png, jpg, bmp that OpenCV can read.
- Watermark image: grayscale-friendly, binary-like logo/text performs best.
- Use multiple host images with different textures for robust evaluation.

Current repository state:

- No fixed benchmark dataset link is provided yet.
- Add your own host set or document your dataset source in data/README.md.

## 5. How To Run Experiments

From project root:

```bash
python scripts/main.py --help
python scripts/main.py
```

What scripts/main.py does:

1. Loads all host images from data/host_images.
2. Optionally optimizes alpha globally/per-image using Bayesian optimization.
3. Runs embedding and extraction under many attacks.
4. Saves attacked and recovered outputs.
5. Writes per-image metric tables to an Excel summary (including extracted text/id when applicable).

## 6. Parameters And Experiment Scenarios

The script now supports command-line flags via argparse.

Main CLI parameters:

- --img-dir: directory containing host images
- --wm-raw: watermark image path (image mode)
- --mode: image, text, or id
- --text-input: watermark text (text mode, UTF-8)
- --id-input: numeric ID (id mode, digits only)
- --text-encoding: text encoding for text mode (default: utf-8)
- --repeat-k: bit repetition factor (odd number)
- --payload-repeat: payload repetition factor
- --auto-repeat-k / --no-auto-repeat-k
- --fill-repeat-k / --no-fill-repeat-k
- --fill-repeat-payload / --no-fill-repeat-payload
- --repeat-k-max
- --payload-repeat-max
- --repeat-k-target-success
- --alpha: embedding strength baseline
- --n: number of selected ROIs
- --seed: random seed for reproducibility
- --auto-optimize-alpha-global / --no-auto-optimize-alpha-global
- --auto-optimize-alpha-per-image / --no-auto-optimize-alpha-per-image
- --use-affine-correction / --no-use-affine-correction
- --global-bo-trials: BO trials for global optimization
- --per-image-bo-trials: BO trials for per-image optimization

CLI examples (copy-paste):

```bash
# 1) Baseline fixed alpha (no BO)
python scripts/main.py --no-auto-optimize-alpha-global --no-auto-optimize-alpha-per-image --alpha 100 --n 12 --seed 42

# 2) Global BO only
python scripts/main.py --auto-optimize-alpha-global --no-auto-optimize-alpha-per-image --global-bo-trials 8

# 3) Per-image BO only (default behavior)
python scripts/main.py --no-auto-optimize-alpha-global --auto-optimize-alpha-per-image --per-image-bo-trials 10

# 4) Text watermark mode (UTF-8)
python scripts/main.py --mode text --text-input "Đặng Lâm Sơn" --alpha 95 --n 12

# 5) ID watermark mode
python scripts/main.py --mode id --id-input 0123456789 --alpha 95 --n 12

# 6) Fill repeat-k to use 32x32
python scripts/main.py --mode text --fill-repeat-k

# 7) Fill payload repetition to use 32x32
python scripts/main.py --mode text --fill-repeat-payload

# 8) Disable affine correction for ablation
python scripts/main.py --no-use-affine-correction
```

Suggested experiment scenarios:

1. Baseline (fixed alpha)
   - auto_optimize_alpha_global = False
   - auto_optimize_alpha_per_image = False
2. Global BO only
   - auto_optimize_alpha_global = True
   - auto_optimize_alpha_per_image = False
3. Per-image BO
   - auto_optimize_alpha_global = False
   - auto_optimize_alpha_per_image = True
4. Watermark modality comparison
   - MODE = "image" vs MODE = "text"
5. Geometry handling ablation
   - run with and without affine correction (requires code toggle)

## 7. Outputs, Main Results, And Figures

Per host image named <host_name>, generated files are stored in:

- results/<host_name>/attack/ (attacked images)
- results/<host_name>/recover/ (recovered watermark images)

Global summary files:

- results/watermark_binary.jpg
- results/watermark_evaluation_summary.xlsx

Metrics reported:

- PSNR (image quality)
- SSIM (structural similarity)
- NC (watermark similarity)
- BER (bit error rate)
- ExtractedText (text mode)
- ExtractedID (id mode)

Current repository state:

- The pipeline exports numeric results to Excel.
- Pre-generated chart images are not included yet.

How to present key results in your report:

1. Build attack-wise bar charts for NC/BER per host.
2. Build quality-vs-robustness plots (PSNR vs NC).
3. Compare scenario pairs (fixed alpha vs BO) using means/std.

## 8. Jupyter Notebook Guidance (Visualization)

Available notebooks:

1. notebooks/01_data_preview.ipynb
   - visualize host and watermark inputs
2. notebooks/02_embedding_steps.ipynb
   - inspect workflow and preview embed/attack/recover outputs
3. notebooks/03_attack_and_recovery.ipynb
   - compare attacked images and recovered watermark images
4. notebooks/04_metrics_and_charts.ipynb
   - load Excel summary and generate metric charts

How to run notebooks:

```bash
jupyter notebook
```

Then open files in the notebooks/ folder and run cells from top to bottom.

Each notebook should include:

- fixed random seed
- file path assumptions
- expected output images/tables

## 9. Demo Guidance (Application)

Current repository state:

- Web demo is available for quick usage/testing.
- Full research experiments are still run locally via scripts/main.py.

Web demo:

- URL: https://sondepzaivn.github.io/Web-For-Watermarking-Images/
- Purpose: fast interactive demo for watermarking workflow on browser.
- Recommended use: presentation, quick trial, and qualitative checking.

Suggested web demo flow:

1. Open the deployed page.
2. Upload/select host image and watermark image.
3. Run embed/extract actions from the UI.
4. Inspect visual outputs and compare quality/recovery.

Minimal demo flow:

1. Put one host image in data/host_images/.
2. Put one watermark in data/watermark/cict.png.
3. Run python scripts/main.py.
4. Show attack and recover folders in results/<host_name>/.
5. Open results/watermark_evaluation_summary.xlsx for metrics.

## 10. Other Utilities And Notes

### 10.1 Reproducibility

- Keep seed fixed.
- Keep BO options fixed (n_trials, alpha range, thresholds).
- Keep package versions pinned via requirements.txt.

### 10.2 Troubleshooting

No images found:

- Verify files exist under data/host_images.

SIFT unavailable:

- Ensure opencv-contrib-python is installed in the active environment.

Excel write issues:

- Ensure openpyxl is installed.
- Close the Excel file before rerunning the script.
- Extracted text is sanitized to avoid Excel illegal character errors.

### 10.3 Planned Improvements

- Extend CLI with advanced attack/BO threshold controls.
- Add benchmark dataset references and figure templates.

## 11. Citation

Related manuscript title:

- An Adaptive Hybrid-Domain Watermarking Framework via Bayesian Learning for Content Authentication and Integrity Verification

Publication status:

- The manuscript is currently unpublished.
- Formal citation metadata (journal/conference, year, DOI, pages) is not available yet.
- Please cite this repository URL and commit/version for now.

## 12. Authors

Corresponding author:

- Hai Thanh Nguyen: nthai.cit@ctu.edu.vn

Contributing authors:

- Phuong Ngan Truong Nguyen: phuongb2404957@student.ctu.edu.vn
- Son Lam Dang: sonb2405134@student.ctu.edu.vn
- Tan Pham: tanb2303848@student.ctu.edu.vn
- Khoa Anh Le: khoab2303756@student.ctu.edu.vn
- Phuong Bich Dang: phuongb2404958@student.ctu.edu.vn
- Tran Phuong Quang Nguyen: tranb2404970@student.ctu.edu.vn

## 13. License

This project is released under the license specified in the LICENSE file.
