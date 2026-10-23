# Dataset Directory for Skin Disease Detection

This directory contains the dataset used for training, validation, and testing.

## Dataset Structure

When prepared, the directory structure will be organized as follows:

```
data/
├── raw/
│   ├── HAM10000_metadata.csv
│   └── images/
│       ├── ISIC_0024306.jpg
│       └── ...
└── processed/
    ├── train/
    │   ├── akiec/
    │   ├── bcc/
    │   ├── bkl/
    │   ├── df/
    │   ├── mel/
    │   ├── nv/
    │   └── vasc/
    ├── val/
    │   └── ... (same subfolders)
    ├── test/
    │   └── ... (same subfolders)
    └── dataset_summary.json
```

## How to Obtain the HAM10000 Dataset Legally

The HAM10000 ("Human Against Machine with 10000 training images") dataset is a collection of multi-source dermatoscopic images of pigmented lesions.

- **Official Source**: [Harvard Dataverse (HAM10000 Dataset)](https://doi.org/10.7910/DVN/DBW86T) or [ISIC Archive](https://www.isic-archive.com/)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- **Instructions**:
  1. Download `HAM10000_metadata.csv`.
  2. Download image parts `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`.
  3. Extract all `.jpg` files into `data/raw/images/`.
  4. Place `HAM10000_metadata.csv` into `data/raw/`.
  5. Run `python training/prepare_data.py`.

## Quick Pipeline Verification (Smoke-Test Mode)

If you do not have the full 3 GB dataset yet or are demonstrating the software pipeline on a student laptop:
```bash
python training/prepare_data.py --smoke-test
```
This generates a synthetic verified miniature split adhering to lesion specifications so you can execute the entire training, evaluation, and web inference pipeline end-to-end without waiting hours for downloads.
