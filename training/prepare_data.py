"""
Data preparation script for Skin Disease Detection Using CNN.
Handles raw dataset inspection, patient-level grouping to prevent data leakage,
and creation of train/validation/test directory splits.
Includes a verified --smoke-test mode for instant pipeline testing.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("prepare_data")


def create_smoke_test_dataset(output_raw_dir: Path, num_samples_per_class: int = 15) -> Tuple[Path, Path]:
    """
    Generate a miniature synthetic dataset resembling HAM10000 structure for smoke-testing.
    Generates synthetic lesion images with distinctive class patterns, colors, and patient groupings.
    Allows students to run the entire CNN pipeline locally without downloading 3GB files.
    """
    logger.info("Generating verified smoke-test dataset in %s...", output_raw_dir)
    images_dir = output_raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    classes = list(config.DEFAULT_CLASSES.keys())

    # Palettes for synthetic lesion types to make them distinct
    palettes = {
        "akiec": (220, 140, 130),  # Rough reddish-pink
        "bcc": (200, 110, 120),    # Translucent pearly pink
        "bkl": (130, 95, 65),      # Tan/brown warty
        "df": (160, 100, 70),      # Firm brownish nodule
        "mel": (50, 40, 45),       # Dark, irregular, asymmetrical
        "nv": (110, 75, 55),       # Uniform brown mole
        "vasc": (180, 30, 50)      # Cherry red / purple vascular
    }

    image_idx = 1
    patient_id_counter = 1000

    for cls in classes:
        base_color = palettes[cls]
        for s in range(num_samples_per_class):
            image_id = f"ISIC_SMOKE_{image_idx:06d}"
            # 2 images per patient for half the samples to test patient-leakage prevention
            if s % 2 == 0:
                patient_id_counter += 1
            patient_id = f"IP_{patient_id_counter}"
            lesion_id = f"IL_{patient_id_counter}"

            # Create synthetic lesion image (128x128 with skin tone and lesion spot)
            # Skin background: warm tone with subtle noise
            bg_r = np.random.randint(225, 245)
            bg_g = np.random.randint(190, 215)
            bg_b = np.random.randint(170, 195)
            img = Image.new("RGB", (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), (bg_r, bg_g, bg_b))
            draw = ImageDraw.Draw(img)

            # Draw lesion spot
            cx = config.IMAGE_WIDTH // 2 + np.random.randint(-10, 10)
            cy = config.IMAGE_HEIGHT // 2 + np.random.randint(-10, 10)
            rx = np.random.randint(20, 38)
            ry = np.random.randint(18, 36)

            # Class specific variance
            lr = int(np.clip(base_color[0] + np.random.randint(-15, 15), 0, 255))
            lg = int(np.clip(base_color[1] + np.random.randint(-15, 15), 0, 255))
            lb = int(np.clip(base_color[2] + np.random.randint(-15, 15), 0, 255))
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(lr, lg, lb))

            # Inner pigment variation
            draw.ellipse([cx - rx // 2, cy - ry // 2, cx + rx // 2, cy + ry // 2],
                         fill=(max(0, lr - 25), max(0, lg - 25), max(0, lb - 25)))

            img_path = images_dir / f"{image_id}.jpg"
            img.save(img_path, format="JPEG", quality=90)

            records.append({
                "lesion_id": lesion_id,
                "image_id": image_id,
                "dx": cls,
                "dx_type": "consensus",
                "age": 45.0,
                "sex": "female" if s % 2 == 0 else "male",
                "localization": "back" if s % 3 == 0 else "face",
                "dataset": "smoke_test"
            })
            image_idx += 1

    metadata_path = output_raw_dir / "HAM10000_metadata.csv"
    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    logger.info("Created %d synthetic samples across %d classes.", len(df), len(classes))
    return metadata_path, images_dir


def check_raw_dataset(raw_dir: Path) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Check if raw dataset files exist in the expected location.
    Looks for HAM10000_metadata.csv and an images directory or flat raw images.
    """
    metadata_candidates = [
        raw_dir / "HAM10000_metadata.csv",
        raw_dir / "metadata.csv",
        raw_dir / "HAM10000_metadata"
    ]
    metadata_path = next((p for p in metadata_candidates if p.exists()), None)

    images_dir_candidates = [
        raw_dir / "images",
        raw_dir / "HAM10000_images",
        raw_dir
    ]
    images_dir = None
    for cand in images_dir_candidates:
        if cand.exists() and any(cand.glob("*.jpg")):
            images_dir = cand
            break

    is_available = (metadata_path is not None and images_dir is not None)
    return is_available, metadata_path, images_dir


def split_dataset_without_leakage(
    df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Splits dataset into train, val, and test subsets.
    Crucial: Groups by lesion_id / patient_id if present to guarantee no duplicate
    lesions or patient images leak across splits!
    """
    logger.info("Splitting dataset without data leakage...")
    np.random.seed(random_state)

    # Determine grouping column
    group_col = None
    if "lesion_id" in df.columns:
        group_col = "lesion_id"
    elif "patient_id" in df.columns:
        group_col = "patient_id"

    if group_col:
        logger.info("Guaranteed no data leakage: Grouping splits by '%s'.", group_col)
        # Group-aware stratified split
        unique_groups = df[[group_col, "dx"]].drop_duplicates(subset=[group_col])
        groups = unique_groups[group_col].values
        labels = unique_groups["dx"].values

        # Stratified partition on unique lesion groups
        group_to_split: Dict[str, str] = {}
        for cls in np.unique(labels):
            cls_groups = unique_groups[unique_groups["dx"] == cls][group_col].values
            np.random.shuffle(cls_groups)
            n_total = len(cls_groups)
            n_train = int(np.round(n_total * train_ratio))
            n_val = int(np.round(n_total * val_ratio))
            # Ensure at least 1 per split if possible
            if n_total >= 3 and n_train == 0:
                n_train = 1
            if n_total >= 3 and n_val == 0:
                n_val = 1

            train_grp = cls_groups[:n_train]
            val_grp = cls_groups[n_train:n_train + n_val]
            test_grp = cls_groups[n_train + n_val:]

            for g in train_grp:
                group_to_split[g] = "train"
            for g in val_grp:
                group_to_split[g] = "val"
            for g in test_grp:
                group_to_split[g] = "test"

        df["split"] = df[group_col].map(group_to_split)
        # Fill any unassigned with train
        df["split"] = df["split"].fillna("train")
    else:
        logger.warning("No lesion_id or patient_id found. Performing standard stratified random split.")
        df["split"] = "train"
        for cls in df["dx"].unique():
            cls_idx = df[df["dx"] == cls].index.values
            np.random.shuffle(cls_idx)
            n = len(cls_idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            df.loc[cls_idx[:n_train], "split"] = "train"
            df.loc[cls_idx[n_train:n_train + n_val], "split"] = "val"
            df.loc[cls_idx[n_train + n_val:], "split"] = "test"

    # Clean existing processed directory if any
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    # Copy / symlink images into split folders: data/processed/<split>/<class>/<img_id>.jpg
    split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for cls in df["dx"].unique():
        for s in ["train", "val", "test"]:
            (output_dir / s / cls).mkdir(parents=True, exist_ok=True)
            split_counts[s][cls] = 0

    copied_total = 0
    missing_images = 0

    for _, row in df.iterrows():
        img_id = str(row["image_id"]).replace(".jpg", "")
        cls = str(row["dx"])
        split = str(row["split"])

        # Locate source image file
        source_candidates = [
            images_dir / f"{img_id}.jpg",
            images_dir / f"{img_id}.jpeg",
            images_dir / f"{img_id}.png",
            images_dir / img_id
        ]
        source_path = next((p for p in source_candidates if p.exists()), None)

        if source_path is None:
            missing_images += 1
            continue

        target_path = output_dir / split / cls / f"{img_id}.jpg"
        shutil.copy2(source_path, target_path)
        split_counts[split][cls] += 1
        copied_total += 1

    summary = {
        "total_metadata_records": len(df),
        "total_images_processed": copied_total,
        "missing_images_skipped": missing_images,
        "split_counts": split_counts,
        "classes": sorted(list(df["dx"].unique())),
        "group_column_used": group_col
    }

    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dataset preparation complete.")
    logger.info("Summary saved to %s", summary_file)
    logger.info("Train set: %d images | Val set: %d images | Test set: %d images",
                sum(split_counts["train"].values()),
                sum(split_counts["val"].values()),
                sum(split_counts["test"].values()))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare HAM10000 skin lesion dataset for CNN training.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Generate a synthetic smoke-test dataset for rapid laptop pipeline verification.")
    parser.add_argument("--samples-per-class", type=int, default=15,
                        help="Number of samples per class in smoke-test mode (default: 15).")
    args = parser.parse_args()

    raw_dir = config.RAW_DATA_DIR
    processed_dir = config.PROCESSED_DATA_DIR

    if args.smoke_test:
        logger.info("Running in SMOKE-TEST mode: creating synthetic miniature dataset...")
        metadata_path, images_dir = create_smoke_test_dataset(raw_dir, args.samples_per_class)
        df = pd.read_csv(metadata_path)
    else:
        is_avail, metadata_path, images_dir = check_raw_dataset(raw_dir)
        if not is_avail:
            print("=" * 70)
            print("DATASET NOT FOUND IN LOCAL REPOSITORY")
            print("=" * 70)
            print(f"Expected metadata: {raw_dir / 'HAM10000_metadata.csv'}")
            print(f"Expected images dir: {raw_dir / 'images/'}")
            print("\nHow to obtain the official HAM10000 dataset:")
            print("1. Download from Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T")
            print("   or ISIC Archive: https://www.isic-archive.com/")
            print("2. Extract metadata CSV to 'data/raw/HAM10000_metadata.csv'")
            print("3. Extract image archives into 'data/raw/images/'")
            print("4. Re-run: python training/prepare_data.py")
            print("\nOr verify the entire training/prediction pipeline right now using:")
            print("   python training/prepare_data.py --smoke-test")
            print("=" * 70)
            sys.exit(1)

        logger.info("Found raw dataset at %s with images in %s", metadata_path, images_dir)
        df = pd.read_csv(metadata_path)

    # Inspect class distribution
    print("\n--- Raw Dataset Class Distribution ---")
    print(df["dx"].value_counts())
    print("-" * 38)

    split_dataset_without_leakage(df, images_dir, processed_dir)


if __name__ == "__main__":
    main()
