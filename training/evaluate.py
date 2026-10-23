"""
Model evaluation script for Skin Disease Detection Using CNN.
Evaluates the trained model on the held-out test set, generating precision, recall,
F1-score, confusion matrix heatmap, and comprehensive evaluation artifacts.
"""

import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_cnn")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """
    Renders and saves a high-resolution confusion matrix heatmap.
    """
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Skin Disease CNN", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Annotate numbers in cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color=color, fontsize=11, fontweight="medium")

    plt.ylabel("True Class Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info("Confusion matrix saved to %s", save_path)


def evaluate_test_set(test_dir: Path, model_path: Path):
    """
    Evaluates test set against trained model and saves reports.
    """
    if not model_path.exists():
        logger.error("Trained model not found at %s. Please run training first.", model_path)
        sys.exit(1)

    if not test_dir.exists() or not any(test_dir.glob("*/*.jpg")):
        logger.error("Held-out test set not found in %s.", test_dir)
        sys.exit(1)

    import tensorflow as tf

    logger.info("Loading trained CNN model from %s...", model_path)
    model = tf.keras.models.load_model(str(model_path))

    logger.info("Loading test dataset from %s...", test_dir)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    class_names = test_ds.class_names
    logger.info("Evaluating on classes: %s", class_names)

    # Normalize inputs matching preprocessing
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    test_ds = test_ds.map(lambda x, y: (rescale_layer(x), y))

    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_batch = np.argmax(labels.numpy(), axis=1)
        y_pred_batch = np.argmax(preds, axis=1)
        y_true_list.extend(y_true_batch)
        y_pred_list.extend(y_pred_batch)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Metrics calculation
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 65)
    print("HELD-OUT TEST SET EVALUATION REPORT")
    print("=" * 65)
    print(report_text)
    print("-" * 65)
    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Avg F1:      {macro_f1:.4f}")
    print(f"Weighted Avg F1:   {weighted_f1:.4f}")
    print("=" * 65 + "\n")

    # Save artifacts
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved classification report to %s", config.EVALUATION_REPORT_PATH)

    summary_metrics = {
        "accuracy": accuracy,
        "accuracy_percentage": round(accuracy * 100.0, 2),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_samples_count": len(y_true),
        "classes": class_names
    }
    with open(config.METRICS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    logger.info("Saved metrics summary to %s", config.METRICS_SUMMARY_PATH)

    plot_confusion_matrix(cm, class_names, config.CONFUSION_MATRIX_PATH)

    return summary_metrics


def main():
    test_dir = config.PROCESSED_DATA_DIR / "test"
    model_path = config.MODEL_PATH
    evaluate_test_set(test_dir, model_path)


if __name__ == "__main__":
    main()
