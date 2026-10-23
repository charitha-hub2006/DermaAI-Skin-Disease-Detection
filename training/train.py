"""
CNN Model Architecture and Training Pipeline for Skin Disease Detection.
Trains a Convolutional Neural Network on skin lesion images, saves the best model,
class mappings, and generates training history loss/accuracy visualization.
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_cnn")


def build_skin_disease_cnn(
    input_shape=config.INPUT_SHAPE,
    num_classes: int = len(config.DEFAULT_CLASSES)
):
    """
    Constructs a robust, lightweight Convolutional Neural Network (CNN) tailored for
    dermatoscopic skin lesion image classification on local laptops.

    Architecture:
      - Data Augmentation (active during training only):
          RandomFlip, RandomRotation, RandomZoom
      - Block 1: Conv2D(32) -> BatchNormalization -> Conv2D(32) -> MaxPooling -> Dropout
      - Block 2: Conv2D(64) -> BatchNormalization -> Conv2D(64) -> MaxPooling -> Dropout
      - Block 3: Conv2D(128) -> BatchNormalization -> Conv2D(128) -> MaxPooling -> Dropout
      - Block 4: Conv2D(256) -> BatchNormalization -> GlobalAveragePooling2D -> Dropout
      - Dense Head: Dense(128, ReLU) -> Dropout(0.4) -> Dense(num_classes, Softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Augmentation layers (executed only in training mode)
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.15, fill_mode="reflect", name="aug_rotation")(x)
    x = layers.RandomZoom(0.1, fill_mode="reflect", name="aug_zoom")(x)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # Conv Block 4 & Global Pooling
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # Fully Connected Classification Head
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.4, name="drop_dense")(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SkinDisease_CNN")
    return model


def plot_and_save_history(history, save_path: Path):
    """Save accuracy and loss history curves to artifacts directory."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", linewidth=2)
    plt.title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Validation Loss", color="#f59e0b", linewidth=2)
    plt.title("Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info("Saved training curves to %s", save_path)


def train_model(
    train_dir: Path,
    val_dir: Path,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE
):
    """
    Main training execution function.
    Loads data generators, builds CNN, compiles with Adam, trains with early stopping,
    and serializes the trained model to models/skin_disease_cnn.keras.
    """
    import tensorflow as tf

    logger.info("Setting up datasets from %s and %s...", train_dir, val_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    logger.info("Detected %d classes: %s", num_classes, class_names)

    # Class to index mapping
    class_indices = {i: name for i, name in enumerate(class_names)}
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)
    logger.info("Saved class indices mapping to %s", config.CLASS_INDICES_PATH)

    val_ds = None
    if val_dir.exists() and any(val_dir.glob("*/*.jpg")):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(val_dir),
            labels="inferred",
            label_mode="categorical",
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            shuffle=False
        )

    # Normalize inputs from [0, 255] to [0.0, 1.0] matching utils/preprocessing.py
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y))
    if val_ds:
        val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y))

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    if val_ds:
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    logger.info("Building CNN model architecture...")
    model = build_skin_disease_cnn(input_shape=config.INPUT_SHAPE, num_classes=num_classes)
    model.summary(print_fn=logger.info)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds else "loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_accuracy" if val_ds else "accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info("Commencing CNN training for %d epochs...", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure best model is saved
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(config.MODEL_PATH))
    logger.info("Trained model successfully saved to %s", config.MODEL_PATH)

    # Save training configuration metadata
    training_config = {
        "model_architecture": "SkinDisease_CNN",
        "input_shape": list(config.INPUT_SHAPE),
        "classes": class_names,
        "num_classes": num_classes,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": batch_size,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]) if val_ds else None,
        "final_val_loss": float(history.history["val_loss"][-1]) if val_ds else None
    }
    with open(config.TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
    logger.info("Saved training configuration to %s", config.TRAINING_CONFIG_PATH)

    # Plot & save curves
    plot_and_save_history(history, config.TRAINING_HISTORY_PATH)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train CNN model on skin lesion dataset.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {config.DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 2-epoch run for pipeline verification.")
    args = parser.parse_args()

    train_dir = config.PROCESSED_DATA_DIR / "train"
    val_dir = config.PROCESSED_DATA_DIR / "val"

    if not train_dir.exists() or not any(train_dir.glob("*/*.jpg")):
        logger.warning("Processed training data not found in %s.", train_dir)
        logger.info("Preparing data automatically now...")
        from training.prepare_data import split_dataset_without_leakage, create_smoke_test_dataset, check_raw_dataset
        import pandas as pd

        is_avail, metadata_path, images_dir = check_raw_dataset(config.RAW_DATA_DIR)
        if not is_avail or args.smoke_test:
            logger.info("Generating synthetic smoke-test dataset for training...")
            metadata_path, images_dir = create_smoke_test_dataset(config.RAW_DATA_DIR, num_samples_per_class=12)

        df = pd.read_csv(metadata_path)
        split_dataset_without_leakage(df, images_dir, config.PROCESSED_DATA_DIR)

    epochs = 2 if args.smoke_test else args.epochs
    train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
