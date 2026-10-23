"""
Model management and inference service for Skin Disease Detection.
Implements lazy-loading singleton pattern to ensure the model is loaded once at runtime.
"""

import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np

import config

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for loading and serving predictions with the trained CNN model.
    """
    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.model = None
        self.class_indices: Dict[int, str] = {}
        self.class_metadata: Dict[str, Dict[str, str]] = config.DEFAULT_CLASSES
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.load_model()
        self._initialized = True

    def load_model(self) -> bool:
        """
        Attempts to load the .keras model and class indices from disk.
        Does not fabricate random models or crash if model is missing.
        """
        if not config.MODEL_PATH.exists():
            self.is_loaded = False
            self.load_error = (
                f"Model file not found at '{config.MODEL_PATH}'. "
                "Please run 'python training/train.py' (or 'python training/prepare_data.py --smoke-test && python training/train.py --smoke-test') "
                "to train and save the model before performing inference."
            )
            logger.warning(self.load_error)
            return False

        try:
            # Import tensorflow inside method so other lightweight checks don't block
            import tensorflow as tf

            # Load model
            logger.info("Loading trained CNN model from %s...", config.MODEL_PATH)
            self.model = tf.keras.models.load_model(str(config.MODEL_PATH))

            # Load class mapping
            if config.CLASS_INDICES_PATH.exists():
                with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                    raw_indices = json.load(f)
                    # Convert string keys back to int if saved as dict {"0": "akiec", ...}
                    self.class_indices = {int(k): v for k, v in raw_indices.items()}
            else:
                sorted_keys = sorted(config.DEFAULT_CLASSES.keys())
                self.class_indices = {i: key for i, key in enumerate(sorted_keys)}

            # Warmup inference to initialize computation graph
            dummy_input = np.zeros((1, *config.INPUT_SHAPE), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)

            self.is_loaded = True
            self.load_error = None
            logger.info("Model loaded successfully. Classes: %s", self.class_indices)
            return True

        except Exception as exc:  # pylint: disable=broad-except
            self.model = None
            self.is_loaded = False
            self.load_error = f"Failed to load trained model: {str(exc)}"
            logger.error(self.load_error, exc_info=True)
            return False

    def predict(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on preprocessed image batch of shape (1, H, W, 3).

        Returns:
            Dictionary containing predicted label, confidence, all class probabilities,
            clinical details, and disclaimer.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If input tensor shape is invalid.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        if preprocessed_image.shape != (1, *config.INPUT_SHAPE):
            raise ValueError(
                f"Expected input shape (1, {config.IMAGE_WIDTH}, {config.IMAGE_HEIGHT}, {config.IMAGE_CHANNELS}), "
                f"but received {preprocessed_image.shape}"
            )

        # Run real inference
        raw_predictions = self.model.predict(preprocessed_image, verbose=0)
        probabilities = raw_predictions[0]

        top_index = int(np.argmax(probabilities))
        top_prob = float(probabilities[top_index])
        top_code = self.class_indices.get(top_index, f"class_{top_index}")

        # Assemble full class probabilities list sorted descending
        breakdown: List[Dict[str, Any]] = []
        for idx, prob in enumerate(probabilities):
            code = self.class_indices.get(idx, f"class_{idx}")
            meta = self.class_metadata.get(code, {
                "name": code.upper(),
                "category": "Unknown",
                "description": "Information not available."
            })
            percentage = round(float(prob) * 100.0, 2)
            breakdown.append({
                "index": idx,
                "code": code,
                "name": meta["name"],
                "category": meta["category"],
                "probability": float(prob),
                "percentage": percentage
            })

        breakdown.sort(key=lambda x: x["probability"], reverse=True)

        meta_top = self.class_metadata.get(top_code, {
            "name": top_code.upper(),
            "category": "Unknown",
            "description": "Consult a healthcare provider for clinical evaluation."
        })

        return {
            "success": True,
            "prediction": {
                "class_code": top_code,
                "class_name": meta_top["name"],
                "category": meta_top["category"],
                "description": meta_top["description"],
                "confidence_score": top_prob,
                "confidence_percentage": round(top_prob * 100.0, 2),
                "is_confident": top_prob >= 0.50
            },
            "class_probabilities": breakdown,
            "disclaimer": (
                "Educational Prototype Notice: This prediction is generated by an experimental "
                "Convolutional Neural Network (CNN) trained for academic demonstration. It is NOT "
                "a medical diagnostic device and must not replace professional clinical evaluation."
            )
        }

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of model."""
        return {
            "model_loaded": self.is_loaded,
            "model_path": str(config.MODEL_PATH),
            "model_exists_on_disk": config.MODEL_PATH.exists(),
            "load_error": self.load_error,
            "supported_classes": list(self.class_indices.values()),
            "input_shape": config.INPUT_SHAPE
        }


def get_model_manager() -> ModelManager:
    """Helper factory function to retrieve the singleton model manager."""
    return ModelManager()
