"""
Configuration module for Skin Disease Detection Using CNN.
Defines paths, model hyperparameters, image preprocessing specifications,
and security settings for file uploads and Flask runtime.
"""

from pathlib import Path
import os

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_FOLDER = BASE_DIR / "uploads"

# Ensure critical directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, UPLOAD_FOLDER]:
    directory.mkdir(parents=True, exist_ok=True)

# Model artifact paths
MODEL_PATH = MODELS_DIR / "skin_disease_cnn.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"

# Evaluation and metric output paths
EVALUATION_REPORT_PATH = ARTIFACTS_DIR / "classification_report.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.png"
METRICS_SUMMARY_PATH = ARTIFACTS_DIR / "metrics_summary.json"

# Image Preprocessing & Model Hyperparameters
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
LEARNING_RATE = 0.001

# Flask Security & Upload Limits
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-skin-disease-detection-2026")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max payload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Class metadata for HAM10000 dataset standards
DEFAULT_CLASSES = {
    "akiec": {
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "category": "Precancerous / Non-invasive",
        "description": "Common pre-cancerous lesions or in-situ carcinomas caused by chronic sun exposure."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "description": "The most common form of skin cancer, arising from basal cells of the deepest layer of the epidermis."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "category": "Benign",
        "description": "Non-cancerous skin growths including solar lentigines, seborrheic keratoses, and lichen-planus like keratoses."
    },
    "df": {
        "name": "Dermatofibroma",
        "category": "Benign",
        "description": "Benign fibrous nodule commonly found on lower extremities."
    },
    "mel": {
        "name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "description": "Serious form of skin cancer originating in pigment-producing melanocytes, requiring immediate clinical staging."
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "category": "Benign",
        "description": "Common benign moles formed by clusters of melanocytes."
    },
    "vasc": {
        "name": "Vascular Lesions",
        "category": "Benign / Vascular",
        "description": "Lesions of blood vessels including cherry angiomas, angiokeratomas, and pyogenic granulomas."
    }
}
