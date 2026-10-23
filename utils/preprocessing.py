"""
Image preprocessing and validation utilities for Skin Disease Detection.
Ensures identical preprocessing steps between training and Flask inference pipelines.
"""

import io
import base64
from typing import Tuple, Union, Optional
from PIL import Image, UnidentifiedImageError
import numpy as np
from werkzeug.utils import secure_filename

import config


def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed.
    Only allows jpg, jpeg, and png.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def validate_image_stream(file_stream: Union[io.BytesIO, bytes]) -> Image.Image:
    """
    Validate that an image stream contains a non-corrupt, valid image.
    Converts any color space (RGBA, grayscale, CMYK) to standard 3-channel RGB.

    Raises:
        ValueError: If file is empty, corrupt, or unreadable.
    """
    if isinstance(file_stream, bytes):
        file_stream = io.BytesIO(file_stream)

    try:
        file_stream.seek(0)
        img = Image.open(file_stream)
        img.verify()  # Verify integrity
        # Re-open after verify because verify can alter stream state
        file_stream.seek(0)
        img = Image.open(file_stream)
        # Convert to RGB to ensure 3 channels
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Uploaded file is not a valid or readable image: {str(exc)}") from exc


def preprocess_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = config.IMAGE_SIZE
) -> np.ndarray:
    """
    Preprocess an input image for model input.
    Pipeline:
      1. Load/validate image
      2. Convert to RGB (3 channels)
      3. Resize to target_size (default: 128x128)
      4. Convert to float32 NumPy array
      5. Normalize pixel values to [0.0, 1.0] by dividing by 255.0
      6. Expand dimensions to (1, height, width, 3)

    Returns:
        np.ndarray of shape (1, height, width, 3), dtype float32, range [0.0, 1.0].
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, str):
        with open(image_input, "rb") as f:
            img = validate_image_stream(f.read())
    else:
        img = validate_image_stream(image_input)

    # Resize with high-quality resampling
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to array & normalize
    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0

    # Ensure shape is (1, H, W, 3)
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def image_to_base64_preview(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    Convert image bytes into a base64 data URI string for HTML rendering.
    Avoids saving images permanently to disk.
    """
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
