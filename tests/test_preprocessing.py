"""
Unit tests for image preprocessing and validation utilities.
"""

import io
import pytest
import numpy as np
from PIL import Image

import config
from utils.preprocessing import (
    allowed_file,
    validate_image_stream,
    preprocess_image,
    image_to_base64_preview
)


def create_dummy_image_bytes(format="JPEG", size=(100, 100), color=(255, 0, 0)) -> bytes:
    """Helper to generate in-memory valid image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestFileValidation:
    """Tests for file extension and type checking."""

    def test_allowed_file_valid_extensions(self):
        assert allowed_file("lesion.jpg") is True
        assert allowed_file("lesion.jpeg") is True
        assert allowed_file("lesion.png") is True
        assert allowed_file("LESION.JPG") is True
        assert allowed_file("path/to/my.lesion.PNG") is True

    def test_allowed_file_invalid_extensions(self):
        assert allowed_file("virus.exe") is False
        assert allowed_file("script.py") is False
        assert allowed_file("data.csv") is False
        assert allowed_file("image.gif") is False
        assert allowed_file("no_extension") is False
        assert allowed_file("") is False
        assert allowed_file(None) is False


class TestImageStreamValidation:
    """Tests for image stream integrity verification."""

    def test_valid_jpeg_stream(self):
        data = create_dummy_image_bytes("JPEG")
        img = validate_image_stream(data)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_valid_png_stream(self):
        data = create_dummy_image_bytes("PNG")
        img = validate_image_stream(data)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_corrupt_image_bytes(self):
        corrupt_bytes = b"NotARealImageHeader123456789"
        with pytest.raises(ValueError, match="not a valid or readable image"):
            validate_image_stream(corrupt_bytes)

    def test_empty_image_bytes(self):
        with pytest.raises(ValueError):
            validate_image_stream(b"")


class TestPreprocessingPipeline:
    """Tests for output dimensions, normalization, and data types."""

    def test_output_tensor_shape_and_dtype(self):
        data = create_dummy_image_bytes("JPEG", size=(250, 300))
        tensor = preprocess_image(data)

        # Expected shape is (1, 128, 128, 3)
        assert tensor.shape == (1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_CHANNELS)
        assert tensor.dtype == np.float32

    def test_pixel_normalization_range(self):
        # Create image with specific max/min colors
        data = create_dummy_image_bytes("JPEG", size=(50, 50), color=(255, 255, 255))
        tensor = preprocess_image(data)

        # Values should be scaled to [0.0, 1.0]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_pil_image_input(self):
        pil_img = Image.new("RGBA", (80, 80), (100, 150, 200, 255))
        tensor = preprocess_image(pil_img)
        assert tensor.shape == (1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_CHANNELS)
        assert tensor.dtype == np.float32

    def test_image_to_base64_preview(self):
        data = create_dummy_image_bytes("JPEG")
        data_uri = image_to_base64_preview(data, "image/jpeg")
        assert data_uri.startswith("data:image/jpeg;base64,")
        assert len(data_uri) > 50
