"""
Integration and functional tests for the Skin Disease Detection Flask web application.
"""

import io
import pytest
from PIL import Image

import config
from app import create_app
from utils.prediction import get_model_manager


@pytest.fixture
def client():
    """Create Flask test client with test configuration."""
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def create_test_image(format="JPEG", size=(128, 128), color=(200, 100, 80)) -> io.BytesIO:
    """Helper to generate a valid test image file stream."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf


class TestPublicRoutes:
    """Verify standard GET endpoints return HTTP 200 and expected markup."""

    def test_home_page_response(self, client):
        response = client.get("/")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Skin Disease Detection" in html
        assert "DermaAI CNN" in html
        assert "ACADEMIC & EDUCATIONAL USE ONLY" in html

    def test_about_page_response(self, client):
        response = client.get("/about")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "About the Project" in html
        assert "Viva Voce Preparation" in html
        assert "Data Leakage" in html or "data leakage" in html

    def test_model_info_page_response(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Model Architecture" in html
        assert "Conv Block" in html
        assert "Softmax" in html

    def test_predict_page_response(self, client):
        response = client.get("/predict")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Analyze Skin Lesion" in html
        assert "dropZone" in html


class TestHealthEndpoint:
    """Verify system diagnostics and health check endpoint."""

    def test_api_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code in (200, 503)
        data = response.get_json()
        assert "model_loaded" in data
        assert "supported_classes" in data
        assert "input_shape" in data


class TestUploadValidationAndErrors:
    """Verify input validation, error handling, and rejection of malformed inputs."""

    def test_missing_image_upload_form(self, client):
        # Submitting without 'image' file field
        response = client.post("/predict", data={}, follow_redirects=True)
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "No file part detected" in html

    def test_empty_filename_upload_form(self, client):
        response = client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "No image file was selected" in html

    def test_invalid_file_extension_form(self, client):
        response = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"Malicious shell script"), "exploit.sh")},
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Invalid file extension" in html

    def test_corrupt_image_upload_form(self, client):
        # File has .jpg extension but corrupt contents
        response = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"CorruptedHeaderStringNotAJPEG"), "corrupt.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Invalid or corrupted image" in html

    def test_api_missing_file(self, client):
        response = client.post("/api/predict", data={})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "No 'image' file field" in data["error"]

    def test_api_invalid_extension(self, client):
        response = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"some content"), "notes.txt")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "Invalid file type" in data["error"]

    def test_api_corrupt_image(self, client):
        response = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"RandomByteStream123"), "broken.png")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "not a valid or readable image" in data["error"]

    def test_oversized_upload_limit(self, client):
        # Exceed MAX_CONTENT_LENGTH (16MB)
        large_bytes = b"0" * (17 * 1024 * 1024)
        response = client.post(
            "/predict",
            data={"image": (io.BytesIO(large_bytes), "oversized.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True
        )
        # Should be rejected with 413 or handled redirect
        assert response.status_code in (200, 413)


class TestPredictionWorkflow:
    """Verify successful end-to-end inference and prediction reporting."""

    def test_successful_api_prediction(self, client):
        mgr = get_model_manager()
        if not mgr.is_loaded:
            pytest.skip("Model not loaded yet; skipping live prediction test.")

        img_buf = create_test_image("JPEG")
        response = client.post(
            "/api/predict",
            data={"image": (img_buf, "sample_lesion.jpg")},
            content_type="multipart/form-data"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "prediction" in data
        assert "class_name" in data["prediction"]
        assert "confidence_score" in data["prediction"]
        assert 0.0 <= data["prediction"]["confidence_score"] <= 1.0
        assert len(data["class_probabilities"]) == len(config.DEFAULT_CLASSES)
        assert "disclaimer" in data

    def test_successful_form_prediction_results_page(self, client):
        mgr = get_model_manager()
        if not mgr.is_loaded:
            pytest.skip("Model not loaded yet; skipping live prediction test.")

        img_buf = create_test_image("PNG")
        response = client.post(
            "/predict",
            data={"image": (img_buf, "sample_lesion.png")},
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Analysis Result" in html
        assert "Primary Classification" in html
        assert "Model Statistical Confidence" in html
        assert "Differential Probability Breakdown" in html
        assert "Important Clinical Recommendation" in html

    def test_prediction_error_handling_when_model_missing(self, monkeypatch):
        # Simulate missing model condition
        mgr = get_model_manager()
        original_state = mgr.is_loaded
        mgr.is_loaded = False
        mgr.load_error = "Mock: Model file missing"

        try:
            test_app = create_app({"TESTING": True})
            with test_app.test_client() as test_client:
                img_buf = create_test_image("JPEG")
                response = test_client.post(
                    "/api/predict",
                    data={"image": (img_buf, "test.jpg")},
                    content_type="multipart/form-data"
                )
                assert response.status_code == 503
                data = response.get_json()
                assert data["success"] is False
                assert "Model is not loaded" in data["error"]
        finally:
            mgr.is_loaded = original_state
