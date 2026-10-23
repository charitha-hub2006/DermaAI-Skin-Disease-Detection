
"""
Flask application for DermaAI skin disease classification.
Preserves the existing CNN prediction and information routes.
"""

import os
import sys
import json
import logging
import uuid
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_from_directory,
    abort,
)
from werkzeug.utils import secure_filename
from flask_login import LoginManager, current_user, login_required

# Ensure project root is available before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models_db import db, User, ScanHistory
from utils.preprocessing import (
    allowed_file,
    validate_image_stream,
    preprocess_image,
    image_to_base64_preview,
)
from utils.prediction import get_model_manager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("app")


def create_app(test_config=None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)

    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "sqlite:///" + str(config.BASE_DIR / "dermaai.db")
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Apply testing overrides before initializing extensions.
    if test_config:
        app.config.update(test_config)

    db.init_app(app)

    from auth_routes import auth
    from doctor_routes import doctors
    from support_routes import support

    app.register_blueprint(auth)
    app.register_blueprint(doctors)
    app.register_blueprint(support)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access your account."

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    # Create database tables defined in models_db.py.
    with app.app_context():
        db.create_all()

    # Initialize the existing CNN model manager.
    model_mgr = get_model_manager()

    @app.route("/", methods=["GET"])
    def index():
        """Render the home page."""
        return render_template("index.html")

    @app.route("/history-image/<int:scan_id>", methods=["GET"])
    @login_required
    def scan_image(scan_id):
        """Serve a saved scan image only to its owner."""
        scan = db.session.execute(
            db.select(ScanHistory).where(
                ScanHistory.id == scan_id,
                ScanHistory.user_id == current_user.id,
            )
        ).scalar_one_or_none()

        if scan is None or not scan.image_filename:
            abort(404)

        # Only serve image files stored directly in the upload directory.
        safe_name = secure_filename(scan.image_filename)
        if not safe_name or safe_name != scan.image_filename:
            abort(404)

        image_path = config.UPLOAD_FOLDER / safe_name

        if not image_path.is_file():
            abort(404)

        return send_from_directory(
            str(config.UPLOAD_FOLDER),
            safe_name,
        )

    @app.route("/predict", methods=["GET"])
    def predict_page():
        """Render the image upload page."""
        status = model_mgr.get_status()
        return render_template("predict.html", model_status=status)

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        """Validate an uploaded image and display the CNN prediction."""

        if "image" not in request.files:
            flash("No file part detected in request.", "error")
            return redirect(url_for("predict_page"))

        file = request.files["image"]

        if file.filename == "":
            flash("No image file was selected.", "error")
            return redirect(url_for("predict_page"))

        if not allowed_file(file.filename):
            flash(
                "Invalid file extension. Please upload a JPG, JPEG, or PNG image.",
                "error",
            )
            return redirect(url_for("predict_page"))

        try:
            image_bytes = file.read()

            if not image_bytes:
                flash("Uploaded file is empty.", "error")
                return redirect(url_for("predict_page"))

            validate_image_stream(image_bytes)
            processed_array = preprocess_image(image_bytes)

            if not model_mgr.is_loaded:
                flash(
                    "The trained CNN model is not loaded. "
                    "Check the model file and training setup.",
                    "warning",
                )
                return redirect(url_for("predict_page"))

            result = model_mgr.predict(processed_array)

            # Save scan history and image for authenticated users only.
            if current_user.is_authenticated:
                extension = Path(file.filename).suffix.lower()
                image_filename = f"{uuid.uuid4().hex}{extension}"
                image_path = config.UPLOAD_FOLDER / image_filename

                image_path.write_bytes(image_bytes)

                try:
                    scan = ScanHistory(
                        user_id=current_user.id,
                        prediction_name=result["prediction"]["class_name"],
                        prediction_code=result["prediction"]["class_code"],
                        confidence_percentage=float(
                            result["prediction"]["confidence_percentage"]
                        ),
                        image_filename=image_filename,
                    )
                    db.session.add(scan)
                    db.session.commit()

                except Exception:
                    db.session.rollback()

                    try:
                        image_path.unlink(missing_ok=True)
                    except OSError:
                        logger.exception(
                            "Could not remove an image after saving failed."
                        )

                    raise

            # Preserve the existing result-page preview.
            mime_type = file.mimetype or "image/jpeg"
            preview_uri = image_to_base64_preview(
                image_bytes,
                mime_type=mime_type,
            )

            return render_template(
                "result.html",
                prediction=result["prediction"],
                class_probabilities=result["class_probabilities"],
                image_preview=preview_uri,
                disclaimer=result["disclaimer"],
            )

        except ValueError as exc:
            logger.warning("Invalid image submission: %s", exc)
            flash(f"Invalid or corrupted image: {exc}", "error")
            return redirect(url_for("predict_page"))

        except RuntimeError as exc:
            logger.error("Prediction error: %s", exc)
            flash(str(exc), "error")
            return redirect(url_for("predict_page"))

        except Exception:
            logger.exception("Unexpected image processing error")
            flash(
                "An unexpected error occurred while analyzing the image.",
                "error",
            )
            return redirect(url_for("predict_page"))

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """Return CNN predictions as JSON."""

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No 'image' file field found in request.",
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename submitted.",
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Allowed: jpg, jpeg, png.",
            }), 400

        try:
            image_bytes = file.read()

            if not image_bytes:
                return jsonify({
                    "success": False,
                    "error": "Uploaded file is empty.",
                }), 400

            validate_image_stream(image_bytes)
            processed_array = preprocess_image(image_bytes)

            if not model_mgr.is_loaded:
                return jsonify({
                    "success": False,
                    "error": "CNN model is not loaded.",
                }), 503

            result = model_mgr.predict(processed_array)
            return jsonify(result), 200

        except ValueError as exc:
            return jsonify({
                "success": False,
                "error": str(exc),
            }), 400

        except RuntimeError as exc:
            return jsonify({
                "success": False,
                "error": str(exc),
            }), 503

        except Exception:
            logger.exception("API prediction error")
            return jsonify({
                "success": False,
                "error": "Internal processing failure.",
            }), 500

    @app.route("/api/health", methods=["GET"])
    def api_health():
        """Report CNN model availability."""
        status = model_mgr.get_status()
        status_code = 200 if status["model_loaded"] else 503

        return jsonify({
            "status": (
                "healthy" if status["model_loaded"] else "degraded"
            ),
            **status,
        }), status_code

    @app.route("/about", methods=["GET"])
    def about():
        """Render the existing about page."""
        return render_template("about.html")

    @app.route("/model-info", methods=["GET"])
    def model_info():
        """Render model information and evaluation metrics."""
        status = model_mgr.get_status()
        evaluation_summary = None

        if config.METRICS_SUMMARY_PATH.exists():
            try:
                with open(
                    config.METRICS_SUMMARY_PATH,
                    "r",
                    encoding="utf-8",
                ) as file:
                    evaluation_summary = json.load(file)
            except (OSError, ValueError):
                logger.exception("Could not read evaluation metrics")

        return render_template(
            "model_info.html",
            model_status=status,
            evaluation_summary=evaluation_summary,
            has_confusion_matrix=config.CONFUSION_MATRIX_PATH.exists(),
            has_training_history=config.TRAINING_HISTORY_PATH.exists(),
        )

    @app.route("/artifacts/<path:filename>", methods=["GET"])
    def artifact_file(filename):
        """Serve generated evaluation artifacts safely."""
        safe_name = secure_filename(filename)

        if not safe_name:
            abort(404)

        artifact_path = config.ARTIFACTS_DIR / safe_name

        if not artifact_path.is_file():
            abort(404)

        return send_from_directory(
            str(config.ARTIFACTS_DIR),
            safe_name,
        )

    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": "File size exceeds 16 MB maximum limit.",
            }), 413

        flash("File is too large. Maximum upload size is 16 MB.", "error")
        return redirect(url_for("predict_page"))

    @app.errorhandler(404)
    def page_not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": "Requested resource not found.",
            }), 404

        return render_template("base.html"), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": "Internal server error.",
            }), 500

        logger.error("Internal server error: %s", error)
        flash("An unexpected server error occurred.", "error")
        return redirect(url_for("index"))

    return app


# Create the default application instance.
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in (
        "true",
        "1",
    )

    logger.info(
        "Starting DermaAI Flask server on port %d...",
        port,
    )
    app.run(host="0.0.0.0", port=port, debug=debug)
