
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """A registered DermaAI user."""

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    
    email = db.Column(
        db.String(255),
        unique=True,
        nullable=False,
        index=True,
    

    )
    password_hash = db.Column(db.String(255), nullable=False)

    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(30), nullable=True)
    skin_type = db.Column(db.String(30), nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    skincare_products = db.Column(db.Text, nullable=True)
    previous_skin_problems = db.Column(db.Text, nullable=True)
    profile_notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
        nullable=False,
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ScanHistory(db.Model):
    """Store a user's skin-image classification history without the image."""

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False,
        index=True,
    )

    prediction_name = db.Column(db.String(100), nullable=False)
    prediction_code = db.Column(db.String(30), nullable=False)
    confidence_percentage = db.Column(db.Float, nullable=False)
    """Store a user's skin-image classification history."""
    image_filename = db.Column(db.String(255), nullable=True)

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
        nullable=False,
        index=True,
    )
