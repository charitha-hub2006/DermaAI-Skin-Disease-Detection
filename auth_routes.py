from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from sqlalchemy import select
from models_db import db, User,ScanHistory

auth = Blueprint("auth", __name__)


@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("auth.dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("auth.signup"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return redirect(url_for("auth.signup"))

        existing_user = db.session.execute(
            select(User).where(User.email == email)
        ).scalar_one_or_none()

        if existing_user:
            flash("An account with this email already exists.", "error")
            return redirect(url_for("auth.signup"))

        user = User(name=name, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Your account was created successfully.", "success")
        return redirect(url_for("auth.dashboard"))

    return render_template("signup.html")


@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("auth.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = db.session.execute(
            select(User).where(User.email == email)
        ).scalar_one_or_none()

        if user and user.check_password(password):
            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("auth.dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@auth.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have logged out.", "success")
    return redirect(url_for("index"))


@auth.route("/dashboard")
@login_required
def dashboard():
    user_scans = ScanHistory.query.filter_by(
        user_id=current_user.id
    )

    total_scans = user_scans.count()

    recent_scans = (
        user_scans
        .order_by(ScanHistory.created_at.desc(), ScanHistory.id.desc())
        .limit(5)
        .all()
    )

    latest_scan = recent_scans[0] if recent_scans else None

    return render_template(
        "dashboard.html",
        user=current_user,
        total_scans=total_scans,
        recent_scans=recent_scans,
        latest_scan=latest_scan,
    )


@auth.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        age_text = request.form.get("age", "").strip()

        if age_text:
            try:
                age = int(age_text)
                if not 1 <= age <= 120:
                    raise ValueError
            except ValueError:
                flash("Please enter a valid age between 1 and 120.", "error")
                return redirect(url_for("auth.profile"))
        else:
            age = None

        current_user.age = age
        current_user.gender = request.form.get("gender", "").strip() or None
        current_user.skin_type = request.form.get("skin_type", "").strip() or None
        current_user.allergies = request.form.get("allergies", "").strip() or None
        current_user.skincare_products = request.form.get("skincare_products", "").strip() or None
        current_user.previous_skin_problems = request.form.get("previous_skin_problems", "").strip() or None
        current_user.profile_notes = request.form.get("profile_notes", "").strip() or None

        db.session.commit()
        flash("Your skin-care profile was saved successfully.", "success")
        return redirect(url_for("auth.profile"))

    return render_template("profile.html", user=current_user)


@auth.route("/settings")
@login_required
def settings():
    return render_template("settings.html", user=current_user)


@auth.route("/history/<int:scan_id>")
@login_required
def history_detail(scan_id):
    scan = db.session.execute(
        db.select(ScanHistory).where(
            ScanHistory.id == scan_id,
            ScanHistory.user_id == current_user.id,
        )
    ).scalar_one_or_none()

    if scan is None:
        flash("Scan record not found.", "error")
        return redirect(url_for("auth.history"))

    return render_template(
        "history_detail.html",
        user=current_user,
        scan=scan,
    )


@auth.route("/history/<int:scan_id>/delete", methods=["POST"])
@login_required
def delete_history(scan_id):
    scan = db.session.execute(
        db.select(ScanHistory).where(
            ScanHistory.id == scan_id,
            ScanHistory.user_id == current_user.id,
        )
    ).scalar_one_or_none()

    if scan is None:
        flash("Scan record not found or you do not have permission to delete it.", "error")
        return redirect(url_for("auth.history"))

    db.session.delete(scan)
    db.session.commit()

    flash("Scan record deleted successfully.", "success")
    return redirect(url_for("auth.history"))


@auth.route("/history")
@login_required
def history():
   
    scans = db.session.execute(
        db.select(ScanHistory)
        .where(ScanHistory.user_id == current_user.id)
        .order_by(ScanHistory.created_at.desc())
    ).scalars().all()

    return render_template(
        "history.html",
        user=current_user,
        scans=scans,
    )
