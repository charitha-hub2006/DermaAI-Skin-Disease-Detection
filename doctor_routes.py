from flask import Blueprint, render_template

doctors = Blueprint("doctors", __name__)


@doctors.route("/doctors")
def directory():
    return render_template("doctors.html")
