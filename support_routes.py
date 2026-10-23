from flask import Blueprint, render_template

support = Blueprint("support", __name__)


@support.route("/appointments")
def appointments():
    return render_template("appointments.html")


@support.route("/help")
def help_page():
    return render_template("help.html")


@support.route("/contact")
def contact():
    return render_template("contact.html")
