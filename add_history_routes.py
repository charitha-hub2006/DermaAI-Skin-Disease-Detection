from pathlib import Path

path = Path("auth_routes.py")
source = path.read_text()

marker = '@auth.route("/history")\n@login_required\ndef history():'

if marker not in source:
    raise SystemExit(
        "Expected history route not found. No changes were made."
    )

if 'def history_detail(scan_id):' in source or 'def delete_history(scan_id):' in source:
    raise SystemExit(
        "History detail/delete routes may already exist. No changes were made."
    )

new_routes = '''@auth.route("/history/<int:scan_id>")
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


'''

path.write_text(source.replace(marker, new_routes + marker, 1))
print("History details and delete routes added.")
