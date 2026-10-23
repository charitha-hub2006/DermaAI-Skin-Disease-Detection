from pathlib import Path

path = Path("templates/history.html")
content = path.read_text()

old_header = """                  <th>Model confidence</th>
                </tr>"""

new_header = """                  <th>Model confidence</th>
                  <th>Actions</th>
                </tr>"""

old_row = """                  <td>{{ "%.2f"|format(scan.confidence_percentage) }}%</td>
                </tr>"""

new_row = """                  <td>{{ "%.2f"|format(scan.confidence_percentage) }}%</td>
                  <td>
                    <div class="d-flex flex-wrap gap-2">
                      <a class="btn btn-sm btn-outline-primary"
                         href="{{ url_for('auth.history_detail', scan_id=scan.id) }}">
                        View details
                      </a>
                      <form method="POST"
                            action="{{ url_for('auth.delete_history', scan_id=scan.id) }}"
                            onsubmit="return confirm('Delete this scan record permanently? This action cannot be undone.');">
                        <button type="submit" class="btn btn-sm btn-outline-danger">
                          Delete
                        </button>
                      </form>
                    </div>
                  </td>
                </tr>"""

if content.count(old_header) != 1:
    raise SystemExit("STOP: Expected table header not found exactly once. No changes made.")

if content.count(old_row) != 1:
    raise SystemExit("STOP: Expected table row not found exactly once. No changes made.")

content = content.replace(old_header, new_header, 1)
content = content.replace(old_row, new_row, 1)

path.write_text(content)
print("Success: Added View details and Delete actions.")
