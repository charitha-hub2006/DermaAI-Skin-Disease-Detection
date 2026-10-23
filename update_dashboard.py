from pathlib import Path

template_path = Path("templates/dashboard.html")
original = template_path.read_text()

start_marker = '    <div class="dashboard-grid">'
end_marker = '    <div class="notice">'

if start_marker not in original or end_marker not in original:
    raise SystemExit(
        "Could not find the expected dashboard sections. "
        "No changes were made."
    )

replacement = """    <div class="summary-grid">
        <article class="summary-card">
            <p class="summary-label">Total scans</p>
            <h2>{{ total_scans }}</h2>
            <p>Scans saved to your account</p>
        </article>

        <article class="summary-card">
            <p class="summary-label">Latest prediction</p>
            <h2>
                {% if latest_scan %}
                    {{ latest_scan.prediction_name }}
                {% else %}
                    No scans yet
                {% endif %}
            </h2>
            {% if latest_scan %}
                <p>{{ "%.1f"|format(latest_scan.confidence_percentage) }}% confidence</p>
            {% else %}
                <p>Your first scan will appear here.</p>
            {% endif %}
        </article>

        <article class="summary-card">
            <p class="summary-label">Recent activity</p>
            <h2>{{ recent_scans|length }}</h2>
            <p>Showing up to five recent scans</p>
        </article>
    </div>

    <section class="recent-section">
        <div class="section-heading">
            <h2>Recent scan results</h2>
            <a class="action-link" href="{{ url_for('auth.history') }}">View full history →</a>
        </div>

        {% if recent_scans %}
            <div class="table-wrapper">
                <table class="recent-table">
                    <thead>
                        <tr>
                            <th>Prediction</th>
                            <th>Model code</th>
                            <th>Confidence</th>
                            <th>Date and time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for scan in recent_scans %}
                            <tr>
                                <td>{{ scan.prediction_name }}</td>
                                <td>{{ scan.prediction_code }}</td>
                                <td>{{ "%.1f"|format(scan.confidence_percentage) }}%</td>
                                <td>{{ scan.created_at.strftime("%d %b %Y, %I:%M %p") if scan.created_at else "Not available" }}</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        {% else %}
            <div class="empty-state">
                <h3>No scans yet</h3>
                <p>Your saved scan results will appear here after your first analysis.</p>
                <a class="action-link" href="{{ url_for('predict_page') }}">Start your first analysis →</a>
            </div>
        {% endif %}
    </section>

    <div class="dashboard-grid">
        <article class="dashboard-card">
            <span class="card-icon">✦</span>
            <h2>Skin analysis</h2>
            <p>Upload a skin image to explore an experimental CNN classification.</p>
            <a class="action-link" href="{{ url_for('predict_page') }}">
                Start an analysis →
            </a>
        </article>

        <article class="dashboard-card">
            <span class="card-icon">♡</span>
            <h2>Skin-health guides</h2>
            <p>Learn general information about skin conditions and prevention.</p>
            <a class="action-link" href="{{ url_for('about') }}">
                Explore information →
            </a>
        </article>

        <article class="dashboard-card">
            <span class="card-icon">◎</span>
            <h2>Your account</h2>
            <p><strong>Name:</strong> {{ user.name }}</p>
            <p><strong>Email:</strong> {{ user.email }}</p>
            <p class="small-note">Your account was created for your personal use.</p>
            <a class="action-link" href="{{ url_for('auth.profile') }}">View profile →</a>
        </article>
    </div>

"""

updated = original.replace(start_marker, replacement + end_marker, 1)

style_marker = ".dashboard-grid {"
new_styles = """
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 18px;
    margin: 25px 0;
}
.summary-card {
    padding: 22px;
    border: 1px solid #e2e8e2;
    border-radius: 16px;
    background: #fff;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.04);
    overflow-wrap: anywhere;
}
.summary-label {
    color: #547a68;
    font-weight: 700;
    margin: 0 0 10px;
}
.summary-card h2 {
    font-size: 25px;
    margin: 8px 0;
}
.summary-card p:last-child { margin-bottom: 0; }
.recent-section {
    margin: 30px 0;
    padding: 22px;
    border: 1px solid #e2e8e2;
    border-radius: 16px;
    background: #fff;
}
.section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}
.section-heading h2 { margin: 0; }
.table-wrapper { width: 100%; overflow-x: auto; }
.recent-table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
}
.recent-table th, .recent-table td {
    padding: 12px;
    border-bottom: 1px solid #e8ece8;
    white-space: nowrap;
}
.recent-table th {
    color: #426b58;
    background: #f6f8f5;
}
.empty-state {
    padding: 24px 10px;
    text-align: center;
    background: #f6f8f5;
    border-radius: 12px;
}
.empty-state p { line-height: 1.6; }
"""

if style_marker not in updated:
    raise SystemExit("Could not find the CSS insertion point. No changes were made.")

updated = updated.replace(style_marker, new_styles + "\n" + style_marker, 1)
template_path.write_text(updated)
print("Dashboard template updated successfully.")
