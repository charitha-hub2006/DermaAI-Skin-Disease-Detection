import sqlite3
from pathlib import Path

db_path = Path("dermaai.db")

if not db_path.exists():
    raise SystemExit("Database not found. No changes made.")

columns_to_add = {
    "age": "INTEGER",
    "gender": "VARCHAR(30)",
    "skin_type": "VARCHAR(30)",
    "allergies": "TEXT",
    "skincare_products": "TEXT",
    "previous_skin_problems": "TEXT",
    "profile_notes": "TEXT",
}

connection = sqlite3.connect(db_path)

try:
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(user)").fetchall()
    }

    if not existing_columns:
        raise RuntimeError(
            "The user table was not found. No columns were added."
        )

    for column, data_type in columns_to_add.items():
        if column not in existing_columns:
            connection.execute(
                f'ALTER TABLE user ADD COLUMN "{column}" {data_type}'
            )
            print(f"Added column: {column}")
        else:
            print(f"Already exists: {column}")

    connection.commit()
    print("\nDatabase migration completed successfully.")

except Exception:
    connection.rollback()
    raise

finally:
    connection.close()
