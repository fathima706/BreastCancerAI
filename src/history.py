import os
import sqlite3
import pandas as pd


DB_PATH = "database/patients.db"


def load_history():
    # Create database folder if it doesn't exist
    os.makedirs("database", exist_ok=True)

    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create predictions table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        prediction TEXT,
        probability REAL,
        timestamp TEXT
    )
    """)

    conn.commit()

    # Read history
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    return df