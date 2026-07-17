import os
import sqlite3
import pandas as pd

DB_PATH = "database/patients.db"


def load_history():
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        age REAL,

        tumor_size REAL,

        tumor_stage REAL,

        lymph_nodes REAL,

        histologic_grade REAL,

        er_status TEXT,

        pr_status TEXT,

        her2_status TEXT,

        chemotherapy TEXT,

        hormone_therapy TEXT,

        prediction TEXT,

        confidence REAL

    )
    """)

    conn.commit()

    try:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )
    except Exception:
        df = pd.DataFrame()

    conn.close()

    return df