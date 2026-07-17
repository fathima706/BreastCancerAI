import sqlite3
import pandas as pd

DATABASE = "database/patients.db"


# ===========================================================
# Create Database
# ===========================================================

def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS predictions(

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

    conn.close()


# ===========================================================
# Save Prediction
# ===========================================================

def save_prediction(

    age,
    tumor_size,
    tumor_stage,
    lymph_nodes,
    histologic_grade,
    er_status,
    pr_status,
    her2_status,
    chemotherapy,
    hormone_therapy,
    prediction,
    confidence

):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

    INSERT INTO predictions(

        age,
        tumor_size,
        tumor_stage,
        lymph_nodes,
        histologic_grade,
        er_status,
        pr_status,
        her2_status,
        chemotherapy,
        hormone_therapy,
        prediction,
        confidence

    )

    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)

    """,(

        age,
        tumor_size,
        tumor_stage,
        lymph_nodes,
        histologic_grade,
        er_status,
        pr_status,
        her2_status,
        chemotherapy,
        hormone_therapy,
        prediction,
        confidence

    ))

    conn.commit()

    conn.close()


# ===========================================================
# Load History
# ===========================================================

def load_history():

    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


# ===========================================================
# Delete All Records
# ===========================================================

def delete_all_predictions():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions"
    )

    conn.commit()

    conn.close()


# ===========================================================
# Statistics
# ===========================================================

def total_predictions():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    total = cursor.fetchone()[0]

    conn.close()

    return total


def average_confidence():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT AVG(confidence) FROM predictions"
    )

    avg = cursor.fetchone()[0]

    conn.close()

    if avg is None:
        return 0

    return round(avg,2)