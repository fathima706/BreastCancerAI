import sqlite3


def create_database():
    conn = sqlite3.connect("database/patients.db")

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

    conn = sqlite3.connect("database/patients.db")

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