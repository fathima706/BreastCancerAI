import sqlite3
import pandas as pd


def load_history():

    conn = sqlite3.connect("database/patients.db")

    df = pd.read_sql_query(
        "SELECT * FROM predictions",
        conn
    )

    conn.close()

    return df