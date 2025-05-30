import pyodbc

DB_CONFIG = {
    'server': 'HIEU',
    'database': 'HOSOBENHAN',
    'username': 'sa',
    'password': '123',
    'driver': '{ODBC Driver 17 for SQL Server}'
}

def connect_db():
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']}"
    )
    return pyodbc.connect(conn_str)
