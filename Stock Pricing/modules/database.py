import os
from sqlalchemy import create_engine, text


def create_connection(db_uri):
    engine = create_engine(db_uri)
    return engine


def initialize_database(db_uri):
    db_path = db_uri.split('///')[-1]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = create_engine(db_uri)
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume INTEGER
        )
        """))


def load_data_to_db(data, table_name, engine):
    data.to_sql(table_name, engine, if_exists='replace', index=True)


















