import os
import sys
from sqlalchemy import create_engine, inspect
from logs.sql import Base, get_storage

def diag():
    url = os.getenv("DATABASE_URL")
    print(f"DATABASE_URL: {url}")
    if not url:
        print("No DATABASE_URL set.")
        return

    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            print("Successfully connected to Postgres.")
        
        print("Creating tables if not exist...")
        Base.metadata.create_all(engine)
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables in DB: {tables}")
        
    except Exception as e:
        print(f"Error during diagnostics: {e}")

if __name__ == "__main__":
    diag()
