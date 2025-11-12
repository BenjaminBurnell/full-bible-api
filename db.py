
import sqlite3
import os

DB_PATH = os.environ.get("INTERLINEAR_DB", "interlinear.sqlite3")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_code TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    verse INTEGER NOT NULL,
    token_index INTEGER NOT NULL,
    surface TEXT NOT NULL,
    lemma TEXT,
    translit TEXT,
    gloss TEXT,
    morph TEXT,
    strong TEXT
);
CREATE INDEX IF NOT EXISTS idx_ref ON tokens(book_code, chapter, verse);
"""

def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Initialized DB at", DB_PATH)
