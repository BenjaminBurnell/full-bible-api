import json
import os
import sqlite3

DB_PATH = os.environ.get("METADATA_DB", "/var/data/metadata.db")
JSON_PATH = os.environ.get("PERSON_DESC_JSON", "person_descriptions.json")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Add column if missing
try:
    cur.execute("ALTER TABLE people ADD COLUMN brief_description TEXT")
except sqlite3.OperationalError:
    pass  # column already exists

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# people table uses id as the primary key (your person_id values match it)
cur.executemany(
    "UPDATE people SET brief_description = ? WHERE id = ?",
    [(d["description"], d["person_id"]) for d in data if d.get("person_id")],
)

conn.commit()
conn.close()
print("✅ Updated people.brief_description")
