import sqlite3

DB_PATH = "metadata.db"
PREFIX = "In this dataset, "

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# SQLite substr() is 1-indexed, so start at len(prefix)+1
start = len(PREFIX) + 1

cur.execute(
    "UPDATE people SET description = substr(description, ?) "
    "WHERE description LIKE ?",
    (start, PREFIX + "%"),
)

conn.commit()
print(f"✅ Updated rows: {cur.rowcount}")
conn.close()