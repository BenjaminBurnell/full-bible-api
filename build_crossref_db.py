import sqlite3
import re

INPUT_FILE = "cross_references.txt"       # your OpenBible file
OUTPUT_DB = "cross_references.db" # final DB used by API

# Normalize the book.code into "Book Chap:Verse"
def normalize_reference(code):
    # Example: "Gen.1.1" → ["Gen", "1", "1"]
    parts = code.split(".")
    if len(parts) != 3:
        return None

    book, chap, verse = parts

    # Clean book name: remove trailing dots like "Gen."
    book = book.strip().replace(".", "")

    # Fix books like "1John" → "1 John"
    m = re.match(r"^([123])(\D.+)$", book)
    if m:
        num, name = m.groups()
        book = f"{num} {name.capitalize()}"

    return f"{book} {chap}:{verse}"

def build_db():
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()

    print("Creating table...")

    cur.execute("DROP TABLE IF EXISTS cross_references")
    cur.execute("""
        CREATE TABLE cross_references (
            verse TEXT NOT NULL,
            cross_ref TEXT NOT NULL,
            votes INTEGER NOT NULL
        )
    """)

    print("Reading input file...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:  # skip header row
                first = False
                continue

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            from_raw, to_raw, votes_raw = parts[:3]

            from_norm = normalize_reference(from_raw)
            to_norm = normalize_reference(to_raw)

            if not from_norm or not to_norm:
                continue

            votes = int(votes_raw)

            cur.execute(
                "INSERT INTO cross_references (verse, cross_ref, votes) VALUES (?, ?, ?)",
                (from_norm, to_norm, votes)
            )

    conn.commit()
    conn.close()
    print("DONE! Database saved as:", OUTPUT_DB)

if __name__ == "__main__":
    build_db()
