# build_people_db.py
import os
import csv
import sqlite3
import argparse

RAW_DIR_DEFAULT = os.path.join("data", "metadata_raw")

# Matches your reference format: "GEN 1:1"
def parse_reference_id(ref_id: str):
    if not ref_id:
        return None
    ref_id = ref_id.strip()
    last_colon = ref_id.rfind(":")
    if last_colon == -1:
        return None

    verse_part = ref_id[last_colon + 1 :].strip()
    try:
        verse = int("".join(ch for ch in verse_part if ch.isdigit()))
    except:
        return None

    remainder = ref_id[:last_colon].strip()
    last_space = remainder.rfind(" ")
    if last_space == -1:
        return None

    chap_str = remainder[last_space + 1 :].strip()
    book_str = remainder[:last_space].strip()

    try:
        chapter = int(chap_str)
    except:
        return None

    # book_str is usually already "GEN" style; keep it uppercase
    book = book_str.upper()
    return (book, chapter, verse)

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS people")
    cur.execute("DROP TABLE IF EXISTS verse_people")

    cur.execute("""
        CREATE TABLE people (
            person_id TEXT PRIMARY KEY,
            person_name TEXT,
            surname TEXT,
            unique_attribute TEXT,
            sex TEXT,
            tribe TEXT,
            person_notes TEXT,
            name_instance INTEGER,
            person_sequence INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE verse_people (
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            person_id TEXT NOT NULL,

            person_verse_id TEXT,
            person_label_id TEXT,
            person_label TEXT,
            person_label_count INTEGER,
            person_verse_sequence INTEGER,
            person_verse_notes TEXT,

            PRIMARY KEY (book, chapter, verse, person_id, person_label_id, person_verse_sequence)
        )
    """)

    cur.execute("CREATE INDEX idx_people_name ON people(person_name)")
    cur.execute("CREATE INDEX idx_vp_ref ON verse_people(book, chapter, verse)")
    cur.execute("CREATE INDEX idx_vp_person ON verse_people(person_id)")

    conn.commit()

def load_people(conn: sqlite3.Connection, persons_csv: str):
    cur = conn.cursor()
    count = 0
    with open(persons_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("person_id") or "").strip()
            name = (row.get("person_name") or "").strip()
            if not pid or not name:
                continue
            cur.execute("""
                INSERT OR REPLACE INTO people
                (person_id, person_name, surname, unique_attribute, sex, tribe, person_notes, name_instance, person_sequence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid,
                name,
                (row.get("surname") or "").strip(),
                (row.get("unique_attribute") or "").strip(),
                (row.get("sex") or "").strip(),
                (row.get("tribe") or "").strip(),
                (row.get("person_notes") or "").strip(),
                int((row.get("name_instance") or "0").strip() or 0),
                int((row.get("person_sequence") or "0").strip() or 0),
            ))
            count += 1
    conn.commit()
    print(f"✅ Loaded people: {count:,}")

def load_verse_links(conn: sqlite3.Connection, person_verse_csv: str):
    cur = conn.cursor()
    count = 0
    with open(person_verse_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ref = (row.get("reference_id") or "").strip()
            pid = (row.get("person_id") or "").strip()
            if not ref or not pid:
                continue

            parsed = parse_reference_id(ref)
            if not parsed:
                continue
            book, chapter, verse = parsed

            def to_int(x, default=0):
                try:
                    return int(str(x).strip())
                except:
                    return default

            cur.execute("""
                INSERT OR REPLACE INTO verse_people
                (book, chapter, verse, person_id,
                 person_verse_id, person_label_id, person_label, person_label_count,
                 person_verse_sequence, person_verse_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book, chapter, verse, pid,
                (row.get("person_verse_id") or "").strip(),
                (row.get("person_label_id") or "").strip(),
                (row.get("person_label") or "").strip(),
                to_int(row.get("person_label_count"), 0),
                to_int(row.get("person_verse_sequence"), 0),
                (row.get("person_verse_notes") or "").strip(),
            ))
            count += 1

            if count % 50000 == 0:
                conn.commit()
                print(f"… inserted {count:,} verse links")

    conn.commit()
    print(f"✅ Linked people to verses: {count:,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    ap.add_argument("--out", default="people.db")
    args = ap.parse_args()

    persons_csv = os.path.join(args.raw_dir, "persons.csv")
    person_verse_csv = os.path.join(args.raw_dir, "person_verse.csv")

    if not os.path.isfile(persons_csv):
        raise SystemExit(f"Missing: {persons_csv}")
    if not os.path.isfile(person_verse_csv):
        raise SystemExit(f"Missing: {person_verse_csv}")

    if os.path.exists(args.out):
        os.remove(args.out)

    conn = sqlite3.connect(args.out)
    try:
        init_db(conn)
        load_people(conn, persons_csv)
        load_verse_links(conn, person_verse_csv)
    finally:
        conn.close()

    print(f"🎉 Done. DB written to: {args.out}")

if __name__ == "__main__":
    main()