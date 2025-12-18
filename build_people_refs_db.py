import os
import csv
import sqlite3
import argparse

RAW_DIR_DEFAULT = os.path.join("data", "metadata_raw")

def parse_reference_id(ref_id: str):
    """
    Parses reference_id like 'GEN 1:1' into (book, chapter, verse).
    Returns None if it can't parse.
    """
    if not ref_id:
        return None
    ref_id = str(ref_id).strip()

    colon_i = ref_id.rfind(":")
    if colon_i == -1:
        return None

    verse_part = ref_id[colon_i + 1 :].strip()
    remainder = ref_id[:colon_i].strip()

    space_i = remainder.rfind(" ")
    if space_i == -1:
        return None

    book = remainder[:space_i].strip().upper()
    chap_str = remainder[space_i + 1 :].strip()

    try:
        chapter = int(chap_str)
        verse = int("".join(ch for ch in verse_part if ch.isdigit()))
    except Exception:
        return None

    return (book, chapter, verse)

def to_int(x, default=0):
    try:
        s = "" if x is None else str(x).strip()
        return int(s) if s else default
    except Exception:
        return default

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS people")
    cur.execute("DROP TABLE IF EXISTS person_verses")

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
        CREATE TABLE person_verses (
            person_id TEXT NOT NULL,
            reference_id TEXT NOT NULL,   -- original string like "GEN 1:1"
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,

            person_verse_id TEXT,
            person_label_id TEXT,
            person_label TEXT,
            person_label_count INTEGER,
            person_verse_sequence INTEGER,
            person_verse_notes TEXT,

            PRIMARY KEY (person_id, book, chapter, verse, person_label_id, person_verse_sequence)
        )
    """)

    # Fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pv_person ON person_verses(person_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pv_ref ON person_verses(book, chapter, verse)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_people_name ON people(person_name)")

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
                to_int(row.get("name_instance"), 0),
                to_int(row.get("person_sequence"), 0),
            ))
            count += 1

    conn.commit()
    print(f"✅ Loaded people: {count:,}")

def load_person_verses(conn: sqlite3.Connection, person_verse_csv: str):
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

            cur.execute("""
                INSERT OR REPLACE INTO person_verses
                (person_id, reference_id, book, chapter, verse,
                 person_verse_id, person_label_id, person_label, person_label_count,
                 person_verse_sequence, person_verse_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid, ref, book, chapter, verse,
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
                print(f"… inserted {count:,} person-verse links")

    conn.commit()
    print(f"✅ Linked people to verse references: {count:,}")

def build_db(raw_dir: str, out_db: str):
    persons_csv = os.path.join(raw_dir, "persons.csv")
    person_verse_csv = os.path.join(raw_dir, "person_verse.csv")

    if not os.path.isfile(persons_csv):
        raise SystemExit(f"Missing: {persons_csv}")
    if not os.path.isfile(person_verse_csv):
        raise SystemExit(f"Missing: {person_verse_csv}")

    os.makedirs(os.path.dirname(out_db) or ".", exist_ok=True)
    if os.path.exists(out_db):
        os.remove(out_db)

    conn = sqlite3.connect(out_db)
    try:
        init_db(conn)
        load_people(conn, persons_csv)
        load_person_verses(conn, person_verse_csv)
    finally:
        conn.close()

    print(f"🎉 Done. DB written to: {out_db}")

def find_person_id(conn: sqlite3.Connection, person: str) -> str | None:
    """
    Accepts either a person_id (Adam_1) or an exact name (Adam).
    Returns resolved person_id if found.
    """
    person = (person or "").strip()
    if not person:
        return None

    row = conn.execute("SELECT person_id FROM people WHERE person_id = ?", (person,)).fetchone()
    if row:
        return row[0]

    row = conn.execute(
        "SELECT person_id FROM people WHERE lower(person_name) = lower(?)",
        (person,),
    ).fetchone()
    if row:
        return row[0]

    return None

def print_refs(out_db: str, person: str, limit: int, offset: int):
    if not os.path.isfile(out_db):
        raise SystemExit(f"DB not found: {out_db} (run: build first)")

    conn = sqlite3.connect(out_db)
    try:
        pid = find_person_id(conn, person)
        if not pid:
            raise SystemExit(f"Person not found by id or exact name: {person}")

        person_row = conn.execute(
            "SELECT person_id, person_name, description FROM people WHERE person_id = ?",
            (pid,),
        ).fetchone()

        total = conn.execute(
            "SELECT COUNT(1) FROM person_verses WHERE person_id = ?",
            (pid,),
        ).fetchone()[0]

        rows = conn.execute(
            """
            SELECT reference_id, book, chapter, verse
            FROM person_verses
            WHERE person_id = ?
            ORDER BY book, chapter, verse
            LIMIT ? OFFSET ?
            """,
            (pid, limit, offset),
        ).fetchall()

        print(f"Person: {pid} ({person_row[1] if person_row else ''})")
        print(f"Total references: {total}")
        print("References:")
        for r in rows:
            print(f"- {r[0]}")
    finally:
        conn.close()

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build people_refs.db from CSVs")
    b.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    b.add_argument("--out", default=os.path.join("data", "people_refs.db"))

    r = sub.add_parser("refs", help="Print verse references for a person")
    r.add_argument("--db", default=os.path.join("data", "people_refs.db"))
    r.add_argument("--person", required=True, help="person_id like Adam_1 OR exact name like Adam")
    r.add_argument("--limit", type=int, default=200)
    r.add_argument("--offset", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "build":
        build_db(args.raw_dir, args.out)
    elif args.cmd == "refs":
        print_refs(args.db, args.person, args.limit, args.offset)

if __name__ == "__main__":
    main()
