# build_places_db.py
import os, csv, sqlite3, argparse, re

RAW_DIR_DEFAULT = os.path.join("data", "metadata_raw")

# Same style as your builder: parse "GEN 1:1" → ("GEN", 1, 1) :contentReference[oaicite:1]{index=1}
def parse_reference_id(ref_id: str):
    if not ref_id:
        return None
    ref_id = str(ref_id).strip()
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
    try:
        chapter = int(chap_str)
    except:
        return None

    book = remainder[:last_space].strip().upper()
    return (book, chapter, verse)

def to_int(x, default=0):
    try:
        s = "" if x is None else str(x).strip()
        return int(float(s)) if s else default
    except:
        return default

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS places")
    cur.execute("DROP TABLE IF EXISTS verse_places")

    # place_id is TEXT (e.g., "heaven_1")
    cur.execute("""
        CREATE TABLE places (
            place_id TEXT PRIMARY KEY,
            place_name TEXT,
            place_type TEXT,
            modern_equivalent TEXT,
            place_notes TEXT,
            openbible_id TEXT,
            openbible_url TEXT,
            name_instance INTEGER,
            place_sequence INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE verse_places (
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            place_id TEXT NOT NULL,

            place_verse_id TEXT,
            place_label_id TEXT,
            place_label TEXT,
            place_label_count INTEGER,
            place_verse_sequence INTEGER,
            place_verse_notes TEXT,

            PRIMARY KEY (book, chapter, verse, place_id, place_label_id, place_verse_sequence)
        )
    """)

    cur.execute("CREATE INDEX idx_vp_ref ON verse_places(book, chapter, verse)")
    cur.execute("CREATE INDEX idx_vp_place ON verse_places(place_id)")
    cur.execute("CREATE INDEX idx_places_name ON places(place_name)")

    conn.commit()

def load_places(conn: sqlite3.Connection, places_csv: str):
    cur = conn.cursor()
    count = 0
    with open(places_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("place_id") or "").strip()
            name = (row.get("place_name") or "").strip()
            if not pid or not name:
                continue
            cur.execute("""
                INSERT OR REPLACE INTO places
                (place_id, place_name, place_type, modern_equivalent, place_notes, openbible_id, openbible_url, name_instance, place_sequence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pid,
                name,
                (row.get("place_type") or "").strip(),
                (row.get("modern_equivalent") or "").strip(),
                (row.get("place_notes") or "").strip(),
                (row.get("openbible_id") or "").strip(),
                (row.get("openbible_url") or "").strip(),
                to_int(row.get("name_instance"), 0),
                to_int(row.get("place_sequence"), 0),
            ))
            count += 1
    conn.commit()
    print(f"✅ Loaded places: {count:,}")

def load_place_verse(conn: sqlite3.Connection, place_verse_csv: str):
    cur = conn.cursor()
    count = 0
    with open(place_verse_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ref = (row.get("reference_id") or "").strip()
            place_id = (row.get("place_id") or "").strip()
            if not ref or not place_id:
                continue

            parsed = parse_reference_id(ref)
            if not parsed:
                continue
            book, chapter, verse = parsed

            cur.execute("""
                INSERT OR REPLACE INTO verse_places
                (book, chapter, verse, place_id,
                 place_verse_id, place_label_id, place_label, place_label_count,
                 place_verse_sequence, place_verse_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book, chapter, verse, place_id,
                (row.get("place_verse_id") or "").strip(),
                (row.get("place_label_id") or "").strip(),
                (row.get("place_label") or "").strip(),
                to_int(row.get("place_label_count"), 0),
                to_int(row.get("place_verse_sequence"), 0),
                (row.get("place_verse_notes") or "").strip(),
            ))
            count += 1

            if count % 50000 == 0:
                conn.commit()
                print(f"… inserted {count:,} verse-place links")

    conn.commit()
    print(f"✅ Linked places to verses: {count:,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW_DIR_DEFAULT)
    ap.add_argument("--out", default=os.path.join("data", "places.db"))
    args = ap.parse_args()

    places_csv = os.path.join(args.raw_dir, "places.csv")
    place_verse_csv = os.path.join(args.raw_dir, "place_verse.csv")

    if not os.path.isfile(places_csv):
        raise SystemExit(f"Missing: {places_csv}")
    if not os.path.isfile(place_verse_csv):
        raise SystemExit(f"Missing: {place_verse_csv}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if os.path.exists(args.out):
        os.remove(args.out)

    conn = sqlite3.connect(args.out)
    try:
        init_db(conn)
        load_places(conn, places_csv)
        load_place_verse(conn, place_verse_csv)
    finally:
        conn.close()

    print(f"🎉 Done. DB written to: {args.out}")

if __name__ == "__main__":
    main()
