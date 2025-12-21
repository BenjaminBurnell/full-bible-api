import os
import sqlite3
import pandas as pd
from pathlib import Path

# --- BookID (1..66) -> 3-letter code (matches your API's _normalize_bible_book output) ---
BOOKID_TO_CODE = {
    1: "GEN", 2: "EXO", 3: "LEV", 4: "NUM", 5: "DEU", 6: "JOS", 7: "JDG", 8: "RUT",
    9: "1SA", 10: "2SA", 11: "1KI", 12: "2KI", 13: "1CH", 14: "2CH", 15: "EZR",
    16: "NEH", 17: "EST", 18: "JOB", 19: "PSA", 20: "PRO", 21: "ECC", 22: "SNG",
    23: "ISA", 24: "JER", 25: "LAM", 26: "EZK", 27: "DAN", 28: "HOS", 29: "JOL",
    30: "AMO", 31: "OBA", 32: "JON", 33: "MIC", 34: "NAM", 35: "HAB", 36: "ZEP",
    37: "HAG", 38: "ZEC", 39: "MAL", 40: "MAT", 41: "MRK", 42: "LUK", 43: "JHN",
    44: "ACT", 45: "ROM", 46: "1CO", 47: "2CO", 48: "GAL", 49: "EPH", 50: "PHP",
    51: "COL", 52: "1TH", 53: "2TH", 54: "1TI", 55: "2TI", 56: "TIT", 57: "PHM",
    58: "HEB", 59: "JAS", 60: "1PE", 61: "2PE", 62: "1JN", 63: "2JN", 64: "3JN",
    65: "JUD", 66: "REV",
}

def build_places_db(
    mainindex_csv: str,
    metav_places_csv: str,
    out_db_path: str,
) -> None:
    mainindex_csv = str(Path(mainindex_csv))
    metav_places_csv = str(Path(metav_places_csv))
    out_db_path = str(Path(out_db_path))

    if not os.path.isfile(mainindex_csv):
        raise FileNotFoundError(f"MainIndex.csv not found: {mainindex_csv}")
    if not os.path.isfile(metav_places_csv):
        raise FileNotFoundError(f"MetaV_Places.csv not found: {metav_places_csv}")

    os.makedirs(str(Path(out_db_path).parent), exist_ok=True)

    # Load datasets
    main_df = pd.read_csv(mainindex_csv)
    places_df = pd.read_csv(metav_places_csv)

    # --- Build 'places' table data ---
    # MetaV_Places.csv columns: PlaceID, PlaceName, Root, Comment, Lat, Lon, PlaceMarkID
    # Your API expects (at least): place_id, place_name, place_type, modern_equivalent, place_notes,
    # openbible_id, openbible_url, name_instance, place_sequence
    places_df = places_df.copy()
    places_df["PlaceID"] = pd.to_numeric(places_df["PlaceID"], errors="coerce").fillna(0).astype(int)
    places_df["PlaceName"] = places_df["PlaceName"].fillna("").astype(str)

    places_out = pd.DataFrame({
        "place_id": places_df["PlaceID"],
        "place_name": places_df["PlaceName"],
        "place_type": "",  # MetaV doesn't provide a type in this CSV
        "modern_equivalent": places_df.get("Root", pd.Series([""] * len(places_df))).fillna("").astype(str),
        "place_notes": places_df.get("Comment", pd.Series([""] * len(places_df))).fillna("").astype(str),
        "openbible_id": "",
        "openbible_url": "",
        "name_instance": 0,
        # place_sequence: keep deterministic ordering (by place_id)
        "place_sequence": 0,
        # optional geo columns (not used by your endpoint, but handy to keep)
        "lat": pd.to_numeric(places_df.get("Lat", pd.Series([None] * len(places_df))), errors="coerce"),
        "lon": pd.to_numeric(places_df.get("Lon", pd.Series([None] * len(places_df))), errors="coerce"),
        "place_mark_id": pd.to_numeric(places_df.get("PlaceMarkID", pd.Series([None] * len(places_df))), errors="coerce"),
    }).sort_values("place_id")

    places_out["place_sequence"] = range(1, len(places_out) + 1)

    # --- Build 'verse_places' table data ---
    # MainIndex.csv is word-level; PlaceID is assigned per word.
    needed_cols = {"BookID", "Chapter", "VerseNum", "VersePos", "PlaceID"}
    missing = needed_cols - set(main_df.columns)
    if missing:
        raise ValueError(f"MainIndex.csv missing columns: {sorted(list(missing))}")

    df = main_df[["BookID", "Chapter", "VerseNum", "VersePos", "PlaceID"]].copy()
    df["PlaceID"] = pd.to_numeric(df["PlaceID"], errors="coerce").fillna(0).astype(int)
    df = df[df["PlaceID"] != 0].copy()  # only words tagged with a place

    # Map BookID -> book code used by API ("GEN", etc.)
    df["book"] = df["BookID"].map(BOOKID_TO_CODE)
    df = df[df["book"].notna()].copy()

    df["chapter"] = pd.to_numeric(df["Chapter"], errors="coerce").fillna(0).astype(int)
    df["verse"] = pd.to_numeric(df["VerseNum"], errors="coerce").fillna(0).astype(int)
    df["verse_pos"] = pd.to_numeric(df["VersePos"], errors="coerce").fillna(0).astype(int)
    df["place_id"] = df["PlaceID"]

    # label_count = how many words in that verse were tagged with that PlaceID
    # place_verse_sequence = order places by first occurrence position in verse
    grouped = (
        df.groupby(["book", "chapter", "verse", "place_id"], as_index=False)
          .agg(
              place_label_count=("place_id", "size"),
              first_pos=("verse_pos", "min"),
          )
          .sort_values(["book", "chapter", "verse", "first_pos", "place_id"])
    )

    # Join to place_name for label
    place_name_map = dict(zip(places_out["place_id"], places_out["place_name"]))
    grouped["place_label"] = grouped["place_id"].map(place_name_map).fillna("")
    grouped["place_label_id"] = grouped["place_id"].astype(str)  # simple stable id
    grouped["place_verse_notes"] = ""

    # Within each verse, assign sequence 1..N
    grouped["place_verse_sequence"] = (
        grouped.groupby(["book", "chapter", "verse"])["first_pos"]
        .rank(method="dense")
        .astype(int)
    )

    verse_places_out = grouped[[
        "book", "chapter", "verse",
        "place_id",
        "place_label_id", "place_label", "place_label_count",
        "place_verse_sequence", "place_verse_notes",
    ]].copy()

    # --- Write SQLite ---
    if os.path.isfile(out_db_path):
        os.remove(out_db_path)

    conn = sqlite3.connect(out_db_path)
    try:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE places (
            place_id INTEGER PRIMARY KEY,
            place_name TEXT,
            place_type TEXT,
            modern_equivalent TEXT,
            place_notes TEXT,
            openbible_id TEXT,
            openbible_url TEXT,
            name_instance INTEGER,
            place_sequence INTEGER,
            lat REAL,
            lon REAL,
            place_mark_id INTEGER
        )
        """)

        cur.execute("""
        CREATE TABLE verse_places (
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            place_id INTEGER NOT NULL,
            place_label_id TEXT,
            place_label TEXT,
            place_label_count INTEGER,
            place_verse_sequence INTEGER,
            place_verse_notes TEXT,
            PRIMARY KEY (book, chapter, verse, place_id)
        )
        """)

        # Insert
        places_out.to_sql("places", conn, if_exists="append", index=False)
        verse_places_out.to_sql("verse_places", conn, if_exists="append", index=False)

        # Indexes (important for API speed)
        cur.execute("CREATE INDEX idx_verse_places_ref ON verse_places(book, chapter, verse)")
        cur.execute("CREATE INDEX idx_verse_places_place ON verse_places(place_id)")
        cur.execute("CREATE INDEX idx_places_name ON places(place_name)")

        conn.commit()

        # Quick sanity printouts
        places_count = cur.execute("SELECT COUNT(*) FROM places").fetchone()[0]
        links_count = cur.execute("SELECT COUNT(*) FROM verse_places").fetchone()[0]
        print(f"✅ Built {out_db_path}")
        print(f"   places rows: {places_count}")
        print(f"   verse_places rows: {links_count}")

    finally:
        conn.close()


if __name__ == "__main__":
    # Adjust these if your folder layout differs
    root = Path(__file__).resolve().parent
    mainindex = root / "data" / "metadata_raw" / "MainIndex.csv"
    metav_places = root / "data" / "metadata_raw" / "MetaV_Places.csv"
    out_db = root / "data" / "places.db"

    build_places_db(str(mainindex), str(metav_places), str(out_db))
