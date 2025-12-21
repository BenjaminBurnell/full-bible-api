import os
import sqlite3
from pathlib import Path
import pandas as pd

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

def build_place_refs_db(
    mainindex_csv: str,
    metav_places_csv: str,
    out_db_path: str,
    chunksize: int = 750_000,
    build_agg_table: bool = True,
) -> None:
    mainindex_csv = str(Path(mainindex_csv))
    metav_places_csv = str(Path(metav_places_csv))
    out_db_path = str(Path(out_db_path))

    if not os.path.isfile(mainindex_csv):
        raise FileNotFoundError(f"MainIndex.csv not found: {mainindex_csv}")
    if not os.path.isfile(metav_places_csv):
        raise FileNotFoundError(f"MetaV_Places.csv not found: {metav_places_csv}")

    os.makedirs(str(Path(out_db_path).parent), exist_ok=True)

    if os.path.exists(out_db_path):
        os.remove(out_db_path)

    conn = sqlite3.connect(out_db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB cache if available

    try:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE places (
            place_id INTEGER PRIMARY KEY,
            place_name TEXT,
            root TEXT,
            comment TEXT,
            lat REAL,
            lon REAL,
            place_mark_id INTEGER
        )
        """)

        cur.execute("""
        CREATE TABLE place_verses (
            place_id INTEGER NOT NULL,
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            PRIMARY KEY (place_id, book, chapter, verse)
        )
        """)

        cur.execute("CREATE INDEX idx_place_verses_ref ON place_verses(book, chapter, verse)")
        cur.execute("CREATE INDEX idx_place_verses_place ON place_verses(place_id)")
        cur.execute("CREATE INDEX idx_places_name ON places(place_name)")
        conn.commit()

        # ---- places metadata ----
        places_df = pd.read_csv(metav_places_csv)

        def safe_col(df: pd.DataFrame, name: str):
            return df[name] if name in df.columns else pd.Series([None] * len(df))

        places_out = pd.DataFrame({
            "place_id": pd.to_numeric(safe_col(places_df, "PlaceID"), errors="coerce").fillna(0).astype(int),
            "place_name": safe_col(places_df, "PlaceName").fillna("").astype(str),
            "root": safe_col(places_df, "Root").fillna("").astype(str),
            "comment": safe_col(places_df, "Comment").fillna("").astype(str),
            "lat": pd.to_numeric(safe_col(places_df, "Lat"), errors="coerce"),
            "lon": pd.to_numeric(safe_col(places_df, "Lon"), errors="coerce"),
            "place_mark_id": pd.to_numeric(safe_col(places_df, "PlaceMarkID"), errors="coerce"),
        })

        places_out = places_out.loc[places_out["place_id"] != 0].drop_duplicates(subset=["place_id"])
        places_out.to_sql("places", conn, if_exists="append", index=False)
        conn.commit()

        # ---- ingest verse refs ----
        usecols = ["BookID", "Chapter", "VerseNum", "PlaceID"]
        reader = pd.read_csv(mainindex_csv, usecols=usecols, chunksize=chunksize)

        insert_sql = """
        INSERT OR IGNORE INTO place_verses(place_id, book, chapter, verse)
        VALUES (?, ?, ?, ?)
        """

        total_candidates = 0
        for i, chunk in enumerate(reader, start=1):
            # Use .loc + .copy() to avoid SettingWithCopyWarning
            chunk = chunk.copy()

            chunk.loc[:, "PlaceID"] = pd.to_numeric(chunk["PlaceID"], errors="coerce").fillna(0).astype(int)
            chunk = chunk.loc[chunk["PlaceID"] != 0].copy()
            if chunk.empty:
                continue

            chunk.loc[:, "BookID"] = pd.to_numeric(chunk["BookID"], errors="coerce").fillna(0).astype(int)
            chunk.loc[:, "Chapter"] = pd.to_numeric(chunk["Chapter"], errors="coerce").fillna(0).astype(int)
            chunk.loc[:, "VerseNum"] = pd.to_numeric(chunk["VerseNum"], errors="coerce").fillna(0).astype(int)

            chunk.loc[:, "book"] = chunk["BookID"].map(BOOKID_TO_CODE)
            chunk = chunk.loc[chunk["book"].notna()].copy()
            if chunk.empty:
                continue

            # Unique refs per place within this chunk
            uniq = chunk.drop_duplicates(subset=["PlaceID", "book", "Chapter", "VerseNum"])

            rows = [
                (int(r.PlaceID), str(r.book), int(r.Chapter), int(r.VerseNum))
                for r in uniq.itertuples(index=False)
            ]

            conn.executemany(insert_sql, rows)
            conn.commit()

            total_candidates += len(rows)
            print(f"Chunk {i}: inserted candidate rows {len(rows):,} (running total {total_candidates:,})")

        # ---- optional aggregated table ----
        if build_agg_table:
            # DON'T use a column name "references"
            cur.execute("""
            CREATE TABLE place_refs_agg (
                place_id INTEGER PRIMARY KEY,
                ref_count INTEGER NOT NULL,
                refs TEXT NOT NULL
            )
            """)
            cur.execute("CREATE INDEX idx_place_refs_agg_count ON place_refs_agg(ref_count)")
            conn.commit()

            # Build aggregation from place_verses
            # (Ordering in group_concat is enforced by subquery ORDER BY.)
            cur.execute("""
            INSERT INTO place_refs_agg(place_id, ref_count, refs)
            SELECT
                pv.place_id,
                COUNT(*) AS ref_count,
                (
                    SELECT group_concat(ref, '; ')
                    FROM (
                        SELECT pv2.book || ' ' || pv2.chapter || ':' || pv2.verse AS ref
                        FROM place_verses pv2
                        WHERE pv2.place_id = pv.place_id
                        ORDER BY
                            CASE pv2.book
                                WHEN 'GEN' THEN 1 WHEN 'EXO' THEN 2 WHEN 'LEV' THEN 3 WHEN 'NUM' THEN 4 WHEN 'DEU' THEN 5
                                WHEN 'JOS' THEN 6 WHEN 'JDG' THEN 7 WHEN 'RUT' THEN 8 WHEN '1SA' THEN 9 WHEN '2SA' THEN 10
                                WHEN '1KI' THEN 11 WHEN '2KI' THEN 12 WHEN '1CH' THEN 13 WHEN '2CH' THEN 14 WHEN 'EZR' THEN 15
                                WHEN 'NEH' THEN 16 WHEN 'EST' THEN 17 WHEN 'JOB' THEN 18 WHEN 'PSA' THEN 19 WHEN 'PRO' THEN 20
                                WHEN 'ECC' THEN 21 WHEN 'SNG' THEN 22 WHEN 'ISA' THEN 23 WHEN 'JER' THEN 24 WHEN 'LAM' THEN 25
                                WHEN 'EZK' THEN 26 WHEN 'DAN' THEN 27 WHEN 'HOS' THEN 28 WHEN 'JOL' THEN 29 WHEN 'AMO' THEN 30
                                WHEN 'OBA' THEN 31 WHEN 'JON' THEN 32 WHEN 'MIC' THEN 33 WHEN 'NAM' THEN 34 WHEN 'HAB' THEN 35
                                WHEN 'ZEP' THEN 36 WHEN 'HAG' THEN 37 WHEN 'ZEC' THEN 38 WHEN 'MAL' THEN 39 WHEN 'MAT' THEN 40
                                WHEN 'MRK' THEN 41 WHEN 'LUK' THEN 42 WHEN 'JHN' THEN 43 WHEN 'ACT' THEN 44 WHEN 'ROM' THEN 45
                                WHEN '1CO' THEN 46 WHEN '2CO' THEN 47 WHEN 'GAL' THEN 48 WHEN 'EPH' THEN 49 WHEN 'PHP' THEN 50
                                WHEN 'COL' THEN 51 WHEN '1TH' THEN 52 WHEN '2TH' THEN 53 WHEN '1TI' THEN 54 WHEN '2TI' THEN 55
                                WHEN 'TIT' THEN 56 WHEN 'PHM' THEN 57 WHEN 'HEB' THEN 58 WHEN 'JAS' THEN 59 WHEN '1PE' THEN 60
                                WHEN '2PE' THEN 61 WHEN '1JN' THEN 62 WHEN '2JN' THEN 63 WHEN '3JN' THEN 64 WHEN 'JUD' THEN 65
                                WHEN 'REV' THEN 66 ELSE 999 END,
                            pv2.chapter, pv2.verse
                    )
                ) AS refs
            FROM place_verses pv
            GROUP BY pv.place_id
            """)
            conn.commit()

        places_count = cur.execute("SELECT COUNT(*) FROM places").fetchone()[0]
        link_count = cur.execute("SELECT COUNT(*) FROM place_verses").fetchone()[0]
        print(f"✅ Built {out_db_path}")
        print(f"   places: {places_count:,}")
        print(f"   place_verses: {link_count:,}")
        if build_agg_table:
            agg_count = cur.execute("SELECT COUNT(*) FROM place_refs_agg").fetchone()[0]
            print(f"   place_refs_agg: {agg_count:,}")

    finally:
        conn.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build_place_refs_db(
        mainindex_csv=str(root / "data" / "metadata_raw" / "MainIndex.csv"),
        metav_places_csv=str(root / "data" / "metadata_raw" / "MetaV_Places.csv"),
        out_db_path=str(root / "data" / "place_refs.db"),
        chunksize=750_000,
        build_agg_table=True,
    )
