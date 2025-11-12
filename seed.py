#!/usr/bin/env python3
# seed.py
import os
import csv
import sys
import sqlite3
import argparse
from typing import Dict, Any, Iterable

DEFAULT_DB = os.environ.get("INTERLINEAR_DB", "interlinear.sqlite3")

# Defaults:
# - normalized file produced by normalize_ot_csv.py
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "interlinear_tokens.normalized.csv")

INSERT_SQL = """
INSERT INTO tokens
(book_code, chapter, verse, token_index, surface, lemma, translit, gloss, morph, strong)
VALUES (:book_code, :chapter, :verse, :token_index, :surface, :lemma, :translit, :gloss, :morph, :strong)
"""

def batched(iterable: Iterable[Dict[str, Any]], n: int):
    """Yield lists of size n from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def coerce_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce and normalize one CSV row to the schema types."""
    try:
        row["chapter"] = int(str(row["chapter"]).strip())
        row["verse"] = int(str(row["verse"]).strip())
        row["token_index"] = int(str(row["token_index"]).strip())
    except Exception as e:
        raise ValueError(f"Bad numeric fields in row: {row}") from e

    # Normalize empties so SQLite executemany won't choke
    for key in ("lemma", "translit", "gloss", "morph", "strong", "surface", "book_code"):
        row[key] = (row.get(key) or "").strip()

    # Minimal sanity checks
    if not row["book_code"] or not row["surface"]:
        raise ValueError(f"Missing required fields in row: {row}")

    return row

def ensure_schema(conn: sqlite3.Connection):
    """Create table/indexes if they don't exist (safe to run)."""
    conn.executescript(
        """
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
    )
    conn.commit()

def seed(csv_path: str,
         db_path: str,
         append: bool = False,
         batch_size: int = 50_000,
         vacuum: bool = False):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Connect
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Speed PRAGMAs (safe for bulk-loading)
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB page cache
    ensure_schema(conn)

    # Clear existing data unless --append
    if not append:
        print("🧹 Clearing existing rows in tokens …")
        conn.execute("DELETE FROM tokens;")
        conn.commit()

    # Stream CSV and insert in big batches
    total = 0
    bad = 0
    print(f"📥 Reading CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"book_code","chapter","verse","token_index","surface","lemma","translit","gloss","morph","strong"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

        # Process in batches
        for batch in batched((coerce_row(r) for r in reader), batch_size):
            try:
                conn.execute("BEGIN;")
                conn.executemany(INSERT_SQL, batch)
                conn.commit()
                total += len(batch)
                print(f"  … inserted {total:,} rows", end="\r", flush=True)
            except Exception as e:
                conn.rollback()
                # Try row-by-row to isolate bad lines
                for r in batch:
                    try:
                        conn.execute(INSERT_SQL, r)
                        total += 1
                    except Exception:
                        bad += 1
                conn.commit()
                print(f"\n⚠️ Batch error; isolated bad rows so far: {bad:,}")

    print(f"\n✅ Done. Inserted: {total:,} rows. Bad rows skipped: {bad:,}.")

    if vacuum:
        print("🧽 VACUUM …")
        conn.execute("VACUUM;")

    conn.close()
    print(f"📦 DB ready at: {db_path}")

def parse_args():
    ap = argparse.ArgumentParser(description="Seed interlinear tokens into SQLite.")
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"Path to normalized CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--db", default=DEFAULT_DB,
                    help=f"SQLite path (default: {DEFAULT_DB} or $INTERLINEAR_DB)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing rows instead of clearing.")
    ap.add_argument("--batch-size", type=int, default=50_000,
                    help="Rows per transaction batch (default 50k).")
    ap.add_argument("--vacuum", action="store_true",
                    help="Run VACUUM after insert (shrinks DB).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        seed(csv_path=args.csv,
             db_path=args.db,
             append=args.append,
             batch_size=args.batch_size,
             vacuum=args.vacuum)
    except Exception as e:
        print(f"❌ Seeding failed: {e}", file=sys.stderr)
        sys.exit(1)
