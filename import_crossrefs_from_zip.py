#!/usr/bin/env python3
"""
import_crossrefs_from_zip.py

Download OpenBible.info cross-reference ZIP and import into an SQLite database.

Usage:
  python import_crossrefs_from_zip.py --db cross_references.db
  python import_crossrefs_from_zip.py --db cross_references.db --url https://a.openbible.info/data/cross-references.zip --force

By default the script downloads:
  https://a.openbible.info/data/cross-references.zip

It will:
 - download ZIP (unless already downloaded and --force not given)
 - inspect files inside and parse JSON or CSV
 - insert rows (verse, cross_ref, votes) into an SQLite DB
 - create UNIQUE index to avoid duplicates

License / attribution note:
If you redistribute the data, include:
"Cross-reference data provided by OpenBible.info (https://www.openbible.info/labs/cross-references/), used under the Creative Commons Attribution License."
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import os
import sqlite3
import sys
import tempfile
import re
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

import requests
import zipfile

DEFAULT_URL = "https://a.openbible.info/data/cross-references.zip"
USER_AGENT = "openbible-crossrefs-importer/1.0 (+https://your-site.example)"

# --- DB helpers ---------------------------------
def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS cross_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        verse TEXT NOT NULL,
        cross_ref TEXT NOT NULL,
        votes INTEGER DEFAULT 0
    )
    ''')
    cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS ux_verse_crossref ON cross_references(verse, cross_ref)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_verse ON cross_references(verse)')
    conn.commit()

def insert_records(conn: sqlite3.Connection, records: Iterable[Tuple[str,str,int]], batch: int = 1000):
    cur = conn.cursor()
    buf = []
    count = 0
    for rec in records:
        buf.append(rec)
        if len(buf) >= batch:
            cur.executemany('INSERT OR IGNORE INTO cross_references (verse, cross_ref, votes) VALUES (?, ?, ?)', buf)
            conn.commit()
            count += len(buf)
            buf.clear()
    if buf:
        cur.executemany('INSERT OR IGNORE INTO cross_references (verse, cross_ref, votes) VALUES (?, ?, ?)', buf)
        conn.commit()
        count += len(buf)
    return count

# --- Parsing helpers -----------------------------
def try_decode_bytes(raw: bytes) -> str:
    """
    Decode bytes to str. Try utf-8, fallback to chardet if available, else replace errors.
    """
    try:
        return raw.decode('utf-8')
    except Exception:
        try:
            import chardet as _ch
            enc = _ch.detect(raw).get('encoding') or 'utf-8'
            return raw.decode(enc, errors='replace')
        except Exception:
            return raw.decode('utf-8', errors='replace')

def parse_json_stream_text(text: str):
    """Parse JSON array or line-delimited JSON. Yield (verse, cross_ref, votes)."""
    stripped = text.lstrip()
    if stripped.startswith('[') or stripped.startswith('{'):
        # try parse entire JSON
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for entry in data:
                    verse = (entry.get('verse') or entry.get('src') or entry.get('source') or entry.get('from'))
                    cross_ref = (entry.get('ref') or entry.get('to') or entry.get('dst') or entry.get('target'))
                    votes = entry.get('votes') or entry.get('vote') or 0
                    if cross_ref:
                        yield (verse, cross_ref, int(votes or 0))
                return
        except Exception:
            # Maybe it's line-delimited JSON: fall through to line parsing
            pass

    # try line-delimited JSON (each line is a JSON object)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if isinstance(entry, dict):
                verse = (entry.get('verse') or entry.get('src') or entry.get('source') or entry.get('from'))
                cross_ref = (entry.get('ref') or entry.get('to') or entry.get('dst') or entry.get('target'))
                votes = entry.get('votes') or entry.get('vote') or 0
                if cross_ref:
                    yield (verse, cross_ref, int(votes or 0))
                continue
        except Exception:
            # not JSON line; skip to next
            pass
    # if nothing yielded, caller will try CSV/text fallback

def convert_dot_notation(s: str) -> str:
    """
    Convert dot-notation verses like 'Gen.1.1' or 'Ps.90.2' into 'Gen 1:1' / 'Ps 90:2'.
    If the string doesn't look like that, return it unchanged (trimmed).
    """
    if not s:
        return s
    s = s.strip()
    # quick check: must have at least two dots and digits
    if '.' not in s or not any(ch.isdigit() for ch in s):
        return s
    parts = s.split('.')
    # Expect at least 3 parts: Book . Chapter . Verse
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        book_part = '.'.join(parts[:-2])  # keep any dots in long book abbreviations intact
        # Replace dots inside book part with spaces for readability (e.g., "1John" or "Gen" remain OK)
        book_part_clean = book_part.replace('.', ' ').strip()
        try:
            chap = int(parts[-2])
            vnum = int(parts[-1])
            return f"{book_part_clean} {chap}:{vnum}"
        except Exception:
            return s
    # fallback: if exactly two parts and second contains colon-like, try small transform
    return s

def parse_json_stream(text: str):
    """
    Expecting JSON array of objects like: [{"ref": "Jeremiah 29:11", "votes": 3, ...}, ...]
    We'll yield tuples (verse, cross_ref, votes).
    """
    try:
        data = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"JSON parse error: {exc}")
    for entry in data:
        # openbible uses 'ref' for referenced verse (target)
        # but some datasets might have different shapes; we'll handle common keys
        # If entry has keys 'src' and 'dst' use those; otherwise assume 'ref' is the cross ref and the query verse
        # Historically the file contains objects with keys: 'source'/'target' or maybe 'ref' only.
        # We'll be flexible:
        if isinstance(entry, dict):
            # prefer (verse, cross_ref) where verse is the 'verse' or 'source' or 'referrer'
            # The downloaded file likely pairs with 'from' and 'to' or 'src' and 'dst' or 'ref' referencing target and 'verse' referencing source
            # We'll inspect keys:
            verse = entry.get('verse') or entry.get('src') or entry.get('source') or entry.get('from') or entry.get('ref_from')
            cross_ref = entry.get('ref') or entry.get('to') or entry.get('dst') or entry.get('target') or entry.get('ref_to')
            votes = entry.get('votes') or entry.get('vote') or 0
            # fallback if we only have a pair in a single string (unlikely)
            if not verse and cross_ref and isinstance(cross_ref, str) and ' ' in cross_ref and ':' in cross_ref:
                # no source, odd; skip
                continue
            if not cross_ref:
                continue
            yield (verse, cross_ref, int(votes or 0))
        else:
            # not a dict - skip
            continue

import csv
import io

def parse_csv_text(text: str):
    """
    Parse TSV/CSV text. Prefer TSV (tab) if header or tabs present.
    Yields (verse, cross_ref, votes).
    """
    if not text:
        return
    # Use first few lines to detect delimiter (tab preferred)
    lines = text.splitlines()
    first = lines[0] if lines else ''
    delim = '\t' if '\t' in first else ','
    f = io.StringIO(text)

    # If header appears to contain 'From' or 'From Verse', use DictReader
    if delim == '\t':
        reader = csv.DictReader(f, delimiter='\t')
        # Normalize header keys to lowercase no-spaces for robust lookup
        fieldmap = {}
        if reader.fieldnames:
            for name in reader.fieldnames:
                if not name:
                    continue
                k = name.strip().lower().replace(' ', '').replace('_','')
                fieldmap[k] = name  # map normalized key -> actual header name

        for row in reader:
            # Try common header names
            verse_raw = None
            cross_raw = None
            votes = 0
            # possible normalized keys
            for k in ('fromverse','from','from_verse','fromverse'):
                if k in fieldmap:
                    verse_raw = row.get(fieldmap[k])
                    break
            for k in ('toverse','to','toversion','to_verse','tov'):
                if k in fieldmap:
                    cross_raw = row.get(fieldmap[k])
                    break
            # votes
            for k in ('votes','vote','count'):
                if k in fieldmap:
                    v = row.get(fieldmap[k])
                    if v and v.strip().isdigit():
                        votes = int(v.strip())
                    break

            # If headers not present, fallback to positional columns
            if verse_raw is None or cross_raw is None:
                # attempt to read positional columns from the row mapping
                vals = [v for v in (row.values()) if v is not None]
                if len(vals) >= 2:
                    verse_raw = vals[0]
                    cross_raw = vals[1]
                    if len(vals) >= 3 and isinstance(vals[2], str) and vals[2].strip().isdigit():
                        votes = int(vals[2].strip())

            if cross_raw:
                yield (convert_dot_notation(verse_raw.strip()) if verse_raw else None,
                       convert_dot_notation(cross_raw.strip()),
                       int(votes or 0))
        return

    # Non-tab CSV fallback
    f.seek(0)
    reader = csv.reader(f, delimiter=delim)
    for row in reader:
        if not row:
            continue
        if len(row) >= 2:
            verse_raw = row[0].strip()
            cross_raw = row[1].strip()
            votes = 0
            if len(row) > 2 and row[2].strip().isdigit():
                votes = int(row[2].strip())
            yield (convert_dot_notation(verse_raw), convert_dot_notation(cross_raw), int(votes or 0))
            
def parse_csv_stream(text: str):
    """
    Parse CSV where likely columns include: verse,cross_ref,votes or similar.
    We'll try to detect columns.
    """
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    # try to detect likely column names
    for row in reader:
        # find verse column
        verse = None
        cross_ref = None
        votes = 0
        for candidate in ('verse', 'src', 'source', 'from', 'v1', 'verse_from'):
            if candidate in row and row[candidate].strip():
                verse = row[candidate].strip()
                break
        for candidate in ('ref', 'cross_ref', 'target', 'dst', 'to', 'v2', 'verse_to'):
            if candidate in row and row[candidate].strip():
                cross_ref = row[candidate].strip()
                break
        for candidate in ('votes','vote','weight','count'):
            if candidate in row and row[candidate].strip():
                try:
                    votes = int(row[candidate].strip())
                except Exception:
                    votes = 0
                break
        if not cross_ref:
            # maybe the CSV is two columns without headers
            # fallback: if there are exactly 2 columns, take them
            vals = [v for v in row.values() if v.strip()]
            if len(vals) >= 2:
                verse, cross_ref = vals[0].strip(), vals[1].strip()
        if cross_ref:
            yield (verse, cross_ref, int(votes or 0))

# --- Download & extract --------------------------
def download_zip(url: str, dest_path: str, force: bool = False, chunk_size: int = 1024*32):
    if os.path.exists(dest_path) and not force:
        print(f"[info] ZIP already exists at {dest_path} (use --force to redownload)")
        return dest_path
    print(f"[info] Downloading {url} ...")
    headers = {'User-Agent': USER_AGENT}
    r = requests.get(url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get('content-length') or 0)
    with open(dest_path, 'wb') as fh:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r[download] {pct}% ({downloaded}/{total} bytes)", end='', flush=True)
    if total:
        print()
    print(f"[info] Saved ZIP to {dest_path}")
    return dest_path

def extract_and_parse(zip_path: str):
    """
    Inspect files inside the zip. Parse any JSON or CSV files we can find.
    Yields (verse, cross_ref, votes).
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        print(f"[info] ZIP contains: {namelist}")
        # prefer known sensible filenames, otherwise iterate through all
        # Common expected file: cross-references.json or cross-references.csv
        preferred = [n for n in namelist if n.lower().endswith('.json') or n.lower().endswith('.csv')]
        if not preferred:
            raise RuntimeError("No JSON or CSV files found in ZIP.")
        for name in preferred:
            print(f"[info] Parsing {name} ...")
            with zf.open(name) as fh:
                # read as bytes then decode
                raw = fh.read()
                # try json first
                text = raw.decode('utf-8', errors='replace')
                # heuristic: if starts with '[' treat as JSON array
                stripped = text.lstrip()
                if stripped.startswith('[') or stripped.startswith('{'):
                    # JSON
                    for rec in parse_json_stream(text):
                        yield rec
                else:
                    # try CSV
                    for rec in parse_csv_stream(text):
                        yield rec

_VERSE_DOT_RE = re.compile(r'[A-Za-z0-9\.]+?\.\d+\.\d+')  # matches patterns like Gen.1.1 or Ps.90.2

def parse_text_lines_fallback(text: str):
    """
    Fallback parser for lines like:
      Gen.1.1\tPs.90.2\t58
    or simple whitespace-separated variants. Yields (verse, cross_ref, votes).
    """
    if not text:
        return
    for i, line in enumerate(text.splitlines()):
        raw = line.strip()
        if not raw or raw.startswith('#'):
            continue
        # Split by tab first, then whitespace
        parts = raw.split('\t') if '\t' in raw else raw.split()
        if len(parts) >= 2:
            verse_raw = parts[0].strip()
            cross_raw = parts[1].strip()
            votes = 0
            if len(parts) >= 3:
                v = parts[2].strip()
                if v.isdigit():
                    votes = int(v)
            yield (convert_dot_notation(verse_raw), convert_dot_notation(cross_raw), int(votes or 0))
            continue

        # If nothing obvious, try regex to find two dot-notation tokens in the line
        found = _VERSE_DOT_RE.findall(raw)
        if len(found) >= 2:
            yield (convert_dot_notation(found[0]), convert_dot_notation(found[1]), 0)
            continue

        # else skip ambiguous line
        # Uncomment for debugging on small samples:
        # print(f"[debug] Skipping ambiguous line {i+1}: {raw[:120]}")
        continue

def extract_and_parse(zip_path: str):
    """
    Inspect files inside the zip. Parse JSON, CSV, or TXT files flexibly.
    Yields (verse, cross_ref, votes).
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        print(f"[info] ZIP contains: {namelist}")
        # prefer text/csv/json files
        files_to_try = [n for n in namelist if n.lower().endswith(('.txt', '.tsv', '.csv', '.json'))]
        if not files_to_try:
            files_to_try = namelist
        for name in files_to_try:
            print(f"[info] Parsing {name} ...")
            with zf.open(name) as fh:
                raw = fh.read()
                text = try_decode_bytes(raw)
                # Try CSV/TSV first (our file is TSV)
                yielded = False
                try:
                    for rec in parse_csv_text(text):
                        yielded = True
                        yield rec
                except Exception as e:
                    # CSV attempt failed; fallback to other methods
                    print(f"[warn] parse_csv_text failed for {name}: {e}")

                if yielded:
                    continue

                # Try JSON (array or line-delimited)
                try:
                    for rec in parse_json_stream_text(text):
                        yield rec
                    # If parse_json_stream_text yields, it will have yielded; but we don't short-circuit here
                except Exception:
                    pass

                # Fallback to forgiving text parsing
                for rec in parse_text_lines_fallback(text):
                    yield rec
                    
# --- CLI -----------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Import OpenBible cross-references ZIP into SQLite")
    p.add_argument('--db', required=True, help='SQLite DB file to write (will be created if missing)')
    p.add_argument('--url', default=DEFAULT_URL, help='ZIP URL to download (default: official OpenBible URL)')
    p.add_argument('--out-zip', default=None, help='Optional local path for ZIP (defaults to temp path)')
    p.add_argument('--force', action='store_true', help='Force redownload of ZIP even if file exists')
    p.add_argument('--batch', type=int, default=2000, help='Batch size for DB inserts')
    return p.parse_args()

def main():
    args = parse_args()
    # choose zip path
    if args.out_zip:
        zip_path = args.out_zip
    else:
        # create a stable temp filename in current dir for caching
        parsed = urlparse(args.url)
        filename = os.path.basename(parsed.path) or "crossrefs.zip"
        zip_path = os.path.abspath(filename)

    # download
    try:
        download_zip(args.url, zip_path, force=args.force)
    except Exception as exc:
        print("[error] failed to download ZIP:", exc)
        sys.exit(1)

    # parse & insert
    conn = sqlite3.connect(args.db)
    init_db(conn)
    gen = extract_and_parse(zip_path)

    # The dataset may include entries where the source verse is missing. OpenBible dataset historically
    # contains objects where 'ref' is the cross-ref and the filename or context implies the source.
    # But in practice the downloaded JSON contains objects with 'verse' and 'ref' pairs. We will filter where cross_ref exists.
    def normalised_records():
        for verse, cross_ref, votes in gen:
            if not cross_ref:
                continue
            # If verse is None, skip (we expect both)
            # but if verse is missing and cross_ref contains both in a "A -> B" format, try to split? not required here.
            if not verse:
                # skip or attempt to continue
                continue
            yield (verse.strip(), cross_ref.strip(), int(votes or 0))

    try:
        inserted = insert_records(conn, normalised_records(), batch=args.batch)
        print(f"[done] Inserted (attempted) rows into DB (notes: duplicates ignored): approx {inserted}")
    finally:
        conn.close()

    print("[info] Done. DB file:", os.path.abspath(args.db))
    print('\nAttribution reminder:')
    print('Cross-reference data provided by OpenBible.info (https://www.openbible.info/labs/cross-references/), used under the Creative Commons Attribution License.')

if __name__ == "__main__":
    main()
