# build_metav_interlinear.py
"""
Builds the interlinear `tokens` table from the MetaV word index + Strong's index.

- Reads MetaV MainIndex.csv (one row per KJV word)
- Uses StrongsIndex.csv to map WordID -> StrongsID
- Uses Strongs.csv to map StrongsID -> lemma / transliteration / gloss / part of speech
- Populates interlinear.sqlite3/tokens with the exact same schema your API already expects.

This leaves the FastAPI endpoints unchanged. They keep returning:
    {
        "surface", "lemma", "translit", "gloss",
        "morph", "strong", "index",
        "resolved_lemma", "resolved_translit", "resolved_gloss",
        "translation"
    }
but the backing data now comes from MetaV.
"""

import os
import csv
import sqlite3
from typing import Dict, Tuple

# Re-use the existing DB schema from db.py so the table + index are identical.
try:
    from db import SCHEMA as TOKENS_SCHEMA, DB_PATH as DEFAULT_DB_PATH  # type: ignore
except Exception:
    # Fallback: same schema as in db.py
    DEFAULT_DB_PATH = os.environ.get("INTERLINEAR_DB", "interlinear.sqlite3")
    TOKENS_SCHEMA = """
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
    """.strip()

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.environ.get("INTERLINEAR_DB", DEFAULT_DB_PATH)

def _find_path(*rel_candidates: str) -> str:
    """
    Try a small set of relative locations for a given file name.
    This lets you keep MetaV files under ./data/metadata_raw,
    but also works if they sit in ./data or project root.
    """
    for rel in rel_candidates:
        candidate = os.path.join(BASE_DIR, rel)
        if os.path.isfile(candidate):
            return candidate
    # If nothing found, just return the first candidate (so the caller
    # can raise a helpful error mentioning it).
    return os.path.join(BASE_DIR, rel_candidates[0])

def _load_book_map() -> Dict[int, str]:
    """
    Map MetaV BookID -> 3-letter book_code (GEN, EXO, ...)

    Uses data/metadata_raw/books.csv which has columns:
        book_id, usx_code, book_name, ...
    """
    books_path = _find_path(
        os.path.join("data", "metadata_raw", "books.csv"),
        os.path.join("data", "books.csv"),
        "books.csv",
    )
    if not os.path.isfile(books_path):
        raise FileNotFoundError(f"books.csv not found (looked for {books_path})")

    mapping: Dict[int, str] = {}
    with open(books_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            bid_raw = (row.get("book_id") or "").strip()
            code = (row.get("usx_code") or "").strip()
            if not bid_raw or not code:
                continue
            try:
                bid = int(bid_raw)
            except ValueError:
                continue
            mapping[bid] = code.upper()
    if not mapping:
        raise RuntimeError("books.csv loaded but produced an empty BookID → book_code mapping.")
    return mapping

def _load_strongs_index() -> Dict[int, str]:
    """
    Load WordID -> StrongsID from StrongsIndex.csv.
    """
    path = _find_path(
        os.path.join("data", "metadata_raw", "StrongsIndex.csv"),
        os.path.join("data", "StrongsIndex.csv"),
        "StrongsIndex.csv",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"StrongsIndex.csv not found (looked for {path})")

    mapping: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            wid_raw = (row.get("WordID") or "").strip()
            sid = (row.get("StrongsID") or "").strip()
            if not wid_raw or not sid:
                continue
            try:
                wid = int(wid_raw)
            except ValueError:
                continue
            mapping[wid] = sid
    return mapping

def _load_strongs_lexicon_from_metav() -> Dict[str, Dict[str, str]]:
    """
    Load StrongsID → lemma/translit/gloss/morph from MetaV's Strongs.csv.
    """
    path = _find_path(
        os.path.join("data", "metadata_raw", "Strongs.csv"),
        os.path.join("data", "Strongs.csv"),
        "Strongs.csv",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Strongs.csv not found (looked for {path})")

    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = (row.get("StrongsID") or "").strip()
            if not sid:
                continue
            lemma = (row.get("lemma") or "").strip()
            translit = (row.get("xlit") or "").strip()
            gloss = (row.get("description") or "").strip()
            morph = (row.get("PartOfSpeech") or "").strip()
            lang = (row.get("Language") or "").strip()
            # Store enough for the existing enrich_token() logic.
            out[sid] = {
                "lemma": lemma,
                "translit": translit,
                "gloss": gloss,
                # You weren't previously using morph/lang in the JSON,
                # but we can tuck them into "morph" to be somewhat useful.
                "morph": (morph or lang),
            }
    if not out:
        raise RuntimeError("Strongs.csv loaded but produced no lexical entries.")
    return out

def _load_mainindex() -> Tuple[str, Dict[int, str], Dict[int, str]]:
    """
    Find MainIndex.csv and also return helper mappings:
       - book_map: BookID -> book_code
       - strong_index: WordID -> StrongsID
    """
    main_path = _find_path(
        os.path.join("data", "metadata_raw", "MainIndex.csv"),
        os.path.join("data", "MainIndex.csv"),
        "MainIndex.csv",
    )
    if not os.path.isfile(main_path):
        raise FileNotFoundError(f"MainIndex.csv not found (looked for {main_path})")

    book_map = _load_book_map()
    strong_index = _load_strongs_index()
    return main_path, book_map, strong_index

def build_tokens():
    """
    Re-create the tokens table from MetaV + Strong's.
    """
    main_path, book_map, strong_index = _load_mainindex()
    strong_lex = _load_strongs_lexicon_from_metav()

    print(f"[MetaV] MainIndex: {main_path}")
    print(f"[MetaV] Loaded {len(book_map)} book IDs → codes")
    print(f"[MetaV] Loaded {len(strong_index)} WordID → Strong's mappings")
    print(f"[MetaV] Loaded {len(strong_lex)} Strong's lexical entries")

    # --- Open DB and ensure schema is present ---
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(TOKENS_SCHEMA)

    # Start fresh
    cur.execute("DELETE FROM tokens")

    rows_inserted = 0

    with open(main_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                book_id = int((row.get("BookID") or "").strip() or "0")
                if book_id not in book_map:
                    continue
                book_code = book_map[book_id]

                chapter = int((row.get("Chapter") or "0").strip() or "0")
                verse = int((row.get("VerseNum") or "0").strip() or "0")
                # MetaV's VersePos is 0-based; your interlinear uses 1-based token_index.
                verse_pos = int((row.get("VersePos") or "0").strip() or "0")
                token_index = verse_pos + 1

                surface = (row.get("Word") or "").strip()
                punc = (row.get("Punc") or "").strip()
                if punc:
                    surface = surface + punc
                if not surface:
                    # Skip completely empty rows (should be rare)
                    continue

                word_id_raw = (row.get("WordID") or "").strip()
                strong_id = ""
                if word_id_raw:
                    try:
                        wid = int(word_id_raw)
                        strong_id = strong_index.get(wid, "")
                    except ValueError:
                        strong_id = ""

                lemma = ""
                translit = ""
                gloss = ""
                morph = ""

                if strong_id:
                    le = strong_lex.get(strong_id)
                    if le:
                        lemma = le.get("lemma", "") or lemma
                        translit = le.get("translit", "") or translit
                        gloss = le.get("gloss", "") or gloss
                        morph = le.get("morph", "") or morph

                cur.execute(
                    """
                    INSERT INTO tokens(
                        book_code, chapter, verse, token_index,
                        surface, lemma, translit, gloss, morph, strong
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        book_code,
                        chapter,
                        verse,
                        token_index,
                        surface,
                        lemma,
                        translit,
                        gloss,
                        morph,
                        strong_id,
                    ),
                )
                rows_inserted += 1
            except Exception as e:
                # Be noisy but don't kill the whole build for a single bad row.
                print(f"[WARN] Skipping row due to error: {e!r}")
                continue

    conn.commit()
    conn.close()
    print(f"[MetaV] ✅ Rebuilt tokens table in {DB_PATH} with {rows_inserted:,} rows.")

if __name__ == "__main__":
    build_tokens()