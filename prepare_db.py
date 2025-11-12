import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Any

DATA_DIR = Path("bible_data")
DB_PATH = Path("bible.db")

# Map common 3-letter book codes to full names
BOOK_CODE_TO_NAME: Dict[str, str] = {
    # OT
    "GEN":"Genesis","EXO":"Exodus","LEV":"Leviticus","NUM":"Numbers","DEU":"Deuteronomy",
    "JOS":"Joshua","JDG":"Judges","RUT":"Ruth","1SA":"1 Samuel","2SA":"2 Samuel",
    "1KI":"1 Kings","2KI":"2 Kings","1CH":"1 Chronicles","2CH":"2 Chronicles",
    "EZR":"Ezra","NEH":"Nehemiah","EST":"Esther","JOB":"Job","PSA":"Psalms",
    "PRO":"Proverbs","ECC":"Ecclesiastes","SNG":"Song of Solomon","SON":"Song of Solomon",
    "SOS":"Song of Solomon","ISA":"Isaiah","JER":"Jeremiah","LAM":"Lamentations",
    "EZK":"Ezekiel","EZE":"Ezekiel","DAN":"Daniel","HOS":"Hosea","JOL":"Joel",
    "AMO":"Amos","OBA":"Obadiah","JON":"Jonah","MIC":"Micah","NAH":"Nahum",
    "HAB":"Habakkuk","ZEP":"Zephaniah","HAG":"Haggai","ZEC":"Zechariah","ZCH":"Zechariah",
    "MAL":"Malachi",
    # NT
    "MAT":"Matthew","MRK":"Mark","MAR":"Mark","LUK":"Luke","JHN":"John","JOH":"John",
    "ACT":"Acts","ROM":"Romans","1CO":"1 Corinthians","2CO":"2 Corinthians",
    "GAL":"Galatians","EPH":"Ephesians","PHP":"Philippians","PHI":"Philippians",
    "COL":"Colossians","1TH":"1 Thessalonians","2TH":"2 Thessalonians",
    "1TI":"1 Timothy","2TI":"2 Timothy","TIT":"Titus","PHM":"Philemon",
    "HEB":"Hebrews","JAS":"James","JAM":"James","1PE":"1 Peter","2PE":"2 Peter",
    "1JN":"1 John","2JN":"2 John","3JN":"3 John","JUD":"Jude","REV":"Revelation",
}

SUPPORTED_TRANSLATIONS = {"KJV", "WEBUS"}

# -------- Filesystem discovery ------------------------------------------------

def discover_chapter_files() -> List[Tuple[str, str, Path]]:
    """
    Walk bible_data/<TRAN>/<BOOK_CODE>/<chapter>.json
    Returns list of tuples: (translation, book_code, chapter_path)
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError("Missing bible_data/ folder.")

    results: List[Tuple[str, str, Path]] = []
    for translation_dir in DATA_DIR.iterdir():
        if not translation_dir.is_dir():
            continue
        translation = translation_dir.name.upper()

        # Allow only declared translations (prevents accidental extras)
        if translation not in SUPPORTED_TRANSLATIONS:
            # If you want to allow any folder name, remove this continue.
            continue

        for book_dir in translation_dir.iterdir():
            if not book_dir.is_dir():
                continue
            book_code = book_dir.name.upper()
            for chapter_file in sorted(book_dir.glob("*.json"), key=lambda p: safe_int(p.stem)):
                results.append((translation, book_code, chapter_file))
    if not results:
        raise FileNotFoundError("No chapter JSON files found under bible_data/.")
    return results

def safe_int(s: str) -> int:
    try:
        return int(s)
    except:
        return 0

# -------- JSON parsing --------------------------------------------------------

def parse_int_like(val) -> int | None:
    """Return an int if val looks like a number (handles '1', '1a', '1-2', '#'), else None."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        digits = "".join(ch for ch in val if ch.isdigit())
        if digits:
            return int(digits)
    return None


def parse_chapter_json(path: Path) -> Iterator[Tuple[int, str]]:
    """
    Yields (verse_number, text) for the chapter JSON.

    JSON shapes handled:
      A) { "book": "...", "chapter": 1, "verses": [ {"verse":1,"text":"..."}, ... ] }
      B) { "chapter": 1, "verses": { "1":"...", "2":"..." } }
      C) [ {"verse":1,"text":"..."}, {"verse":2,"text":"..."} ]
      D) { "1":"...", "2":"..." }  # plain mapping of verse#->text

    Verse numbers like '1a', '1-2', or '#' are tolerated:
      - We extract the first digits (e.g., '1a' -> 1, '1-2' -> 1)
      - If no digits exist, we skip that entry.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    def emit(vnum_raw, text_raw):
        vnum = parse_int_like(vnum_raw)
        txt = ("" if text_raw is None else str(text_raw)).strip()
        if vnum is not None and vnum >= 1 and txt:
            yield vnum, txt

    # Case C: list of verse dicts
    if isinstance(data, list):
        for item in data:
            vkey = item.get("verse") or item.get("v") or item.get("num")
            txt = item.get("text") or item.get("t")
            yield from emit(vkey, txt)
        return

    # Case A/B: dict with "verses" key
    if isinstance(data, dict) and "verses" in data:
        verses = data["verses"]
        if isinstance(verses, list):
            for item in verses:
                vkey = item.get("verse") or item.get("v") or item.get("num")
                txt = item.get("text") or item.get("t")
                yield from emit(vkey, txt)
        elif isinstance(verses, dict):
            for k, v in verses.items():
                yield from emit(k, v)
        return

    # Case D: plain dict of verse -> text
    if isinstance(data, dict):
        # Heuristic: if most keys look like verse numbers, treat as mapping
        keys = list(data.keys())
        if keys and sum(1 for k in keys if parse_int_like(k) is not None) >= max(1, len(keys)//2):
            for k, v in data.items():
                yield from emit(k, v)
            return

    raise ValueError(f"Unsupported JSON shape in {path}")

# -------- DB build ------------------------------------------------------------

def build_db():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    cur = conn.cursor()

    # Base table identical to earlier version
    cur.execute("""
        CREATE TABLE verses (
            id INTEGER PRIMARY KEY,
            translation TEXT NOT NULL,  -- e.g., KJV, WEBUS
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            reference TEXT NOT NULL,    -- "Book C:V"
            text TEXT NOT NULL
        );
    """)

    # FTS5 index on verse text
    cur.execute("""
        CREATE VIRTUAL TABLE verses_fts USING fts5(
            text,
            content='verses',
            content_rowid='id',
            tokenize='porter'
        );
    """)

    cur.execute("CREATE INDEX idx_ref ON verses(reference);")
    cur.execute("CREATE INDEX idx_trans_book_chapter_verse ON verses(translation, book, chapter, verse);")

    total_inserted = 0
    translations_seen = set()

    print("Scanning chapter files...")
    batch: List[Tuple[str, str, int, int, str, str]] = []
    for translation, book_code, chapter_path in discover_chapter_files():
        translations_seen.add(translation)
        book_full = BOOK_CODE_TO_NAME.get(book_code, book_code)  # fallback to code if unknown
        chapter_num = safe_int(chapter_path.stem)
        if chapter_num <= 0:
            continue

        for vnum, vtext in parse_chapter_json(chapter_path):
            ref = f"{book_full} {chapter_num}:{vnum}"
            batch.append((translation, book_full, chapter_num, vnum, ref, vtext))
            if len(batch) >= 10_000:
                cur.executemany(
                    "INSERT INTO verses (translation, book, chapter, verse, reference, text) VALUES (?, ?, ?, ?, ?, ?)",
                    batch
                )
                total_inserted += len(batch)
                batch.clear()

    if batch:
        cur.executemany(
            "INSERT INTO verses (translation, book, chapter, verse, reference, text) VALUES (?, ?, ?, ?, ?, ?)",
            batch
        )
        total_inserted += len(batch)
        batch.clear()

    conn.commit()

    print("Indexing FTS…")
    cur.execute("INSERT INTO verses_fts(rowid, text) SELECT id, text FROM verses;")
    conn.commit()

    # meta table for translation list
    cur.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);")
    cur.execute("INSERT INTO meta(k, v) VALUES('translations', ?);", (",".join(sorted(translations_seen)),))

    conn.commit()
    conn.close()
    print(f"Built {DB_PATH} with {total_inserted} verse rows across: {', '.join(sorted(translations_seen))}")

if __name__ == "__main__":
    build_db()
