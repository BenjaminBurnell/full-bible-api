import sqlite3
import re
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Tuple

# CHANGE 1: Use APIRouter instead of FastAPI
router = APIRouter()

DB_PATH = "cross_references.db"

# --- Models ---
class CrossRef(BaseModel):
    verse: str
    cross_ref: str
    votes: int

class CrossRefResponse(BaseModel):
    query: str
    normalized: str
    results: List[CrossRef]

# --- Database Helper ---
def get_db_connection():
    # check_same_thread=False is needed if you don't want to open/close constantly, 
    # but for simple APIs, opening per request is safer.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# --- Normalization Logic ---
# (I am keeping your massive dictionary here to ensure this file runs standalone)
BOOK_ABBREVIATIONS = {
    "genesis": "Gen", "gen": "Gen", "exodus": "Exo", "ex": "Exo",
    "leviticus": "Lev", "lev": "Lev", "numbers": "Num", "num": "Num",
    "deuteronomy": "Deu", "deut": "Deu", "joshua": "Jos", "josh": "Jos",
    "judges": "Jdg", "judg": "Jdg", "ruth": "Rut",
    "1 samuel": "1Sa", "1sam": "1Sa", "2 samuel": "2Sa", "2sam": "2Sa",
    "1 kings": "1Ki", "1kings": "1Ki", "2 kings": "2Ki", "2kings": "2Ki",
    "1 chronicles": "1Ch", "1chron": "1Ch", "2 chronicles": "2Ch", "2chron": "2Ch",
    "ezra": "Ezr", "nehemiah": "Neh", "neh": "Neh", "esther": "Est", "est": "Est",
    "job": "Job", "psalms": "Psa", "psalm": "Psa", "ps": "Psa",
    "proverbs": "Pro", "prov": "Pro", "ecclesiastes": "Ecc", "ecc": "Ecc",
    "song of solomon": "Sng", "song": "Sng", "isaiah": "Isa", "isa": "Isa",
    "jeremiah": "Jer", "jer": "Jer", "lamentations": "Lam", "lam": "Lam",
    "ezekiel": "Ezk", "ezk": "Ezk", "daniel": "Dan", "dan": "Dan",
    "hosea": "Hos", "hos": "Hos", "joel": "Jol", "jol": "Jol",
    "amos": "Amo", "obadiah": "Oba", "jonah": "Jon", "micah": "Mic",
    "nahum": "Nam", "habakkuk": "Hab", "zephaniah": "Zep", "haggai": "Hag",
    "zechariah": "Zec", "malachi": "Mal",
    "matthew": "Mat", "matt": "Mat", "mark": "Mrk", "mrk": "Mrk",
    "luke": "Luk", "luk": "Luk", "john": "Jhn", "jhn": "Jhn",
    "acts": "Act", "act": "Act", "romans": "Rom", "rom": "Rom",
    "1 corinthians": "1Co", "1cor": "1Co", "2 corinthians": "2Co", "2cor": "2Co",
    "galatians": "Gal", "gal": "Gal", "ephesians": "Eph", "eph": "Eph",
    "philippians": "Php", "phil": "Php", "colossians": "Col", "col": "Col",
    "1 thessalonians": "1Th", "1thess": "1Th", "2 thessalonians": "2Th", "2thess": "2Th",
    "1 timothy": "1Ti", "1tim": "1Ti", "2 timothy": "2Ti", "2tim": "2Ti",
    "titus": "Tit", "philemon": "Phm", "hebrews": "Heb", "heb": "Heb",
    "james": "Jas", "jas": "Jas", "1 peter": "1Pe", "1pet": "1Pe",
    "2 peter": "2Pe", "2pet": "2Pe", "1 john": "1Jn", "1jn": "1Jn",
    "2 john": "2Jn", "2jn": "2Jn", "3 john": "3Jn", "3jn": "3Jn",
    "jude": "Jud", "revelation": "Rev", "rev": "Rev"
}

_NORMALIZE_RE = re.compile(r'^\s*(?P<prefix>[1-3]\s+)?(?P<book>[A-Za-z0-9\.\' ]+?)\s*(?P<chap>\d+)\s*:\s*(?P<verse>\d+)\s*$', re.IGNORECASE)
_MIXED_RE = re.compile(r'([A-Za-z\.]+)(\d)')

def query_db_logic(normalized_verse: str, limit: Optional[int] = None) -> List[Tuple[str, int]]:
    conn = get_db_connection()
    cur = conn.cursor()

    # Start with the normalized version
    candidates = [normalized_verse]

    # Split into book + chapter:verse
    parts = normalized_verse.split()
    if len(parts) >= 2:
        book_token = parts[0]          # e.g. "1Pe", "1Pet", "1Peter"
        chapvers = " ".join(parts[1:]) # e.g. "1:1"

        # -----------------------------------------
        # A) FULL BOOK NAME (reverse BOOK_ABBREVIATIONS)
        # -----------------------------------------
        for key, val in BOOK_ABBREVIATIONS.items():
            if val.lower() == book_token.lower():
                full_name = " ".join(w.capitalize() for w in key.split())
                candidates.append(f"{full_name} {chapvers}")

        # -----------------------------------------
        # B) PET / PE ALTERNATES
        # -----------------------------------------
        alt_map = {
            "1pe": "1Pet", "2pe": "2Pet", "3pe": "3Pet",
            "1pet": "1Pe", "2pet": "2Pe", "3pet": "3Pe",
            "1peter": "1Pet", "2peter": "2Pet", "3peter": "3Pet"
        }

        lower_book = book_token.lower()
        if lower_book in alt_map:
            candidates.append(f"{alt_map[lower_book]} {chapvers}")

        # -----------------------------------------
        # C) NUMBER-SPACE-NAME (WHAT YOUR DB ACTUALLY USES)
        # "1Pet" → "1 Pet"
        # "1Jn"  → "1 Jn"
        # -----------------------------------------
        m = re.match(r"^([123])(\w+)$", book_token)
        if m:
            num, name = m.groups()
            spaced = f"{num} {name.capitalize()}"
            candidates.append(f"{spaced} {chapvers}")

        # -----------------------------------------
        # D) If someone sends "1 Peter 1:1", normalize spacing
        # -----------------------------------------
        m2 = re.match(r"^([123])\s*([A-Za-z]+)$", book_token)
        if m2:
            num, name = m2.groups()
            spaced2 = f"{num} {name.capitalize()}"
            candidates.append(f"{spaced2} {chapvers}")

    # Remove duplicates but keep order
    candidates = list(dict.fromkeys(candidates))

    # Query database for each candidate
    results_map = {}

    for cand in candidates:
        sql = "SELECT cross_ref, votes FROM cross_references WHERE verse = ?"
        params = [cand]

        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = cur.execute(sql, tuple(params)).fetchall()

        for row in rows:
            cref = row["cross_ref"]
            votes = row["votes"]

            # Keep highest vote count for each unique reference
            if cref not in results_map or votes > results_map[cref]:
                results_map[cref] = votes

    conn.close()

    # Return sorted list (highest vote first)
    return sorted(results_map.items(), key=lambda x: x[1], reverse=True)

def normalize_verse(raw: str) -> str:
    """
    Normalize user input so it matches the exact DB format created from OpenBible:
        Gen.1.1   -> Gen 1:1
        Jas.1.1   -> Jas 1:1
        Jude.1.1  -> Jude 1:1
        1Pet.1.1  -> 1 Pet 1:1
        1John1:2  -> 1 John 1:2
    """

    raw = raw.strip().lower()

    # Abbreviations used by your DB
    book_abbrev = {
        # --- Pentateuch ---
        "genesis": "Gen", "gen": "Gen",
        "exodus": "Exod", "exo": "Exod", "ex": "Exod",
        "leviticus": "Lev", "lev": "Lev",
        "numbers": "Num", "num": "Num",
        "deuteronomy": "Deut", "deu": "Deut",

        # --- History Books ---
        "joshua": "Josh", "jos": "Josh",
        "judges": "Judg",
        "ruth": "Ruth",
        "1samuel": "1 Sam", "1 samuel": "1 Sam", "1sam": "1 Sam", "1 sam": "1 Sam",
        "2samuel": "2 Sam", "2 samuel": "2 Sam", "2sam": "2 Sam", "2 sam": "2 Sam",
        "1kings": "1 Kgs", "1 kings": "1 Kgs",
        "2kings": "2 Kgs", "2 kings": "2 Kgs",
        "1chronicles": "1 Chr", "1 chronicles": "1 Chr", "1chr": "1 Chr", "1 chr": "1 Chr",
        "2chronicles": "2 Chr", "2 chronicles": "2 Chr", "2chr": "2 Chr", "2 chr": "2 Chr",

        "ezra": "Ezra",

        "nehemiah": "Neh", "neh": "Neh",

        "esther": "Esth", "esth": "Esth",

        # --- Wisdom ---
        "job": "Job",
        "psalms": "Ps", "psalm": "Ps", "ps": "Ps",
        "proverbs": "Prov", "pro": "Prov",
        "ecclesiastes": "Eccl", "ecc": "Eccl",
        "songofsolomon": "Song", "song of solomon": "Song",

        # --- Major Prophets ---
        "isaiah": "Isa",
        "jeremiah": "Jer",
        "lamentations": "Lam",
        "ezekiel": "Ezek",
        "daniel": "Dan",

        # --- Minor Prophets ---
        "hosea": "Hos",
        "joel": "Joel",
        "amos": "Amos",
        "obadiah": "Obad",
        "jonah": "Jon",
        "micah": "Mic",
        "nahum": "Nah",
        "habakkuk": "Hab",
        "zephaniah": "Zeph",
        "haggai": "Hag",
        "zechariah": "Zech",
        "malachi": "Mal",

        # --- Gospels ---
        "matthew": "Matt",
        "mark": "Mark",
        "luke": "Luke",
        "john": "John",

        # --- Paul ---
        "acts": "Acts",
        "romans": "Rom",
        "1corinthians": "1 Cor", "1 corinthians": "1 Cor",
        "2corinthians": "2 Cor", "2 corinthians": "2 Cor",
        "galatians": "Gal",
        "ephesians": "Eph",
        "philippians": "Phil",
        "colossians": "Col",
        "1thessalonians": "1 Thess", "1 thessalonians": "1 Thess",
        "2thessalonians": "2 Thess", "2 thessalonians": "2 Thess",
        "1timothy": "1 Tim",
        "2timothy": "2 Tim",
        "titus": "Titus",
        "philemon": "Phlm",

        # --- General Letters ---
        "hebrews": "Heb",
        "james": "Jas",
        "1peter": "1 Pet", "2peter": "2 Pet",
        "1john": "1 John", "2john": "2 John", "3john": "3 John",
        "jude": "Jude",

        # --- Revelation ---
        "revelation": "Rev", "rev": "Rev"
    }

    # --- Numbered books ---
    m = re.match(r"^\s*([123])\s*([a-z]+)\s*(\d+)\s*[:.]\s*(\d+)\s*$", raw)
    if m:
        num, book, chap, verse = m.groups()
        key = f"{num}{book}"
        if key in book_abbrev:
            return f"{book_abbrev[key]} {chap}:{verse}"
        else:
            return f"{num} {book.capitalize()} {chap}:{verse}"

    # --- Non-numbered books ---
    m2 = re.match(r"^\s*([a-z]+)\s*(\d+)\s*[:.]\s*(\d+)\s*$", raw)
    if m2:
        book, chap, verse = m2.groups()
        if book in book_abbrev:
            return f"{book_abbrev[book]} {chap}:{verse}"
        else:
            return f"{book.capitalize()} {chap}:{verse}"

    # --- fallback ---
    return raw

def query_db_logic(normalized_verse: str, limit: Optional[int] = None) -> List[Tuple[str,int]]:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        candidates = [normalized_verse]

        # Expand abbreviation back to full title for fallback
        # (The logic here matches your uploaded file)
        parts = normalized_verse.split()
        if len(parts) >= 2:
            book_token = parts[0]
            chapvers = ' '.join(parts[1:])
            rev_name = None
            for k,v in BOOK_ABBREVIATIONS.items():
                if v.lower() == book_token.lower():
                    rev_name = k 
                    break
            if rev_name:
                rev_title = ' '.join(w.capitalize() for w in rev_name.split())
                candidates.append(f"{rev_title} {chapvers}")

        candidates = list(dict.fromkeys(candidates))
        results_map = {} 
        
        for cand in candidates:
            query = "SELECT cross_ref, votes FROM cross_references WHERE verse = ? ORDER BY votes DESC"
            params = [cand]
            
            if limit and int(limit) > 0:
                query += " LIMIT ?"
                params.append(int(limit))

            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            for row in rows:
                cr = row["cross_ref"]
                v = int(row["votes"] or 0)
                if cr not in results_map or v > results_map[cr]:
                    results_map[cr] = v

        sorted_results = sorted(results_map.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_results
    finally:
        conn.close()

# CHANGE 2: Endpoints use @router instead of @app
@router.get("/", response_model=CrossRefResponse)
async def get_crossrefs(
    verse: str = Query(..., description="Verse to lookup, e.g. 'Jeremiah 29:11'"),
    limit: Optional[int] = Query(None)
):
    try:
        normalized = normalize_verse(verse)
    except ValueError as e:
        # If it fails to normalize, try using the raw input
        normalized = verse

    results = query_db_logic(normalized, limit=limit)
    items = [CrossRef(verse=normalized, cross_ref=r[0], votes=r[1]) for r in results]

    return CrossRefResponse(query=verse, normalized=normalized, results=items)