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

def normalize_verse(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("Empty verse")
    s = raw.strip()
    # insert a space between letters and chapter digits when compact form like "John3:16"
    s = _MIXED_RE.sub(r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s).strip()

    m = _NORMALIZE_RE.match(s)
    if not m:
        # try a looser match: maybe user used dot-notation e.g. "Gen.1.1"
        dot_match = re.match(r'^\s*(?P<book>[\w\. ]+?)\.(?P<chap>\d+)\.(?P<verse>\d+)\s*$', s)
        if dot_match:
            book_raw = dot_match.group('book').replace('.', ' ').strip()
            chap = dot_match.group('chap'); verse_num = dot_match.group('verse')
        else:
            raise ValueError(f"Could not parse verse: {raw!r}")
    else:
        prefix = m.group('prefix') or ''
        book_raw = (prefix + (m.group('book') or '')).strip()
        chap = m.group('chap')
        verse_num = m.group('verse')

    book_key = book_raw.lower().replace('.', '').replace('  ', ' ').strip()
    book_key = re.sub(r'^(1|2|3)(\s*)([a-z])', r'\1 \3', book_key)

    abbr = BOOK_ABBREVIATIONS.get(book_key)
    if not abbr:
        last = book_key.split()[-1]
        abbr = BOOK_ABBREVIATIONS.get(last)
    if not abbr:
        # If we can't abbreviate, trust the user's input capitalized
        book_title = ' '.join(w.capitalize() for w in book_key.split())
        return f"{book_title} {int(chap)}:{int(verse_num)}"

    return f"{abbr} {int(chap)}:{int(verse_num)}"

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