from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import re
from typing import List, Tuple, Optional

DB_PATH = "cross_references.db"  # adjust path if your DB is elsewhere

# REPLACE 'app = FastAPI()' WITH THIS:
router = APIRouter()

# Allow CORS from your website only in production; for dev you can allow all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- change "*" to ["https://your-site.com"] for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Pydantic output model ---
class CrossRef(BaseModel):
    verse: str
    cross_ref: str
    votes: int

class CrossRefResponse(BaseModel):
    query: str
    normalized: str
    results: List[CrossRef]

# --- SQLite helper (simple connection-per-request approach) ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# --- Verse normalization ---
# mapping from common full book names (lowercase) to abbreviation used in dataset.
# Extend this map if you discover other book-name variants.
BOOK_ABBREVIATIONS = {
    "genesis": "Gen", "gen": "Gen",
    "exodus": "Exod", "exod": "Exod", "ex": "Exod",
    "leviticus": "Lev", "lev": "Lev",
    "numbers": "Num", "num": "Num",
    "deuteronomy": "Deut", "deut": "Deut",
    "joshua": "Josh", "josh": "Josh",
    "judges": "Judg", "judg": "Judg",
    "ruth": "Ruth",
    "1 samuel": "1Sam", "1samuel": "1Sam", "1 sam": "1Sam", "1sam": "1Sam",
    "2 samuel": "2Sam", "2samuel": "2Sam", "2 sam": "2Sam", "2sam": "2Sam",
    "1 kings": "1Kgs", "1kings": "1Kgs", "1 kgs": "1Kgs",
    "2 kings": "2Kgs", "2kings": "2Kgs", "2 kgs": "2Kgs",
    "1 chronicles": "1Chr", "1chronicles": "1Chr", "1 chr": "1Chr",
    "2 chronicles": "2Chr", "2chronicles": "2Chr", "2 chr": "2Chr",
    "ezra": "Ezra",
    "nehemiah": "Neh", "neh": "Neh",
    "esther": "Esth", "esth": "Esth",
    "job": "Job",
    "psalms": "Ps", "psalm": "Ps", "ps": "Ps",
    "proverbs": "Prov", "prov": "Prov",
    "ecclesiastes": "Eccl", "eccl": "Eccl",
    "song of solomon": "Song", "songofsolomon": "Song", "song": "Song", "songs": "Song",
    "isaiah": "Isa", "isa": "Isa",
    "jeremiah": "Jer", "jer": "Jer",
    "lamentations": "Lam", "lam": "Lam",
    "ezekiel": "Ezek", "ezek": "Ezek",
    "daniel": "Dan", "dan": "Dan",
    "hosea": "Hos", "hos": "Hos",
    "joel": "Joel",
    "amos": "Amos",
    "obadiah": "Obad", "obad": "Obad",
    "jonah": "Jonah", "jon": "Jonah",
    "micah": "Micah", "mic": "Micah",
    "nahum": "Nah", "nah": "Nah",
    "habakkuk": "Hab", "hab": "Hab",
    "zephaniah": "Zeph", "zeph": "Zeph",
    "haggai": "Hag", "hag": "Hag",
    "zechariah": "Zech", "zech": "Zech",
    "malachi": "Mal", "mal": "Mal",
    "matthew": "Matt", "matt": "Matt",
    "mark": "Mark",
    "luke": "Luke",
    "john": "John", "jn": "John", "joh": "John",
    "acts": "Acts",
    "romans": "Rom", "rom": "Rom",
    "1 corinthians": "1Cor", "1corinthians": "1Cor", "1 cor": "1Cor", "1cor": "1Cor",
    "2 corinthians": "2Cor", "2corinthians": "2Cor", "2 cor": "2Cor", "2cor": "2Cor",
    "galatians": "Gal",
    "ephesians": "Eph",
    "philippians": "Phil",
    "colossians": "Col",
    "1 thessalonians": "1Thess", "1thessalonians": "1Thess", "1 thess": "1Thess",
    "2 thessalonians": "2Thess", "2thessalonians": "2Thess",
    "1 timothy": "1Tim", "1timothy": "1Tim",
    "2 timothy": "2Tim", "2timothy": "2Tim",
    "titus": "Titus",
    "philemon": "Phlm", "philem": "Phlm",
    "hebrews": "Heb",
    "james": "James",
    "1 peter": "1Pet", "1peter": "1Pet",
    "2 peter": "2Pet", "2peter": "2Pet",
    "1 john": "1John", "1john": "1John",
    "2 john": "2John", "2john": "2John",
    "3 john": "3John", "3john": "3John",
    "jude": "Jude",
    "revelation": "Rev", "rev": "Rev"
}

# regex to parse input like "1 john 3:16", "John3:16", "Genesis 1:1"
_NORMALIZE_RE = re.compile(r'^\s*(?P<prefix>[1-3]\s+)?(?P<book>[A-Za-z0-9\.\' ]+?)\s*(?P<chap>\d+)\s*:\s*(?P<verse>\d+)\s*$', re.IGNORECASE)
# also handle compact "John3:16" by inserting a space between letters and digits
_MIXED_RE = re.compile(r'([A-Za-z\.]+)(\d)')

def normalize_verse(raw: str) -> str:
    """
    Normalize user input into canonical abbreviated form used by DB, e.g. "Gen 1:1".
    Accepts inputs like "Genesis 1:1", "genesis1:1", "Gen 1:1", "1 John 3:16", "1john3:16".
    Raises ValueError if it cannot parse.
    """
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
    # normalize numeric prefix formatting e.g. "1john" -> "1 john"
    book_key = re.sub(r'^(1|2|3)(\s*)([a-z])', r'\1 \3', book_key)

    # Lookup abbreviation
    abbr = BOOK_ABBREVIATIONS.get(book_key)
    if not abbr:
        # Try a second pass: if book_key has spaces, try only last word (e.g., "song of songs" -> "song")
        last = book_key.split()[-1]
        abbr = BOOK_ABBREVIATIONS.get(last)
    if not abbr:
        # As a last resort, title-case the book and use that (may match DB entries that used full name)
        # e.g. 'Song of Songs' -> 'Song Of Songs 1:1' (less likely to match but better than failing)
        book_title = ' '.join(w.capitalize() for w in book_key.split())
        candidate = f"{book_title} {int(chap)}:{int(verse_num)}"
        return candidate

    # return abbreviated canonical
    return f"{abbr} {int(chap)}:{int(verse_num)}"

# --- Query function ---
def query_crossrefs(normalized_verse: str, limit: Optional[int] = None) -> List[Tuple[str,int]]:
    """
    Query DB for cross-references. Try both the abbreviated canonical form (e.g. 'Gen 1:1')
    and also the original full-title-like form (e.g. 'Genesis 1:1') as fallback.
    Returns unique results ordered by votes (desc).
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        candidates = [normalized_verse]

        # also try a fallback where we expand abbreviation back to a reasonable full title
        # e.g., if normalized_verse is "Gen 1:1" produce "Genesis 1:1" as another candidate
        parts = normalized_verse.split()
        if len(parts) >= 2:
            book_token = parts[0]
            chapvers = ' '.join(parts[1:])
            # try to find a full name mapping for the abbreviation (reverse lookup)
            rev_name = None
            for k,v in BOOK_ABBREVIATIONS.items():
                if v.lower() == book_token.lower():
                    rev_name = k  # this may be normalized lowercase full name
                    break
            if rev_name:
                # title-case the rev_name for a readable full form candidate
                rev_title = ' '.join(w.capitalize() for w in rev_name.split())
                candidates.append(f"{rev_title} {chapvers}")

        # dedupe candidates
        candidates = list(dict.fromkeys(candidates))

        # collect results from DB for each candidate and combine
        results_map = {}  # cross_ref -> votes (max)
        for cand in candidates:
            if limit and int(limit) > 0:
                cur.execute(
                    "SELECT cross_ref, votes FROM cross_references WHERE verse = ? ORDER BY votes DESC LIMIT ?",
                    (cand, int(limit))
                )
            else:
                cur.execute(
                    "SELECT cross_ref, votes FROM cross_references WHERE verse = ? ORDER BY votes DESC",
                    (cand,)
                )
            rows = cur.fetchall()
            for row in rows:
                cr = row["cross_ref"]
                v = int(row["votes"] or 0)
                # keep the maximum votes seen for a cross_ref
                if cr not in results_map or v > results_map[cr]:
                    results_map[cr] = v

        # convert to sorted list by votes desc
        sorted_results = sorted(results_map.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_results
    finally:
        try:
            conn.close()
        except Exception:
            pass

# --- Endpoints ---
@app.get("/", response_class=JSONResponse)
def root():
    return {"ok": True, "message": "OpenBible Crossrefs API. Use /crossrefs?verse=Jeremiah%2029:11"}

@app.get("/crossrefs", response_model=CrossRefResponse)
def get_crossrefs(verse: str = Query(..., description="Verse to lookup, e.g. 'Jeremiah 29:11' or 'john3:16'"),
                  limit: Optional[int] = Query(None, description="Optional max number of cross-refs to return")):
    try:
        normalized = normalize_verse(verse)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = query_crossrefs(normalized, limit=limit)
    items = [CrossRef(verse=normalized, cross_ref=r[0], votes=int(r[1] or 0)) for r in results]

    return CrossRefResponse(query=verse, normalized=normalized, results=items)

# Optional POST endpoint that accepts JSON body { "verse": "Jeremiah 29:11" }
class QueryBody(BaseModel):
    verse: str
    limit: Optional[int] = None

@app.post("/crossrefs", response_model=CrossRefResponse)
def get_crossrefs_post(body: QueryBody):
    return get_crossrefs(verse=body.verse, limit=body.limit)