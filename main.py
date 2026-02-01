from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# --- Semantic search dependencies (FAISS) ---
# These are used by /semantic_refs (and optional semantic search endpoints).
import threading
from typing import Optional, Any, Dict, List

try:
    import numpy as np  # type: ignore
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    # We delay hard failures until the semantic endpoint is called.
    np = None  # type: ignore
    faiss = None  # type: ignore
    SentenceTransformer = None  # type: ignore
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
from cross_reference import router as crossref_router  
from metadata import router as metadata_router
import requests
import json
import os
import sqlite3
import csv
import re



app = FastAPI(title="Bible Unified API")

# ============================================================
# Unified FastAPI app
# ============================================================

app = FastAPI(
    title="Bible Unified API",
    version="1.0.0",
    description=(
        "Single API exposing: Bible text (GitHub JSON), "
        "Interlinear data, and full-text Bible search."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REGISTER THE NEW ROUTER ---
# This means the endpoint will be available at: /crossref/?verse=Gen 1:1
app.include_router(crossref_router, prefix="/crossref", tags=["Cross References"])
app.include_router(metadata_router, prefix="/meta", tags=["Bible Metadata"])

# ============================================================
# 1) BIBLE REST API (from bible-api-main.py)
#    Endpoints:
#      - GET /verse/{version}/{book}/{chapter}/{verse}
#      - GET /chapter/{version}/{book}/{chapter}
# ============================================================

GITHUB_BASE_URL = "https://raw.githubusercontent.com/BenjaminBurnell/Bible/main/bible_data"

BIBLE_CHAPTERS_DB = os.environ.get("BIBLE_CHAPTERS_DB", "/var/data/bible_chapters.sqlite")

BIBLE_BOOK_CODES = {
    "GENESIS": "GEN", "EXODUS": "EXO", "LEVITICUS": "LEV", "NUMBERS": "NUM", "DEUTERONOMY": "DEU",
    "JOSHUA": "JOS", "JUDGES": "JDG", "RUTH": "RUT", "1 SAMUEL": "1SA", "2 SAMUEL": "2SA",
    "1 KINGS": "1KI", "2 KINGS": "2KI", "1 CHRONICLES": "1CH", "2 CHRONICLES": "2CH",
    "EZRA": "EZR", "NEHEMIAH": "NEH", "ESTHER": "EST", "JOB": "JOB", "PSALMS": "PSA",
    "PROVERBS": "PRO", "ECCLESIASTES": "ECC", "SONG OF SOLOMON": "SNG", "ISAIAH": "ISA",
    "JEREMIAH": "JER", "LAMENTATIONS": "LAM", "EZEKIEL": "EZK", "DANIEL": "DAN", "HOSEA": "HOS",
    "JOEL": "JOL", "AMOS": "AMO", "OBADIAH": "OBA", "JONAH": "JON", "MICAH": "MIC", "NAHUM": "NAM",
    "HABAKKUK": "HAB", "ZEPHANIAH": "ZEP", "HAGGAI": "HAG", "ZECHARIAH": "ZEC", "MALACHI": "MAL",
    "MATTHEW": "MAT", "MARK": "MRK", "LUKE": "LUK", "JOHN": "JHN", "ACTS": "ACT", "ROMANS": "ROM",
    "1 CORINTHIANS": "1CO", "2 CORINTHIANS": "2CO", "GALATIANS": "GAL", "EPHESIANS": "EPH",
    "PHILIPPIANS": "PHP", "COLOSSIANS": "COL", "1 THESSALONIANS": "1TH", "2 THESSALONIANS": "2TH",
    "1 TIMOTHY": "1TI", "2 TIMOTHY": "2TI", "TITUS": "TIT", "PHILEMON": "PHM", "HEBREWS": "HEB",
    "JAMES": "JAS", "1 PETER": "1PE", "2 PETER": "2PE", "1 JOHN": "1JN", "2 JOHN": "2JN",
    "3 JOHN": "3JN", "JUDE": "JUD", "REVELATION": "REV"
}

def _normalize_bible_book(book: str) -> str:
    b = (book or "").strip().upper()

    # allow formats like "1john" or "1 john"
    b = re.sub(r"^([123])\s*(.+)$", r"\1 \2", b)
    b = re.sub(r"\s+", " ", b)

    # allow short codes
    if b in BIBLE_BOOK_CODES.values():
        return b
    return BIBLE_BOOK_CODES.get(b, b)

# ============================================================
# PEOPLE PER VERSE (Stephenson persons.csv + person_verse.csv)
# Uses: metadata.db (built by build_metadata_db.py)
# Endpoint:
#   - GET /people_verse/{book}/{chapter:int}/{verse:int}
# ============================================================

ROOT_DIR = os.path.dirname(__file__)
METADATA_DB_PATH = (
    os.environ.get("METADATA_DB_PATH")
    or os.environ.get("METADATA_DB")
    or ("/var/data/metadata.db" if os.path.isfile("/var/data/metadata.db") else os.path.join(ROOT_DIR, "metadata.db"))
)

def _metadata_conn() -> sqlite3.Connection:
    if not os.path.isfile(METADATA_DB_PATH):
        raise HTTPException(
            status_code=500,
            detail=(
                f"metadata.db missing at {METADATA_DB_PATH}. "
                "Build it with build_metadata_db.py and place it on disk, or set METADATA_DB_PATH."
            ),
        )
    conn = sqlite3.connect(METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/people_verse/{book}/{chapter:int}/{verse:int}")
def get_people_for_verse(book: str, chapter: int, verse: int):
    book_code = _normalize_bible_book(book)

    conn = _metadata_conn()
    try:
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT
                    p.id,
                    p.name,
                    p.description,
                    p.sex,
                    p.tribe,
                    p.unique_attribute,
                    vp.role
                FROM verse_people vp
                JOIN people p ON p.id = vp.person_id
                WHERE vp.book = ? AND vp.chapter = ? AND vp.verse = ?
                ORDER BY p.name COLLATE NOCASE
                """,
                (book_code, int(chapter), int(verse)),
            ).fetchall()
        except sqlite3.OperationalError as e:
            raise HTTPException(status_code=500, detail=f"Metadata DB schema error: {e}")

        people = [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"] or "",
                "sex": r["sex"] or "",
                "tribe": r["tribe"] or "",
                "unique_attribute": r["unique_attribute"] or "",
                "role": r["role"] or "",
            }
            for r in rows
        ]

        return {
            "reference": f"{book_code} {chapter}:{verse}",
            "book": book_code,
            "chapter": int(chapter),
            "verse": int(verse),
            "count": len(people),
            "people": people,
        }
    finally:
        conn.close()

PLACES_DB_PATH = (
    os.environ.get("PLACES_DB_PATH")
    or ("/var/data/places.db" if os.path.isfile("/var/data/places.db") else os.path.join(os.path.dirname(__file__), "data", "places.db"))
)

def _places_conn() -> sqlite3.Connection:
    if not os.path.isfile(PLACES_DB_PATH):
        raise HTTPException(status_code=500, detail=f"places.db missing at {PLACES_DB_PATH}")
    conn = sqlite3.connect(PLACES_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/places_verse/{book}/{chapter:int}/{verse:int}")
def get_places_for_verse(book: str, chapter: int, verse: int):
    book_code = _normalize_bible_book(book)

    conn = _places_conn()
    try:
        rows = conn.execute(
            """
            SELECT
              p.place_id,
              p.place_name,
              p.place_type,
              p.modern_equivalent,
              p.place_notes,
              p.openbible_id,
              p.openbible_url,
              p.name_instance,
              p.place_sequence,
              vp.place_label_id,
              vp.place_label,
              vp.place_label_count,
              vp.place_verse_sequence,
              vp.place_verse_notes
            FROM verse_places vp
            JOIN places p ON p.place_id = vp.place_id
            WHERE vp.book = ? AND vp.chapter = ? AND vp.verse = ?
            ORDER BY vp.place_verse_sequence ASC, p.place_name COLLATE NOCASE
            """,
            (book_code, int(chapter), int(verse)),
        ).fetchall()

        places = []
        for r in rows:
            places.append({
                "place_id": r["place_id"],
                "name": r["place_name"] or "",
                "type": r["place_type"] or "",
                "modern_equivalent": r["modern_equivalent"] or "",
                "notes": r["place_notes"] or "",
                "openbible_id": r["openbible_id"] or "",
                "openbible_url": r["openbible_url"] or "",
                "name_instance": int(r["name_instance"] or 0),
                "place_sequence": int(r["place_sequence"] or 0),
                "label_id": r["place_label_id"] or "",
                "label": r["place_label"] or "",
                "label_count": int(r["place_label_count"] or 0),
                "place_verse_sequence": int(r["place_verse_sequence"] or 0),
                "place_verse_notes": r["place_verse_notes"] or "",
            })

        return {
            "reference": f"{book_code} {chapter}:{verse}",
            "book": book_code,
            "chapter": int(chapter),
            "verse": int(verse),
            "count": len(places),
            "places": places,
        }
    finally:
        conn.close()


@app.get("/place_refs/{place_id_or_name}")
def get_place_references(
    place_id_or_name: str,
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """
    Return all verse references that a specific place appears in.

    - `place_id_or_name` can be a numeric `places.place_id` (recommended)
      OR an exact place name match (case-insensitive).
    - Results use 3-letter USFM book codes (e.g., GEN 1:1).
    """
    conn = _places_conn()
    try:
        cur = conn.cursor()

        # 1) Resolve place record
        # Try numeric ID first; if not numeric, fall back to name match.
        place = None
        resolved_id = None

        if place_id_or_name.strip().isdigit():
            resolved_id = int(place_id_or_name.strip())
            place = cur.execute(
                """
                SELECT place_id, place_name, place_type, modern_equivalent,
                       place_notes, openbible_id, openbible_url,
                       name_instance, place_sequence
                FROM places
                WHERE place_id = ?
                LIMIT 1
                """,
                (resolved_id,),
            ).fetchone()
        else:
            place = cur.execute(
                """
                SELECT place_id, place_name, place_type, modern_equivalent,
                       place_notes, openbible_id, openbible_url,
                       name_instance, place_sequence
                FROM places
                WHERE lower(place_name) = lower(?)
                LIMIT 1
                """,
                (place_id_or_name.strip(),),
            ).fetchone()
            resolved_id = int(place["place_id"]) if place else None

        # If we didn't find a place record, still allow querying by raw id-like string
        if resolved_id is None:
            # try coercing if possible, else 404
            try:
                resolved_id = int(place_id_or_name.strip())
            except Exception:
                raise HTTPException(status_code=404, detail="Place not found by id or exact name.")

        # 2) Fetch verse refs (dedupe rows)
        rows = cur.execute(
            """
            SELECT DISTINCT book, chapter, verse
            FROM verse_places
            WHERE place_id = ?
            ORDER BY book, chapter, verse
            LIMIT ? OFFSET ?
            """,
            (resolved_id, limit, offset),
        ).fetchall()

        if not rows and not place:
            raise HTTPException(status_code=404, detail="Place not found (no place record, no verse links).")

        verses = []
        for r in rows:
            ref = f"{r['book']} {int(r['chapter'])}:{int(r['verse'])}"
            verses.append(
                {
                    "book": r["book"],
                    "chapter": int(r["chapter"]),
                    "verse": int(r["verse"]),
                    "reference": ref,
                }
            )

        return {
            "place": None if not place else {
                "place_id": int(place["place_id"]),
                "place_name": place["place_name"] or "",
                "place_type": place["place_type"] or "",
                "modern_equivalent": place["modern_equivalent"] or "",
                "place_notes": place["place_notes"] or "",
                "openbible_id": place["openbible_id"] or "",
                "openbible_url": place["openbible_url"] or "",
                "name_instance": int(place["name_instance"] or 0),
                "place_sequence": int(place["place_sequence"] or 0),
            },
            "place_id": resolved_id,
            "count": len(verses),
            "limit": limit,
            "offset": offset,
            "verses": verses,
        }

    finally:
        conn.close()
        
def _fetch_chapter_from_sqlite(version: str, book_code: str, chapter: int) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(BIBLE_CHAPTERS_DB):
        raise HTTPException(status_code=500, detail=f"NASB2020 DB missing at {BIBLE_CHAPTERS_DB}")

    conn = sqlite3.connect(BIBLE_CHAPTERS_DB)
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT verses_json FROM chapters WHERE version=? AND book=? AND chapter=?",
            (version.upper(), book_code, int(chapter)),
        ).fetchone()

        if not row:
            return None

        verses = json.loads(row[0] or "[]")
        return {
            "version": version.upper(),
            "book": book_code,
            "chapter": int(chapter),
            "verses": verses,
        }
    finally:
        conn.close()

def _fetch_chapter_json(version: str, book: str, chapter: int) -> Dict[str, Any]:
    version = version.upper()
    book_code = _normalize_bible_book(book)

    # ✅ ONLY NASB2020 comes from SQLite
    if version == "NASB2020":
        data = _fetch_chapter_from_sqlite(version, book_code, chapter)
        if data:
            return data
        raise HTTPException(status_code=404, detail="Chapter not found in NASB2020 SQLite DB")

    # ✅ Everything else uses your existing GitHub repo behavior
    url = f"{GITHUB_BASE_URL}/{version}/{book_code}/{chapter}.json"
    res = requests.get(url)
    if res.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Chapter not found at {url}")

    try:
        return json.loads(res.text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing JSON")

@app.get("/verse/{version}/{book}/{chapter}/{verse}")
def get_verse(version: str, book: str, chapter: int, verse: int):
    data = _fetch_chapter_json(version, book, chapter)
    verses = data.get("verses", [])
    for v in verses:
        if str(v.get("verse")) == str(verse):
            return {
                "version": data.get("version", version.upper()),
                "book": data.get("book", _normalize_bible_book(book)),
                "chapter": chapter,
                "verse": verse,
                "text": v.get("text"),
            }
    raise HTTPException(status_code=404, detail="Verse not found")

@app.get("/chapter/{version}/{book}/{chapter}")
def get_chapter(version: str, book: str, chapter: int):
    data = _fetch_chapter_json(version, book, chapter)
    return {
        "version": data.get("version", version.upper()),
        "book": data.get("book", _normalize_bible_book(book)),
        "chapter": data.get("chapter", chapter),
        "verses": data.get("verses", []),
    }

# ============================================================
# 2) BIBLE SEARCH API (from bible-search-api-main.py)
#    Endpoints:
#      - GET /healthz
#      - GET /search
#    Uses sqlite FTS in bible.db
# ============================================================

# Use distinct names to avoid collisions
SEARCH_DB_PATH = "bible.db"
SEARCH_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
EXTRA_SYNONYMS_PATH = "synonyms.json"

def _search_is_simple_word(s: str) -> bool:
    return re.fullmatch(r"[a-z0-9']+", s) is not None

def _search_normalize_phrase(s: str) -> str:
    s = s.replace("-", " ")
    return " ".join(s.split())

def _search_quote_phrase(s: str) -> str:
    s = s.replace('"', '""')
    return f"\"{s}\""

SYNONYMS: Dict[str, List[str]] = {
    # (same defaults as your original search API)
    "love": ["charity","beloved","lovingkindness","loveth"],
    "faith": ["belief","trust","believe","faithful"],
    "hope": ["expectation"],
    "grace": ["favor","mercy","lovingkindness"],
    "repent": ["repentance","turn","return"],
    "forgive": ["forgiveness","pardon","remit"],
    "sin": ["iniquity","transgression","evil","wickedness"],
    "righteous": ["upright","just","holiness","godly"],
    "holy": ["holiness","sanctify","sanctification"],
    "wisdom": ["wise","prudence","understanding","discernment","knowledge"],
    "peace": ["shalom","rest","quietness"],
    "joy": ["gladness","rejoice","rejoicing"],
    "courage": ["be strong","fear not","bold","boldness"],
    "fear": ["afraid","terror","dread","tremble"],
    "anxiety": ["anxious","care","worry","troubled","fearful"],
    "depression": ["downcast","cast down","heavy","brokenhearted","contrite"],
    "anger": ["wrath","rage","indignation"],
    "pride": ["haughty","arrogant","boast","lofty"],
    "humility": ["humble","lowly","meek","meekness"],
    "lust": ["adultery","fornication","sexual immorality","impurity","unclean"],
    "marriage": ["husband","wife","spouse","bride","bridegroom"],
    "money": ["wealth","riches","mammon","covetousness","greed","gold","silver"],
    "generosity": ["give","alms","liberal","share"],
    "work": ["labor","toil","diligent","slothful","idle"],
    "gossip": ["slander","backbite","talebearer","whisperer"],
    "lies": ["lying","falsehood","deceit","deceive"],
    "idolatry": ["idol","graven image","serve other gods"],
    "persecution": ["persecute","revile","tribulation","affliction","oppress","suffer","suffering"],
    "suffering": ["affliction","tribulation","trouble","trial","testing"],
    "addiction": ["bondage","enslaved","slave","mastered","captivity"],
    "oppression": ["oppress","violence","injustice","extortion"],
    "drugs": [
        "pharmakeia","sorcery","witchcraft","enchantments","magic",
        "poison","spell","divination",
        "drunkenness","strong drink","wine","intoxicated","sober","sober-minded"
    ],
    "alcohol": ["wine","strong drink","drunkenness","sober","sober-minded","intoxicated"],
    "sober": ["sober-minded","watchful","vigilant"],
}

def _load_extra_synonyms():
    if os.path.exists(EXTRA_SYNONYMS_PATH):
        try:
            with open(EXTRA_SYNONYMS_PATH, "r", encoding="utf-8") as f:
                extra = json.load(f)
            for k, vals in extra.items():
                key = (k or "").strip().lower()
                if not key:
                    continue
                base = [t.lower().strip() for t in SYNONYMS.get(key, [])]
                more = [str(v).lower().strip() for v in (vals or [])]
                seen = set()
                merged = []
                for x in base + more:
                    if x and x not in seen:
                        merged.append(x)
                        seen.add(x)
                SYNONYMS[key] = merged
        except Exception as e:
            print(f"Could not load {EXTRA_SYNONYMS_PATH}: {e}")

_load_extra_synonyms()

def _search_connect_db():
    conn = sqlite3.connect(SEARCH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _search_get_available_translations(conn) -> List[str]:
    try:
        row = conn.execute("SELECT v FROM meta WHERE k='translations'").fetchone()
        return row["v"].split(",") if row and row["v"] else []
    except:
        return []

def _search_expand_term(term: str, synonyms_map: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []

    def add_atom(s: str):
        s = (s or "").lower().strip()
        if not s:
            return
        if _search_is_simple_word(s):
            out.append(f"{s}*")
        else:
            out.append(_search_quote_phrase(_search_normalize_phrase(s)))

    add_atom(term)
    for syn in synonyms_map.get(term.lower(), []):
        add_atom(syn)

    return out

def _build_fts_query(user_query: str, use_or: bool = True) -> str:
    terms = [t.lower() for t in SEARCH_TOKEN_RE.findall(user_query or "")]
    expanded_atoms: List[str] = []
    for t in terms:
        expanded_atoms.extend(_search_expand_term(t, SYNONYMS))

    seen = set()
    atoms: List[str] = []
    for a in expanded_atoms:
        if a not in seen:
            atoms.append(a)
            seen.add(a)

    if not atoms:
        return ""
    joiner = " OR " if use_or else " AND "
    return joiner.join(atoms)

class SearchResponse(BaseModel):
    references: List[str]

@app.get("/healthz")
def search_healthz():
    try:
        conn = _search_connect_db()
        conn.execute("SELECT COUNT(1) FROM verses LIMIT 1;").fetchone()
        translations = _search_get_available_translations(conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass
    return {"ok": True, "translations": translations}

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query, e.g. 'love'"),
    limit: int = Query(10, ge=1, le=200),
    translation: Optional[str] = Query(None, description="KJV, WEBUS, or omit for both"),
    logic: str = Query("or", pattern="^(or|and)$"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    fts = _build_fts_query(q, use_or=(logic == "or"))
    if not fts:
        return {"references": []}

    try:
        conn = _search_connect_db()
        cur = conn.cursor()

        where = "verses_fts MATCH ?"
        params: List[Any] = [fts]

        if translation:
            where += " AND verses.translation = ?"
            params.append(translation.upper())

        base_sql = f"""
            SELECT DISTINCT verses.reference
            FROM verses
            JOIN verses_fts ON verses_fts.rowid = verses.id
            WHERE {where}
        """

        try:
            sql = base_sql + " ORDER BY bm25(verses_fts) LIMIT ? OFFSET ?"
            rows = cur.execute(sql, params + [limit, offset]).fetchall()
        except sqlite3.OperationalError:
            sql = base_sql + " LIMIT ? OFFSET ?"
            rows = cur.execute(sql, params + [limit, offset]).fetchall()

        refs = [r["reference"] for r in rows]
        return {"references": refs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            conn.close()
        except:
            pass

# ============================================================
# 3) INTERLINEAR API (from interlinear-api-main.py)
#    Endpoints:
#      - GET /health
#      - GET /debug/resolve
#      - GET /books
#      - GET /interlinear/{book}/{chapter:int}
#      - GET /interlinear/{book}/{chapter:int}/{verse:int}
#    Uses: interlinear.sqlite3 + data/*.csv
# ============================================================

BASE_DIR = os.path.dirname(__file__)
INTER_DB_PATH = os.environ.get("INTERLINEAR_DB", os.path.join(BASE_DIR, "interlinear.sqlite3"))
INTER_DATA_DIR = os.path.join(BASE_DIR, "data")
BOOK_CODES_PATH = os.path.join(INTER_DATA_DIR, "book_codes.json")
STRONGS_LEXICON_CSV = os.path.join(INTER_DATA_DIR, "strongs_lexicon.csv")
GREEK_LEXICON_CSV = os.path.join(INTER_DATA_DIR, "greek_lexicon.csv")

FALLBACK_INTER_BOOK_CODES = {
    "GEN":"Genesis","EXO":"Exodus","LEV":"Leviticus","NUM":"Numbers","DEU":"Deuteronomy",
    "JOS":"Joshua","JDG":"Judges","RUT":"Ruth","1SA":"1 Samuel","2SA":"2 Samuel",
    "1KI":"1 Kings","2KI":"2 Kings","1CH":"1 Chronicles","2CH":"2 Chronicles","EZR":"Ezra",
    "NEH":"Nehemiah","EST":"Esther","JOB":"Job","PSA":"Psalms","PRO":"Proverbs","ECC":"Ecclesiastes",
    "SNG":"Song of Solomon","ISA":"Isaiah","JER":"Jeremiah","LAM":"Lamentations","EZK":"Ezekiel",
    "DAN":"Daniel","HOS":"Hosea","JOL":"Joel","AMO":"Amos","OBA":"Obadiah","JON":"Jonah","MIC":"Micah",
    "NAM":"Nahum","HAB":"Habakkuk","ZEP":"Zephaniah","HAG":"Haggai","ZEC":"Zechariah","MAL":"Malachi",
    "MAT":"Matthew","MRK":"Mark","LUK":"Luke","JHN":"John","ACT":"Acts","ROM":"Romans",
    "1CO":"1 Corinthians","2CO":"2 Corinthians","GAL":"Galatians","EPH":"Ephesians","PHP":"Philippians",
    "COL":"Colossians","1TH":"1 Thessalonians","2TH":"2 Thessalonians","1TI":"1 Timothy","2TI":"2 Timothy",
    "TIT":"Titus","PHM":"Philemon","HEB":"Hebrews","JAS":"James","1PE":"1 Peter","2PE":"2 Peter",
    "1JN":"1 John","2JN":"2 John","3JN":"3 John","JUD":"Jude","REV":"Revelation"
}

def _inter_load_book_codes() -> Dict[str, str]:
    try:
        with open(BOOK_CODES_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str,str] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "name" in v:
                out[k.upper()] = v["name"]
            else:
                out[k.upper()] = str(v)
        return out
    except Exception:
        return FALLBACK_INTER_BOOK_CODES.copy()

INTER_BOOK_CODES = _inter_load_book_codes()
INTER_NAME_TO_CODE = {name.lower(): code for code, name in INTER_BOOK_CODES.items()}

def _inter_read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows

def _inter_norm_strong_keys(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s/;]+", raw.strip())
    keys: List[str] = []
    for p in parts:
        if not p:
            continue
        if re.match(r"^[HhGg]\d+$", p):
            prefix = p[0].upper()
            num = re.sub(r"\D", "", p[1:])
            if num:
                keys += [prefix + num, num]
        else:
            num = re.sub(r"\D", "", p)
            if num:
                keys += ["H" + num, "G" + num, num]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

class InterLexicon:
    def __init__(self):
        self.by_strong: Dict[str, Dict[str, str]] = {}
        self.by_lemma: Dict[str, Dict[str, str]] = {}

    def load(self):
        if os.path.isfile(STRONGS_LEXICON_CSV):
            for r in _inter_read_csv(STRONGS_LEXICON_CSV):
                strong = (r.get("strong") or "").strip()
                if strong:
                    entry = {
                        "lemma": (r.get("lemma") or "").strip(),
                        "translit": (r.get("translit") or "").strip(),
                        "gloss": (r.get("gloss") or "").strip(),
                    }
                    for k in _inter_norm_strong_keys(strong):
                        self.by_strong[k] = entry

        if os.path.isfile(GREEK_LEXICON_CSV):
            for r in _inter_read_csv(GREEK_LEXICON_CSV):
                lemma = (r.get("lemma") or "").strip()
                if lemma:
                    self.by_lemma[lemma] = {
                        "lemma": lemma,
                        "translit": (r.get("translit") or "").strip(),
                        "gloss": (r.get("gloss") or "").strip(),
                    }

INTER_LEX = InterLexicon()
INTER_LEX.load()

def _inter_conn():
    conn = sqlite3.connect(INTER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _inter_resolve_book(book_param: str) -> Tuple[str, str]:
    raw = (book_param or "").strip()
    if not raw:
        raise HTTPException(400, "Book is required.")
    up = raw.upper()
    if up in INTER_BOOK_CODES:
        return up, INTER_BOOK_CODES[up]
    low = raw.lower()
    if low in INTER_NAME_TO_CODE:
        code = INTER_NAME_TO_CODE[low]
        return code, INTER_BOOK_CODES[code]
    guess = up[:3]
    if guess in INTER_BOOK_CODES:
        return guess, INTER_BOOK_CODES[guess]
    raise HTTPException(404, f"Unknown book: {book_param}")

def _inter_enrich_token(row: sqlite3.Row) -> Dict[str, Any]:
    surface = (row["surface"] or "")
    lemma   = (row["lemma"] or "")
    transl  = (row["translit"] or "")
    gloss   = (row["gloss"] or "")
    morph   = (row["morph"] or "")
    strong  = (row["strong"] or "")
    idx     = int(row["token_index"])

    if lemma and transl and gloss:
        return {
            "surface": surface, "lemma": lemma, "translit": transl, "gloss": gloss,
            "morph": morph, "strong": strong, "index": idx,
            "resolved_lemma": lemma, "resolved_translit": transl, "resolved_gloss": gloss,
            "translation": gloss,
        }

    resolved: Dict[str, str] = {}
    for k in _inter_norm_strong_keys(strong):
        hit = INTER_LEX.by_strong.get(k)
        if hit:
            resolved = hit
            break
    if not resolved and lemma:
        resolved = INTER_LEX.by_lemma.get(lemma, {})

    r_lemma  = lemma or resolved.get("lemma", "")
    r_transl = transl or resolved.get("translit", "")
    r_gloss  = gloss or resolved.get("gloss", "")

    return {
        "surface": surface, "lemma": lemma, "translit": transl, "gloss": gloss,
        "morph": morph, "strong": strong, "index": idx,
        "resolved_lemma": r_lemma, "resolved_translit": r_transl, "resolved_gloss": r_gloss,
        "translation": r_gloss,
    }

@app.get("/health")
def interlinear_health():
    return {
        "ok": os.path.isfile(INTER_DB_PATH),
        "db": INTER_DB_PATH,
        "data_dir": INTER_DATA_DIR,
        "lexicon_strongs_csv": os.path.isfile(STRONGS_LEXICON_CSV),
        "lexicon_greek_csv": os.path.isfile(GREEK_LEXICON_CSV),
        "strongs_loaded": len(INTER_LEX.by_strong),
        "greek_loaded": len(INTER_LEX.by_lemma),
    }

@app.get("/debug/resolve")
def debug_resolve(strong: str = "", lemma: str = ""):
    hit: Dict[str, Any] = {}
    for k in _inter_norm_strong_keys(strong or ""):
        if k in INTER_LEX.by_strong:
            hit = {"via": f"strong:{k}", **INTER_LEX.by_strong[k]}
            break
    if not hit and lemma:
        if lemma in INTER_LEX.by_lemma:
            hit = {"via": "lemma", **INTER_LEX.by_lemma[lemma]}
    return {"input": {"strong": strong, "lemma": lemma}, "hit": hit}

@app.get("/books")
def list_books():
    with _inter_conn() as c:
        rows = c.execute(
            "SELECT DISTINCT book_code FROM tokens ORDER BY book_code"
        ).fetchall()
    return {
        "books": [
            {"code": r["book_code"], "name": INTER_BOOK_CODES.get(r["book_code"], r["book_code"])}
            for r in rows
        ]
    }

@app.get("/interlinear/{book}/{chapter:int}/{verse:int}")
def get_interlinear_verse(book: str, chapter: int, verse: int):
    code, name = _inter_resolve_book(book)
    with _inter_conn() as c:
        rows = c.execute(
            """
            SELECT surface, lemma, translit, gloss, morph, strong, token_index
            FROM tokens
            WHERE book_code=? AND chapter=? AND verse=?
            ORDER BY token_index ASC
            """,
            (code, chapter, verse),
        ).fetchall()
    tokens = [_inter_enrich_token(r) for r in rows]
    return {
        "reference": f"{name} {chapter}:{verse}",
        "book": name,
        "book_code": code,
        "chapter": chapter,
        "verse": verse,
        "tokens": tokens,
    }

@app.get("/interlinear/{book}/{chapter:int}")
def get_interlinear_chapter(book: str, chapter: int):
    code, name = _inter_resolve_book(book)
    with _inter_conn() as c:
        rows = c.execute(
            """
            SELECT verse, token_index, surface, lemma, translit, gloss, morph, strong
            FROM tokens
            WHERE book_code=? AND chapter=?
            ORDER BY verse ASC, token_index ASC
            """,
            (code, chapter),
        ).fetchall()
    verses: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        v = int(r["verse"])
        verses.setdefault(v, []).append(_inter_enrich_token(r))
    return {
        "reference": f"{name} {chapter}",
        "book": name,
        "book_code": code,
        "chapter": chapter,
        "verses": verses,
    }
    
    
# ============================================================
# 4) CROSS REFERENCE API configuration
# ============================================================

CROSSREF_DB_PATH = "cross_references.db"

# Map 3-letter codes (used internally) back to Full Names (used by OpenBible DB)
CODE_TO_FULL_NAME = {
    "GEN":"Genesis", "EXO":"Exodus", "LEV":"Leviticus", "NUM":"Numbers", "DEU":"Deuteronomy",
    "JOS":"Joshua", "JDG":"Judges", "RUT":"Ruth", "1SA":"1 Samuel", "2SA":"2 Samuel",
    "1KI":"1 Kings", "2KI":"2 Kings", "1CH":"1 Chronicles", "2CH":"2 Chronicles", "EZR":"Ezra",
    "NEH":"Nehemiah", "EST":"Esther", "JOB":"Job", "PSA":"Psalms", "PRO":"Proverbs",
    "ECC":"Ecclesiastes", "SNG":"Song of Solomon", "ISA":"Isaiah", "JER":"Jeremiah",
    "LAM":"Lamentations", "EZK":"Ezekiel", "DAN":"Daniel", "HOS":"Hosea", "JOL":"Joel",
    "AMO":"Amos", "OBA":"Obadiah", "JON":"Jonah", "MIC":"Micah", "NAM":"Nahum",
    "HAB":"Habakkuk", "ZEP":"Zephaniah", "HAG":"Haggai", "ZEC":"Zechariah", "MAL":"Malachi",
    "MAT":"Matthew", "MRK":"Mark", "LUK":"Luke", "JHN":"John", "ACT":"Acts", "ROM":"Romans",
    "1CO":"1 Corinthians", "2CO":"2 Corinthians", "GAL":"Galatians", "EPH":"Ephesians",
    "PHP":"Philippians", "COL":"Colossians", "1TH":"1 Thessalonians", "2TH":"2 Thessalonians",
    "1TI":"1 Timothy", "2TI":"2 Timothy", "TIT":"Titus", "PHM":"Philemon", "HEB":"Hebrews",
    "JAS":"James", "1PE":"1 Peter", "2PE":"2 Peter", "1JN":"1 John", "2JN":"2 John",
    "3JN":"3 John", "JUD":"Jude", "REV":"Revelation"
}

def _get_crossref_conn():
    conn = sqlite3.connect(CROSSREF_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/crossref/{book}/{chapter}/{verse}")
def get_cross_references(book: str, chapter: int, verse: int):
    """
    Get cross references for a specific verse.
    """
    # 1. Normalize the Book Name
    # We use your existing _normalize_bible_book to get the 3-letter code (e.g. "jn" -> "JHN")
    # Then we map "JHN" -> "John" because the CrossRef DB uses full names.
    code = _normalize_bible_book(book)
    
    full_name = CODE_TO_FULL_NAME.get(code)
    if not full_name:
        # Fallback: capitalize user input if not found in map
        full_name = book.capitalize()

    # 2. Construct the reference key (e.g., "Genesis 1:1")
    ref_key = f"{full_name} {chapter}:{verse}"
    
    conn = _get_crossref_conn()
    try:
        # 3. Query the DB
        # The import script creates columns: verse, cross_ref, votes
        rows = conn.execute(
            "SELECT cross_ref, votes FROM cross_references WHERE verse = ? ORDER BY votes DESC",
            (ref_key,)
        ).fetchall()
        
        results = []
        for row in rows:
            results.append({
                "ref": row["cross_ref"],
                "votes": row["votes"]
            })
            
        return {
            "source": ref_key,
            "count": len(results),
            "cross_references": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()


# ============================================================
# 4.5) PERSON -> VERSE REFERENCES (from metadata.db)
#     Endpoint:
#       - GET /person_refs/{person_id_or_name}?limit=200&offset=0
# ============================================================

def _get_metadata_db_path() -> str:
    """
    Resolve metadata DB path robustly for both local dev and Render.
    Prefers METADATA_DB_PATH env var, then falls back to ./metadata.db.
    """
    if METADATA_DB_PATH and os.path.isfile(METADATA_DB_PATH):
        return METADATA_DB_PATH
    local = os.path.join(os.path.dirname(__file__), "metadata.db")
    if os.path.isfile(local):
        return local
    return "metadata.db"


def _get_metadata_conn():
    db_path = _get_metadata_db_path()
    if not os.path.isfile(db_path):
        raise HTTPException(status_code=500, detail=f"Metadata DB missing at {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/person_refs/{person_id_or_name}")
def get_person_references(
    person_id_or_name: str,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """
    Return all verse references that a specific person appears in.

    - `person_id_or_name` can be a `people.id` (recommended) OR an exact name match.
    - Results use 3-letter USFM book codes (e.g., GEN 1:1).
    """
    conn = _get_metadata_conn()
    try:
        cur = conn.cursor()

        # 1) Resolve person record
        person = cur.execute(
            """
            SELECT id, name, description, sex, tribe, unique_attribute
            FROM people
            WHERE id = ?
               OR lower(name) = lower(?)
            LIMIT 1
            """,
            (person_id_or_name, person_id_or_name),
        ).fetchone()

        resolved_id = person["id"] if person else person_id_or_name

        # 2) Fetch verse refs (dedupe rows)
        rows = cur.execute(
            """
            SELECT DISTINCT book, chapter, verse, role
            FROM verse_people
            WHERE person_id = ?
            ORDER BY book, chapter, verse
            LIMIT ? OFFSET ?
            """,
            (resolved_id, limit, offset),
        ).fetchall()

        if not rows and not person:
            raise HTTPException(status_code=404, detail="Person not found (no person record, no verse links).")

        verses = []
        for r in rows:
            ref = f"{r['book']} {int(r['chapter'])}:{int(r['verse'])}"
            verses.append(
                {
                    "book": r["book"],
                    "chapter": int(r["chapter"]),
                    "verse": int(r["verse"]),
                    "reference": ref,
                    # In your build_metadata_db.py this field is populated from person_verse_notes
                    "notes": (r["role"] or "").strip(),
                }
            )

        return {
            "person": None
            if not person
            else {
                "id": person["id"],
                "name": person["name"],
                "description": person["description"],
                "sex": person["sex"],
                "tribe": person["tribe"],
                "unique_attribute": person["unique_attribute"],
            },
            "person_id": resolved_id,
            "count": len(verses),
            "limit": limit,
            "offset": offset,
            "verses": verses,
        }
    finally:
        try:
            conn.close()
        except:
            pass


# ============================================================
# 5) Unified root health (optional convenience)
# ============================================================

# =========================
# Semantic search (FAISS)
# =========================

# =========================
# Semantic search (/semantic_refs)
# =========================
#
# This is designed to run on low-memory instances (e.g., Render 512MB):
#   - FAISS index is memory-mapped (read-only)
#   - meta.jsonl is NOT loaded into RAM (we store only line offsets)
#   - torch/transformers are imported only when semantic search is used

import threading
from array import array

SEMANTIC_DIR = os.environ.get("SEMANTIC_DIR", "/var/data/index_all")
SEMANTIC_INDEX_PATH = os.path.join(SEMANTIC_DIR, "index.faiss")
SEMANTIC_META_PATH = os.path.join(SEMANTIC_DIR, "meta.jsonl")

# If your index was built with a different model, set SEMANTIC_MODEL to match.
SEMANTIC_MODEL_NAME = os.environ.get("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_sem_lock = threading.Lock()
_sem_index = None          # FAISS index
_sem_model = None          # SentenceTransformer
_sem_meta_fp = None        # open file handle (rb)
_sem_meta_offsets = None   # array('Q') offsets
np = None                  # set on load


def _load_semantic_assets() -> None:
    global _sem_index, _sem_model, _sem_meta_fp, _sem_meta_offsets, np

    with _sem_lock:
        if _sem_index is not None and _sem_model is not None and _sem_meta_fp is not None and _sem_meta_offsets is not None:
            return

        # Late imports so other endpoints don't pay the torch/transformers cost.
        import numpy as _np
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        np = _np

        if not os.path.exists(SEMANTIC_INDEX_PATH):
            raise RuntimeError(f"Missing FAISS index: {SEMANTIC_INDEX_PATH}")
        if not os.path.exists(SEMANTIC_META_PATH):
            raise RuntimeError(f"Missing meta.jsonl: {SEMANTIC_META_PATH}")

        # Memory-map the FAISS index to avoid loading everything into RAM.
        _sem_index = faiss.read_index(
            SEMANTIC_INDEX_PATH,
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
        )

        # Load embedding model (CPU).
        _sem_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device="cpu")
        # Prevent giant inputs from blowing up RAM.
        try:
            _sem_model.max_seq_length = int(os.environ.get("SEMANTIC_MAX_SEQ_LEN", "256"))
        except Exception:
            pass

        # Build offsets for meta.jsonl (random access without loading all lines).
        _sem_meta_fp = open(SEMANTIC_META_PATH, "rb")
        offsets = array("Q")
        pos = _sem_meta_fp.tell()
        while True:
            line = _sem_meta_fp.readline()
            if not line:
                break
            offsets.append(pos)
            pos = _sem_meta_fp.tell()
        _sem_meta_offsets = offsets


def _embed_query(text: str):
    assert _sem_model is not None
    # SentenceTransformer returns (dim,) for a single string.
    v = _sem_model.encode(text, normalize_embeddings=False)
    return v


def _get_meta_at_index(idx: int) -> Optional[dict]:
    if _sem_meta_fp is None or _sem_meta_offsets is None:
        return None
    if idx < 0 or idx >= len(_sem_meta_offsets):
        return None
    import json
    try:
        _sem_meta_fp.seek(int(_sem_meta_offsets[idx]))
        line = _sem_meta_fp.readline()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except Exception:
        return None


@app.get("/semantic_refs")
async def semantic_refs(
    q: Optional[str] = Query(None, alias="q"),
    query: Optional[str] = Query(None, alias="query"),
    k: int = Query(7, ge=1, le=50, alias="k"),
    topk: Optional[int] = Query(None, ge=1, le=50, alias="topk"),
    minscore: float = Query(0.22, ge=-1.0, le=1.0, alias="minscore"),
    keep_version: bool = Query(False, alias="keep_version"),
):
    """Return a list of best-matching verse refs for a natural-language query.

    Supports both query param styles:
      - ?q=...
      - ?query=...
    """
    text = (q or query or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Missing query text. Provide ?q=... or ?query=...")

    k_final = int(topk if topk is not None else k)

    try:
        _load_semantic_assets()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Semantic search unavailable: {e}")

    assert _sem_index is not None and _sem_model is not None and np is not None

    qvec = _embed_query(text)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    qvec = qvec.reshape(1, -1).astype("float32")

    D, I = _sem_index.search(qvec, k_final)

    # Convert score for L2 indices (lower is better). For IP/cosine, higher is better already.
    metric = getattr(_sem_index, "metric_type", None)
    is_l2 = False
    try:
        import faiss  # type: ignore
        is_l2 = (metric == faiss.METRIC_L2)
    except Exception:
        pass

    results: List[str] = []
    seen = set()

    for rank, idx in enumerate(I[0].tolist()):
        if idx < 0:
            continue

        meta = _get_meta_at_index(idx)
        if not meta:
            continue

        raw_score = float(D[0][rank])
        score = (-raw_score) if is_l2 else raw_score
        if score < float(minscore):
            continue

        refs = meta.get("refs") or []
        for ref in refs:
            if not isinstance(ref, str):
                continue
            out = ref if keep_version else " ".join(ref.split(" ")[1:])
            out = out.strip()
            if not out or out in seen:
                continue
            seen.add(out)
            results.append(out)
            if len(results) >= k_final:
                break
        if len(results) >= k_final:
            break

    return results



@app.get("/")
def home():
    return {
        "message": "Bible API is running",
        "endpoints": {
            "bible": "/verse/{version}/{book}/{chapter}/{verse}",
            "search": "/search?q=query",
            "interlinear": "/interlinear/{book}/{chapter}/{verse}",
            "cross_references": "/crossref/?verse=Gen 1:1",
            "metadata": "/meta/verse?book=GEN&chapter=1&verse=1",
            "people_verse": "/people_verse/{book}/{chapter}/{verse}",
            "person_refs": "/person_refs/{person_id_or_name}",
            "places_verse": "/places_verse/{book}/{chapter}/{verse}"
        }
    }