from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
import requests
import json
import os
import sqlite3
import csv
import re

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
    allow_origins=["*"],          # tighten later for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 1) BIBLE REST API (from bible-api-main.py)
#    Endpoints:
#      - GET /verse/{version}/{book}/{chapter}/{verse}
#      - GET /chapter/{version}/{book}/{chapter}
# ============================================================

GITHUB_BASE_URL = "https://raw.githubusercontent.com/BenjaminBurnell/Bible/main/bible_data"

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
    b = book.upper()
    # allow short codes
    if b in BIBLE_BOOK_CODES.values():
        return b
    return BIBLE_BOOK_CODES.get(b, b)

def _fetch_chapter_json(version: str, book: str, chapter: int) -> Dict[str, Any]:
    version = version.upper()
    book_code = _normalize_bible_book(book)
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
# 4) Unified root health (optional convenience)
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Bible Unified API",
        "ok": True,
        "endpoints": {
            "bible": ["/verse/{version}/{book}/{chapter}/{verse}",
                      "/chapter/{version}/{book}/{chapter}"],
            "search": ["/search", "/healthz"],
            "interlinear": [
                "/interlinear/{book}/{chapter}", "/interlinear/{book}/{chapter}/{verse}",
                "/books", "/debug/resolve", "/health"
            ],
        },
    }
