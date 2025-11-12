# app.py — runtime enrichment version (works even if DB didn't get updated)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Tuple
import sqlite3, os, json, csv, re

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.environ.get("INTERLINEAR_DB", os.path.join(BASE_DIR, "interlinear.sqlite3"))
DATA_DIR = os.path.join(BASE_DIR, "data")
BOOK_CODES_PATH = os.path.join(DATA_DIR, "book_codes.json")
STRONGS_LEXICON_CSV = os.path.join(DATA_DIR, "strongs_lexicon.csv")
GREEK_LEXICON_CSV   = os.path.join(DATA_DIR, "greek_lexicon.csv")

# ---------- Book codes ----------
FALLBACK_BOOK_CODES = {
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

def load_book_codes() -> Dict[str, str]:
    try:
        with open(BOOK_CODES_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "name" in v:
                out[k.upper()] = v["name"]
            else:
                out[k.upper()] = str(v)
        return out
    except Exception:
        return FALLBACK_BOOK_CODES.copy()

BOOK_CODES = load_book_codes()
NAME_TO_CODE = {name.lower(): code for code, name in BOOK_CODES.items()}

# ---------- Lexicon load ----------
def _read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows

def _norm_strong_keys(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s/;]+", raw.strip())
    keys = []
    for p in parts:
        if not p: 
            continue
        if re.match(r"^[HhGg]\d+$", p):
            prefix = p[0].upper(); num = re.sub(r"\D", "", p[1:])
            if num:
                keys += [prefix+num, num]
        else:
            num = re.sub(r"\D", "", p)
            if num:
                keys += ["H"+num, "G"+num, num]
    # dedupe preserving order
    seen = set(); out=[]
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

class Lexicon:
    def __init__(self):
        self.by_strong: Dict[str, Dict[str, str]] = {}
        self.by_lemma: Dict[str, Dict[str, str]] = {}

    def load(self):
        if os.path.isfile(STRONGS_LEXICON_CSV):
            for r in _read_csv(STRONGS_LEXICON_CSV):
                strong = (r.get("strong") or "").strip()
                if strong:
                    entry = {
                        "lemma": (r.get("lemma") or "").strip(),
                        "translit": (r.get("translit") or "").strip(),
                        "gloss": (r.get("gloss") or "").strip(),
                    }
                    for k in _norm_strong_keys(strong):
                        self.by_strong[k] = entry

        if os.path.isfile(GREEK_LEXICON_CSV):
            for r in _read_csv(GREEK_LEXICON_CSV):
                lemma = (r.get("lemma") or "").strip()
                if lemma:
                    self.by_lemma[lemma] = {
                        "lemma": lemma,
                        "translit": (r.get("translit") or "").strip(),
                        "gloss": (r.get("gloss") or "").strip(),
                    }

LEX = Lexicon()
LEX.load()
print(f"[lexicon] strongs loaded: {len(LEX.by_strong)} | greek lemmas loaded: {len(LEX.by_lemma)}")

# ---------- App ----------
app = FastAPI(title="Interlinear Bible API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def resolve_book(book_param: str) -> Tuple[str, str]:
    raw = (book_param or "").strip()
    if not raw:
        raise HTTPException(400, "Book is required.")
    up = raw.upper()
    if up in BOOK_CODES:
        return up, BOOK_CODES[up]
    low = raw.lower()
    if low in NAME_TO_CODE:
        code = NAME_TO_CODE[low]
        return code, BOOK_CODES[code]
    guess = up[:3]
    if guess in BOOK_CODES:
        return guess, BOOK_CODES[guess]
    raise HTTPException(404, f"Unknown book: {book_param}")

def enrich_token(row: sqlite3.Row) -> Dict[str, Any]:
    surface = (row["surface"] or "")
    lemma   = (row["lemma"] or "")
    transl  = (row["translit"] or "")
    gloss   = (row["gloss"] or "")
    morph   = (row["morph"] or "")
    strong  = (row["strong"] or "")
    idx     = int(row["token_index"])

    # Already complete?
    if lemma and transl and gloss:
        return {
            "surface": surface, "lemma": lemma, "translit": transl, "gloss": gloss,
            "morph": morph, "strong": strong, "index": idx,
            "resolved_lemma": lemma, "resolved_translit": transl, "resolved_gloss": gloss,
            "translation": gloss
        }

    # Try Strong's first, then lemma
    resolved = {}
    for k in _norm_strong_keys(strong):
        hit = LEX.by_strong.get(k)
        if hit:
            resolved = hit; break
    if not resolved and lemma:
        resolved = LEX.by_lemma.get(lemma, {})

    r_lemma  = lemma or resolved.get("lemma", "")
    r_transl = transl or resolved.get("translit", "")
    r_gloss  = gloss or resolved.get("gloss", "")

    return {
        "surface": surface, "lemma": lemma, "translit": transl, "gloss": gloss,
        "morph": morph, "strong": strong, "index": idx,
        "resolved_lemma": r_lemma, "resolved_translit": r_transl, "resolved_gloss": r_gloss,
        # your UI wants "the English word being translated"
        "translation": r_gloss
    }

@app.get("/health")
def health():
    return {
        "ok": os.path.isfile(DB_PATH),
        "db": DB_PATH,
        "data_dir": DATA_DIR,
        "lexicon_strongs_csv": os.path.isfile(STRONGS_LEXICON_CSV),
        "lexicon_greek_csv": os.path.isfile(GREEK_LEXICON_CSV),
        "strongs_loaded": len(LEX.by_strong),
        "greek_loaded": len(LEX.by_lemma),
    }

@app.get("/debug/resolve")
def debug_resolve(strong: str = "", lemma: str = ""):
    # try strong then lemma and show what you’d get
    hit = {}
    for k in _norm_strong_keys(strong or ""):
        if k in LEX.by_strong:
            hit = {"via": f"strong:{k}", **LEX.by_strong[k]}
            break
    if not hit and lemma:
        if lemma in LEX.by_lemma:
            hit = {"via": "lemma", **LEX.by_lemma[lemma]}
    return {"input": {"strong": strong, "lemma": lemma}, "hit": hit}

@app.get("/books")
def list_books():
    with get_conn() as c:
        rows = c.execute("SELECT DISTINCT book_code FROM tokens ORDER BY book_code").fetchall()
    return {"books": [{"code": r["book_code"], "name": BOOK_CODES.get(r["book_code"], r["book_code"])} for r in rows]}

@app.get("/interlinear/{book}/{chapter:int}/{verse:int}")
def get_interlinear_verse(book: str, chapter: int, verse: int):
    code, name = resolve_book(book)
    with get_conn() as c:
        rows = c.execute("""
            SELECT surface, lemma, translit, gloss, morph, strong, token_index
            FROM tokens
            WHERE book_code=? AND chapter=? AND verse=?
            ORDER BY token_index ASC
        """, (code, chapter, verse)).fetchall()
    tokens = [enrich_token(r) for r in rows]
    return {"reference": f"{name} {chapter}:{verse}", "book": name, "book_code": code, "chapter": chapter, "verse": verse, "tokens": tokens}

@app.get("/interlinear/{book}/{chapter:int}")
def get_interlinear_chapter(book: str, chapter: int):
    code, name = resolve_book(book)
    with get_conn() as c:
        rows = c.execute("""
            SELECT verse, token_index, surface, lemma, translit, gloss, morph, strong
            FROM tokens
            WHERE book_code=? AND chapter=?
            ORDER BY verse ASC, token_index ASC
        """, (code, chapter)).fetchall()
    verses: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        v = int(r["verse"])
        verses.setdefault(v, []).append(enrich_token(r))
    return {"reference": f"{name} {chapter}", "book": name, "book_code": code, "chapter": chapter, "verses": verses}
