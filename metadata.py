import sqlite3
import json
import traceback
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Union

router = APIRouter()
DB_PATH = "metadata.db"

# --- 1. Standard Book Mapping ---
BOOK_MAP = {
    "GENESIS": "GEN", "EXODUS": "EXO", "LEVITICUS": "LEV", "NUMBERS": "NUM", "DEUTERONOMY": "DEU",
    "JOSHUA": "JOS", "JUDGES": "JDG", "RUTH": "RUT", "1 SAMUEL": "1SA", "2 SAMUEL": "2SA",
    "1 KINGS": "1KI", "2 KINGS": "2KI", "1 CHRONICLES": "1CH", "2 CHRONICLES": "2CH",
    "EZRA": "EZR", "NEHEMIAH": "NEH", "ESTHER": "EST", "JOB": "JOB", "PSALMS": "PSA",
    "PROVERBS": "PRO", "ECCLESIASTES": "ECC", "SONG OF SOLOMON": "SNG", "ISAIAH": "ISA",
    "JEREMIAH": "JER", "LAMENTATIONS": "LAM", "EZEKIEL": "EZK", "DANIEL": "DAN",
    "HOSEA": "HOS", "JOEL": "JOL", "AMOS": "AMO", "OBADIAH": "OBA", "JONAH": "JON",
    "MICAH": "MIC", "NAHUM": "NAM", "HABAKKUK": "HAB", "ZEPHANIAH": "ZEP", "HAGGAI": "HAG",
    "ZECHARIAH": "ZEC", "MALACHI": "MAL", 
    "MATTHEW": "MAT", "MARK": "MRK", "LUKE": "LUK", "JOHN": "JHN", "ACTS": "ACT",
    "ROMANS": "ROM", "1 CORINTHIANS": "1CO", "2 CORINTHIANS": "2CO", "GALATIANS": "GAL",
    "EPHESIANS": "EPH", "PHILIPPIANS": "PHP", "COLOSSIANS": "COL",
    "1 THESSALONIANS": "1TH", "2 THESSALONIANS": "2TH", "1 TIMOTHY": "1TI", "2 TIMOTHY": "2TI",
    "TITUS": "TIT", "PHILEMON": "PHM", "HEBREWS": "HEB", "JAMES": "JAS",
    "1 PETER": "1PE", "2 PETER": "2PE", "1 JOHN": "1JN", "2 JOHN": "2JN", "3 JOHN": "3JN",
    "JUDE": "JUD", "REVELATION": "REV"
}

def normalize_book_code(book_name: str) -> str:
    raw = (book_name or "").strip().upper()
    if raw in BOOK_MAP.values(): return raw
    if raw in BOOK_MAP: return BOOK_MAP[raw]
    simplified = raw.replace(".", "").replace(" ", "")
    for name, code in BOOK_MAP.items():
        if name.replace(" ", "") == simplified:
            return code
    return raw[:3]

# --- Models ---
class AuthorProfile(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    sex: Optional[str] = None
    tribe: Optional[str] = None
    unique_attribute: Optional[str] = None

class Entity(BaseModel):
    id: Union[str, int]  
    name: Optional[str] = "Unknown"
    description: Optional[str] = None
    role: Optional[str] = None

class BookMeta(BaseModel):
    title: str
    date_written: Optional[str]
    place_written: Optional[str]
    audience: Optional[str]
    hebrew_meaning: Optional[str] = None # <--- NEW FIELD
    author_info: Optional[AuthorProfile] = None

class MetaVContext(BaseModel):
    who: List[str] = []
    where: List[str] = []

class VerseMetaResponse(BaseModel):
    reference: str
    book_meta: Optional[BookMeta]
    context: Optional[MetaVContext]
    people: List[Entity] = []
    places: List[Entity] = []

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@router.get("/verse", response_model=VerseMetaResponse)
def get_verse_metadata(book: str, chapter: int, verse: int):
    book_code = normalize_book_code(book)
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # A. Fetch Book Metadata
        book_row = cursor.execute("SELECT * FROM books WHERE code = ?", (book_code,)).fetchone()
        book_meta = None
        
        if book_row:
            # Fetch Author Profile
            author_data = None
            writer_id = book_row['writer_id']
            if writer_id:
                auth_row = cursor.execute("SELECT * FROM people WHERE id = ?", (writer_id,)).fetchone()
                if auth_row:
                    author_data = AuthorProfile(
                        id=auth_row['id'],
                        name=auth_row['name'],
                        description=auth_row['description'],
                        sex=auth_row['sex'],
                        tribe=auth_row['tribe'],
                        unique_attribute=auth_row['unique_attribute']
                    )
            
            # Construct BookMeta
            book_meta = BookMeta(
                title=book_row['title'] or book,
                date_written=book_row['date_written'],
                place_written=book_row['place_written'],
                audience=book_row['audience'],
                hebrew_meaning=book_row['hebrew_meaning'], # <--- NEW
                author_info=author_data
            )

        # B. Fetch MetaV Context
        ctx_row = cursor.execute(
            "SELECT * FROM metav_context WHERE book=? AND chapter=? AND verse=?", 
            (book_code, chapter, verse)
        ).fetchone()
        
        meta_context = MetaVContext(who=[], where=[])
        if ctx_row:
            try:
                if ctx_row['who_list']: meta_context.who = json.loads(ctx_row['who_list'])
                if ctx_row['where_list']: meta_context.where = json.loads(ctx_row['where_list'])
            except json.JSONDecodeError: pass

        # C. Fetch People (Stephenson Granular)
        people_rows = cursor.execute("""
            SELECT DISTINCT p.id, p.name, p.description, vp.role 
            FROM verse_people vp
            JOIN people p ON vp.person_id = p.id
            WHERE vp.book=? AND vp.chapter=? AND vp.verse=?
        """, (book_code, chapter, verse)).fetchall()
        
        people = [Entity(id=r['id'], name=r['name'], description=r['description'], role=r['role']) for r in people_rows]

        # D. Fetch Places
        place_rows = cursor.execute("""
            SELECT DISTINCT p.id, p.name, p.description 
            FROM verse_places vp
            JOIN places p ON vp.place_id = p.id
            WHERE vp.book=? AND vp.chapter=? AND vp.verse=?
        """, (book_code, chapter, verse)).fetchall()
        
        places = [Entity(id=r['id'], name=r['name'], description=r['description']) for r in place_rows if r['name']]

        return VerseMetaResponse(
            reference=f"{book_code} {chapter}:{verse}",
            book_meta=book_meta,
            context=meta_context,
            people=people,
            places=places
        )

    except Exception as e:
        print(f"🔥 CRASH on {book} {chapter}:{verse}")
        print(traceback.format_exc())
        raise e

    finally:
        conn.close()