import sqlite3
import csv
import os
import json
from collections import defaultdict

# Config
DB_PATH = "metadata.db"
RAW_DIR = os.path.join("data", "metadata_raw")

# Standard USFM 3-letter codes
BOOK_MAP = {
    "Genesis": "GEN", "Exodus": "EXO", "Leviticus": "LEV", "Numbers": "NUM", "Deuteronomy": "DEU",
    "Joshua": "JOS", "Judges": "JDG", "Ruth": "RUT", "1 Samuel": "1SA", "2 Samuel": "2SA",
    "1 Kings": "1KI", "2 Kings": "2KI", "1 Chronicles": "1CH", "2 Chronicles": "2CH",
    "Ezra": "EZR", "Nehemiah": "NEH", "Esther": "EST", "Job": "JOB", "Psalms": "PSA",
    "Proverbs": "PRO", "Ecclesiastes": "ECC", "Song of Solomon": "SNG", "Isaiah": "ISA",
    "Jeremiah": "JER", "Lamentations": "LAM", "Ezekiel": "EZK", "Daniel": "DAN",
    "Hosea": "HOS", "Joel": "JOL", "Amos": "AMO", "Obadiah": "OBA", "Jonah": "JON",
    "Micah": "MIC", "Nahum": "NAM", "Habakkuk": "HAB", "Zephaniah": "ZEP", "Haggai": "HAG",
    "Zechariah": "ZEC", "Malachi": "MAL", "Matthew": "MAT", "Mark": "MRK", "Luke": "LUK",
    "John": "JHN", "Acts": "ACT", "Romans": "ROM", "1 Corinthians": "1CO", "2 Corinthians": "2CO",
    "Galatians": "GAL", "Ephesians": "EPH", "Philippians": "PHP", "Colossians": "COL",
    "1 Thessalonians": "1TH", "2 Thessalonians": "2TH", "1 Timothy": "1TI", "2 Timothy": "2TI",
    "Titus": "TIT", "Philemon": "PHM", "Hebrews": "HEB", "James": "JAS", "1 Peter": "1PE",
    "2 Peter": "2PE", "1 John": "1JN", "2 John": "2JN", "3 John": "3JN", "Jude": "JUD",
    "Revelation": "REV"
}

def normalize_book(name):
    if not name: return None
    clean = name.strip()
    if clean in BOOK_MAP: return BOOK_MAP[clean]
    if len(clean) == 3: return clean.upper()
    return None

def parse_reference_id(ref_id):
    """Parses 'GEN 1:1' into ('GEN', 1, 1)."""
    if not ref_id: return None
    try:
        last_colon = ref_id.rfind(':')
        if last_colon == -1: return None
        
        verse_part = ref_id[last_colon+1:].strip()
        verse = int(''.join(filter(str.isdigit, verse_part)))
        
        remainder = ref_id[:last_colon].strip()
        last_space = remainder.rfind(' ')
        if last_space == -1: return None
        
        chapter_str = remainder[last_space+1:].strip()
        chapter = int(chapter_str)
        
        book_str = remainder[:last_space].strip() 
        book_code = normalize_book(book_str)
        
        if book_code:
            return (book_code, chapter, verse)
        return None
    except:
        return None

def init_db(conn):
    cur = conn.cursor()
    
    # 1. Books (Added hebrew_meaning column)
    cur.execute("""CREATE TABLE IF NOT EXISTS books (
        code TEXT PRIMARY KEY, 
        title TEXT, 
        writer_id TEXT,
        date_written TEXT, 
        place_written TEXT, 
        audience TEXT,
        hebrew_meaning TEXT  -- <--- NEW COLUMN
    )""")

    # 2. People
    cur.execute("""CREATE TABLE IF NOT EXISTS people (
        id TEXT PRIMARY KEY, 
        name TEXT, 
        description TEXT,
        sex TEXT,
        tribe TEXT,
        unique_attribute TEXT
    )""")
    
    # 3. Places
    cur.execute("CREATE TABLE IF NOT EXISTS places (id INTEGER PRIMARY KEY, name TEXT, description TEXT)")

    # 4. Links
    cur.execute("CREATE TABLE IF NOT EXISTS verse_people (book TEXT, chapter INT, verse INT, person_id TEXT, role TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS verse_places (book TEXT, chapter INT, verse INT, place_id INTEGER)")
    
    # 5. Context
    cur.execute("""CREATE TABLE IF NOT EXISTS metav_context (
        book TEXT, chapter INT, verse INT, 
        who_list TEXT, where_list TEXT, 
        PRIMARY KEY(book, chapter, verse))""")
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_vp ON verse_people(book, chapter, verse)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_vpl ON verse_places(book, chapter, verse)")
    conn.commit()

def import_stephenson(cur):
    print("--- Importing Stephenson Data ---")
    
    # 1. BOOKS
    path = os.path.join(RAW_DIR, "books.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                code = row.get('usx_code')
                if not code: code = normalize_book(row.get('book_name'))
                
                writer_id = row.get('writer_id', '')
                
                start = row.get('written_start_date', '')
                end = row.get('written_end_date', '')
                date_str = f"{start}" if start == end else f"{start}-{end}"

                if code:
                    # Added row.get('hebrew_meaning', '') to the insert
                    cur.execute("INSERT OR IGNORE INTO books VALUES (?,?,?,?,?,?,?)", 
                        (code, 
                         row.get('book_name'), 
                         writer_id, 
                         date_str, 
                         row.get('written_location_id'), 
                         '', 
                         row.get('hebrew_meaning', '') # <--- NEW
                        ))
                    count += 1
            print(f"✅ Loaded {count} Books.")

    # 2. PEOPLE DEFINITIONS
    path = os.path.join(RAW_DIR, "persons.with_descriptions.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                pid = row.get('person_id')
                name = row.get('person_name')
                if pid and name:
                    cur.execute("INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)", (
                        pid, 
                        name, 
                        row.get('person_notes', ''),
                        row.get('sex', ''),
                        row.get('tribe', ''),
                        row.get('unique_attribute', '')
                    ))
                    count += 1
            print(f"✅ Loaded {count} Person Definitions.")

    # 3. PERSON LINKS
    path = os.path.join(RAW_DIR, "person_verse.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                ref_data = parse_reference_id(row.get('reference_id'))
                pid = row.get('person_id')
                if ref_data and pid:
                    cur.execute("INSERT INTO verse_people VALUES (?,?,?,?,?)", 
                        (ref_data[0], ref_data[1], ref_data[2], pid, row.get('person_verse_notes', '')))
                    count += 1
            print(f"✅ Linked {count} People to Verses.")

def import_metav(cur):
    print("--- Importing MetaV Data (Places & Context) ---")
    
    mp, mpl = {}, {}
    
    path = os.path.join(RAW_DIR, "MetaV_Places.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pid = int(r['PlaceID'])
                name = r.get('Place') or r.get('Name')
                mpl[pid] = name
                cur.execute("INSERT OR IGNORE INTO places VALUES (?,?,?)", (pid, name, ''))

    path = os.path.join(RAW_DIR, "MetaV_People.csv")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f): mp[r['PersonID']] = r['Name']

    path = os.path.join(RAW_DIR, "MainIndex.csv")
    if os.path.exists(path):
        verse_who = defaultdict(set)
        verse_where = defaultdict(set)
        verse_places_links = defaultdict(set)
        book_keys = list(BOOK_MAP.values()) 

        with open(path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    bid = int(row['BookID'])
                    if 1 <= bid <= 66:
                        code = book_keys[bid-1]
                        key = (code, int(row['Chapter']), int(row['VerseNum']))
                        
                        pid = row.get('PersonID')
                        if pid and pid != '0' and pid in mp: 
                            verse_who[key].add(mp[pid])
                        
                        lid = row.get('PlaceID')
                        if lid and lid != '0':
                            lid_int = int(lid)
                            if lid_int in mpl: 
                                verse_where[key].add(mpl[lid_int])
                                verse_places_links[key].add(lid_int)
                except: continue
        
        for key, who in verse_who.items():
            cur.execute("INSERT OR REPLACE INTO metav_context VALUES (?,?,?,?,?)", 
                (key[0], key[1], key[2], json.dumps(list(who)), json.dumps(list(verse_where.get(key, [])))))
        
        for key, place_ids in verse_places_links.items():
            for pid in place_ids:
                cur.execute("INSERT INTO verse_places VALUES (?,?,?,?)", (key[0], key[1], key[2], pid))

        print(f"✅ Loaded Context for {len(verse_who)} verses.")

if __name__ == "__main__":
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    import_stephenson(conn.cursor())
    import_metav(conn.cursor())
    conn.commit()
    conn.close()
    print("--------------------------------")
    print("🎉 Metadata DB rebuild complete.")