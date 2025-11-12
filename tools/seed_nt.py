import os, csv, sqlite3, argparse

NT = {"MAT","MRK","LUK","JHN","ACT","ROM","1CO","2CO","GAL","EPH","PHP","COL",
      "1TH","2TH","1TI","2TI","TIT","PHM","HEB","JAS","1PE","2PE",
      "1JN","2JN","3JN","JUD","REV"}

DB = os.environ.get("INTERLINEAR_DB", "interlinear.sqlite3")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="Path to interlinear_tokens.normalized.csv (can be combined OT+NT)")
args = ap.parse_args()

con = sqlite3.connect(DB)
cur = con.cursor()

# ensure schema
cur.execute("""CREATE TABLE IF NOT EXISTS tokens(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_code TEXT, chapter INTEGER, verse INTEGER,
  token_index INTEGER,
  surface TEXT, lemma TEXT, translit TEXT, gloss TEXT,
  morph TEXT, strong TEXT
)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_loc ON tokens(book_code,chapter,verse)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_strong ON tokens(strong)")

# delete any existing NT rows (idempotent)
cur.execute(f"DELETE FROM tokens WHERE UPPER(book_code) IN ({','.join('?'*len(NT))})", tuple(NT))
con.commit()

inserted = 0
with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        bc = (row.get("book_code") or row.get("book") or "").upper().strip()
        if bc not in NT:
            continue
        cur.execute("""
          INSERT INTO tokens(book_code,chapter,verse,token_index,surface,lemma,translit,gloss,morph,strong)
          VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
          bc,
          int(row.get("chapter") or 0),
          int(row.get("verse") or 0),
          int(row.get("token_index") or 0),
          row.get("surface",""),
          row.get("lemma",""),
          row.get("translit",""),
          row.get("gloss",""),
          row.get("morph",""),
          row.get("strong",""),
        ))
        inserted += 1

con.commit(); con.close()
print(f"Inserted {inserted} NT tokens into {DB}")