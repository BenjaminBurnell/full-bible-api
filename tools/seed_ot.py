# tools/seed_ot.py
import os, csv, sqlite3, argparse

BASE = os.path.dirname(os.path.dirname(__file__))
DB   = os.environ.get("INTERLINEAR_DB", os.path.join(BASE, "interlinear.sqlite3"))

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="Path to interlinear_ot.normalized.csv")
args = ap.parse_args()

con = sqlite3.connect(DB)
cur = con.cursor()
# ensure schema
cur.execute("""
CREATE TABLE IF NOT EXISTS tokens(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  book_code TEXT, chapter INTEGER, verse INTEGER,
  token_index INTEGER,
  surface TEXT, lemma TEXT, translit TEXT, gloss TEXT,
  morph TEXT, strong TEXT
)""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_loc ON tokens(book_code,chapter,verse)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_strong ON tokens(strong)")

# load CSV
count=0
with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
  r=csv.DictReader(f)
  for row in r:
    cur.execute("""
      INSERT INTO tokens(book_code,chapter,verse,token_index,surface,lemma,translit,gloss,morph,strong)
      VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
      (row.get("book_code") or row.get("book") or "").strip(),
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
    count+=1

con.commit(); con.close()
print(f"Inserted {count} OT tokens into {DB}")
