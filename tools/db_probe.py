import os, sqlite3, json

DB = os.environ.get("INTERLINEAR_DB", "interlinear.sqlite3")
print("DB:", DB)
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
cur = con.cursor()

has_tokens = bool(cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tokens'").fetchone())
print("has tokens table:", has_tokens)
if not has_tokens:
    con.close(); raise SystemExit()

books = cur.execute("SELECT book_code, COUNT(*) c FROM tokens GROUP BY book_code ORDER BY book_code").fetchall()
print("books:", [r["book_code"] for r in books][:30])
print("total rows:", sum(r["c"] for r in books))

g_rows = cur.execute("SELECT COUNT(*) FROM tokens WHERE strong LIKE 'G%'").fetchone()[0]
print("rows with G-strong (NT):", g_rows)

# Show a few rows that look like NT (G-strong first; else any lemma-only)
sample = cur.execute("""
  SELECT book_code, chapter, verse, token_index, surface, lemma, strong
    FROM tokens
   WHERE strong LIKE 'G%' OR (TRIM(COALESCE(strong,''))='' AND TRIM(COALESCE(lemma,''))<>'')
   LIMIT 15
""").fetchall()
print("NT-ish sample:", [dict(r) for r in sample])

# What code does John use in your DB?
john = cur.execute("SELECT DISTINCT book_code FROM tokens WHERE UPPER(book_code) IN ('JHN','JOHN','JN','JHN1')").fetchall()
print("John codes present:", [r["book_code"] for r in john])

con.close()
