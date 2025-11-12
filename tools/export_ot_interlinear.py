# tools/export_ot_interlinear.py
# Export OT interlinear JSON without relying on strong LIKE 'H%'.
# Detect OT per-verse by normalizing token strongs and checking for an H#### candidate.

import os, re, csv, json, argparse, sqlite3

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "out", "ot")
DB   = os.environ.get("INTERLINEAR_DB", os.path.join(BASE, "interlinear.sqlite3"))
STRONGS_CSV = os.path.join(DATA, "strongs_lexicon.csv")  # strong,lemma,translit,gloss

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def clean(s): 
    import re as _re; return _re.sub(r"\s+", " ", (s or "").strip())

def norm_strong_keys(raw: str):
    """Return candidate strong keys, preferring Hebrew for digits-only (OT bias)."""
    if not raw: return ()
    parts = re.split(r"[,\s/;]+", raw.strip())
    keys = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if re.match(r"^[HhGg]\d+$", p):
            prefix = p[0].upper()
            num = re.sub(r"\D", "", p[1:])
            if num:
                keys += [prefix+num, num]        # e.g., H7225, 7225
        else:
            num = re.sub(r"\D", "", p)
            if num:
                keys += ["H"+num, num, "G"+num]  # prefer Hebrew first
    # dedupe preserving order
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return tuple(out)

def load_strongs_map(path: str):
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}. Put strongs_lexicon.csv in ./data/")
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s = (row.get("strong") or "").strip()
            if not s: continue
            entry = {"lemma": (row.get("lemma") or "").strip(),
                     "translit": (row.get("translit") or "").strip(),
                     "gloss": (row.get("gloss") or "").strip()}
            for k in norm_strong_keys(s):
                out[k] = entry
    return out

def export_verse(book_code, chapter, verse, tokens, out_base):
    payload = {
        "reference": f"{book_code} {chapter}:{verse}",
        "book": book_code, "book_code": book_code,
        "chapter": chapter, "verse": verse,
        "tokens": tokens,
    }
    dirp = os.path.join(out_base, book_code, str(chapter))
    ensure_dir(dirp)
    fp = os.path.join(dirp, f"{verse}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fp

def main():
    ap = argparse.ArgumentParser(description="Export OT interlinear JSON (normalize strongs; OT bias).")
    ap.add_argument("--books", nargs="*", default=[], help="Optional: restrict to these book_code(s).")
    ap.add_argument("--write-db", action="store_true", help="Write resolved lemma/translit/gloss back to SQLite.")
    args = ap.parse_args()

    strongs = load_strongs_map(STRONGS_CSV)

    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all books present (optionally restricted)
    if args.books:
        placeholders = ",".join("?"*len(args.books))
        cur.execute(f"SELECT DISTINCT book_code FROM tokens WHERE UPPER(book_code) IN ({placeholders}) ORDER BY book_code",
                    tuple(b.upper() for b in args.books))
    else:
        cur.execute("SELECT DISTINCT book_code FROM tokens ORDER BY book_code")
    books = [row["book_code"] for row in cur.fetchall()]

    total_verses = 0
    total_tokens = 0
    updated_rows = 0

    for book in books:
        # enumerate verses
        cur.execute("""SELECT chapter, verse FROM tokens WHERE book_code=? GROUP BY chapter, verse ORDER BY chapter, verse""", (book,))
        refs = [(int(r["chapter"]), int(r["verse"])) for r in cur.fetchall()]

        for ch, vs in refs:
            rows = cur.execute("""
                SELECT id, token_index, surface, lemma, translit, gloss, morph, strong
                FROM tokens
                WHERE book_code=? AND chapter=? AND verse=?
                ORDER BY token_index
            """, (book, ch, vs)).fetchall()

            # Detect if this verse is OT: any token whose normalized candidates include an H#### key
            is_ot = False
            for r in rows:
                for k in norm_strong_keys(r["strong"] or ""):
                    if k.startswith("H"):
                        is_ot = True
                        break
                if is_ot: break
            if not is_ot:
                continue  # skip NT-like verses

            out_tokens = []
            for r in rows:
                pk      = r["id"]
                surface = r["surface"] or ""
                lemma   = r["lemma"] or ""
                transl  = r["translit"] or ""
                gloss   = r["gloss"] or ""
                morph   = r["morph"] or ""
                strong  = r["strong"] or ""
                index   = int(r["token_index"])

                resolved = {}
                for k in norm_strong_keys(strong):
                    hit = strongs.get(k)
                    if hit:
                        resolved = hit; break

                r_lemma  = lemma  or resolved.get("lemma", "")
                r_transl = transl or resolved.get("translit", "")
                r_gloss  = gloss  or resolved.get("gloss", "")

                if args.write_db and (r_lemma != lemma or r_transl != transl or r_gloss != gloss):
                    cur.execute("UPDATE tokens SET lemma=?, translit=?, gloss=? WHERE id=?",
                                (r_lemma, r_transl, r_gloss, pk))
                    updated_rows += cur.rowcount

                out_tokens.append({
                    "surface": surface, "lemma": lemma, "translit": transl, "gloss": gloss,
                    "morph": morph, "strong": strong, "index": index,
                    "resolved_lemma": r_lemma, "resolved_translit": r_transl, "resolved_gloss": r_gloss,
                    "translation": r_gloss
                })
                total_tokens += 1

            export_verse(book, ch, vs, out_tokens, OUT)
            total_verses += 1

    if args.write_db:
        conn.commit()
    conn.close()

    print(f"Exported {total_verses} OT verses, {total_tokens} tokens to {OUT}")
    if args.write_db:
        print(f"DB rows updated with resolved fields: {updated_rows}")

if __name__ == "__main__":
    main()