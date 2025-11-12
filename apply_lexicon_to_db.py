# apply_lexicon_to_db.py
import csv, os, sqlite3, re, argparse
from typing import Dict

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.environ.get("INTERLINEAR_DB", os.path.join(BASE_DIR, "interlinear.sqlite3"))
DATA_DIR = os.path.join(BASE_DIR, "data")

STRONGS_CSV = os.path.join(DATA_DIR, "strongs_lexicon.csv")  # strong,lemma,translit,gloss
GREEK_CSV   = os.path.join(DATA_DIR, "greek_lexicon.csv")    # lemma,translit,gloss (optional)

def norm_strong_keys(raw: str):
    if not raw:
        return ()
    parts = re.split(r"[,\s/;]+", raw.strip())
    keys = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.match(r"^[HhGg]\d+$", p):
            prefix = p[0].upper()
            num = re.sub(r"\D", "", p[1:])
            if num:
                keys.append(prefix + num)  # H7225/G3056
                keys.append(num)           # 7225
        else:
            num = re.sub(r"\D", "", p)
            if num:
                keys.append("H" + num)
                keys.append("G" + num)
                keys.append(num)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return tuple(out)

def load_strongs_map() -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(STRONGS_CSV):
        return {}
    out: Dict[str, Dict[str,str]] = {}
    with open(STRONGS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            strong = (row.get("strong") or "").strip()
            if not strong:
                continue
            entry = {
                "lemma":   (row.get("lemma") or "").strip(),
                "translit":(row.get("translit") or "").strip(),
                "gloss":   (row.get("gloss") or "").strip(),
            }
            for k in norm_strong_keys(strong):
                out[k] = entry
    return out

def load_greek_map() -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(GREEK_CSV):
        return {}
    out: Dict[str, Dict[str,str]] = {}
    with open(GREEK_CSV, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lemma = (row.get("lemma") or "").strip()
            if not lemma:
                continue
            out[lemma] = {
                "lemma": lemma,
                "translit": (row.get("translit") or "").strip(),
                "gloss": (row.get("gloss") or "").strip(),
            }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true",
                    help="Update ALL tokens that have a Strong’s code, regardless of current DB values.")
    args = ap.parse_args()

    strongs_map = load_strongs_map()
    greek_map = load_greek_map()
    if not (strongs_map or greek_map):
        print("No lexicon CSVs found in ./data. Expected strongs_lexicon.csv and/or greek_lexicon.csv.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_strong ON tokens(strong)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma)")

    if args.overwrite:
        # ✅ Select ALL rows with a strong code
        cur.execute("""
            SELECT id, strong,
                   COALESCE(lemma,'')   AS lemma,
                   COALESCE(translit,'') AS translit,
                   COALESCE(gloss,'')    AS gloss
            FROM tokens
            WHERE TRIM(COALESCE(strong,'')) <> ''
        """)
    else:
        # Fill only empties
        cur.execute("""
            SELECT id, strong,
                   COALESCE(lemma,'')   AS lemma,
                   COALESCE(translit,'') AS translit,
                   COALESCE(gloss,'')    AS gloss
            FROM tokens
            WHERE TRIM(COALESCE(lemma,''))   = ''
               OR TRIM(COALESCE(translit,''))= ''
               OR TRIM(COALESCE(gloss,''))   = ''
        """)

    rows = cur.fetchall()

    updated = 0
    by_strong = 0
    by_lemma = 0

    for r in rows:
        pk       = r["id"]
        t_strong = (r["strong"] or "").strip()
        t_lemma  = (r["lemma"] or "").strip()
        t_transl = (r["translit"] or "").strip()
        t_gloss  = (r["gloss"] or "").strip()

        hit = None
        # Prefer Strong’s
        for k in norm_strong_keys(t_strong):
            hit = strongs_map.get(k)
            if hit:
                by_strong += 1
                break
        # Fallback: lemma (Greek map)
        if not hit and t_lemma:
            hit = greek_map.get(t_lemma)
            if hit:
                by_lemma += 1
        if not hit:
            continue

        # Decide new values
        if args.overwrite:
            new_lemma  = hit.get("lemma", "") or t_lemma
            new_transl = hit.get("translit", "") or t_transl
            new_gloss  = hit.get("gloss", "") or t_gloss
        else:
            new_lemma  = t_lemma  or hit.get("lemma", "")
            new_transl = t_transl or hit.get("translit", "")
            new_gloss  = t_gloss  or hit.get("gloss", "")

        # Write if anything changes
        if (new_lemma, new_transl, new_gloss) != (t_lemma, t_transl, t_gloss):
            cur.execute("""
                UPDATE tokens
                   SET lemma = ?, translit = ?, gloss = ?
                 WHERE id = ?
            """, (new_lemma, new_transl, new_gloss, pk))
            updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} tokens (by strong: {by_strong}, by lemma: {by_lemma}).")

if __name__ == "__main__":
    main()
