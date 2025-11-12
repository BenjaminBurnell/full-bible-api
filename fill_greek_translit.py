# fill_greek_translit.py
# Auto-fills 'translit' column for data/greek_lexicon.csv where empty, preserving any existing values.
# Keeps 'gloss' unchanged (you can fill glosses gradually).
import csv, os, unicodedata

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "greek_lexicon.csv")
OUT_PATH = CSV_PATH  # in-place update

# Basic Greek -> Latin mapping (ASCII, simple and readable)
MAP = {
    "Α":"A","α":"a","Β":"B","β":"b","Γ":"G","γ":"g","Δ":"D","δ":"d","Ε":"E","ε":"e",
    "Ζ":"Z","ζ":"z","Η":"E","η":"e","Θ":"Th","θ":"th","Ι":"I","ι":"i","Κ":"K","κ":"k",
    "Λ":"L","λ":"l","Μ":"M","μ":"m","Ν":"N","ν":"n","Ξ":"X","ξ":"x","Ο":"O","ο":"o",
    "Π":"P","π":"p","Ρ":"R","ρ":"r","Σ":"S","σ":"s","ς":"s","Τ":"T","τ":"t","Υ":"Y","υ":"y",
    "Φ":"Ph","φ":"ph","Χ":"Ch","χ":"ch","Ψ":"Ps","ψ":"ps","Ω":"O","ω":"o",
    # rough breathing marks (ῥ etc.) handled by base letter; you can refine later
}

def strip_diacritics(s: str) -> str:
    # Normalize and remove combining marks; keep base Greek letters for mapping
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def transliterate(lemma: str) -> str:
    # Remove diacritics, then map char-by-char
    base = strip_diacritics(lemma)
    out = []
    for ch in base:
        out.append(MAP.get(ch, ch))
    # lightweight fixes for common digraphs
    s = "".join(out)
    # Optional: smoother vowels (e.g., 'Ai'/'Ei') can be tuned later
    return s

assert os.path.isfile(CSV_PATH), f"Could not find {CSV_PATH}"

rows = []
with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames or ["lemma","translit","gloss"]
    for row in r:
        lemma = (row.get("lemma") or "").strip()
        translit = (row.get("translit") or "").strip()
        gloss = (row.get("gloss") or "").strip()
        if lemma and not translit:
            translit = transliterate(lemma)
        rows.append({"lemma": lemma, "translit": translit, "gloss": gloss})

with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["lemma","translit","gloss"])
