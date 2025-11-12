# Build a lemma-based Greek lexicon (lemma, translit, gloss) from data/strongs_lexicon.csv (G#### rows).
import os, csv, unicodedata

BASE = os.path.dirname(os.path.dirname(__file__))  # project root
DATA = os.path.join(BASE, "data")
SRC  = os.path.join(DATA, "strongs_lexicon.csv")    # must contain both H#### + G#### rows
OUT  = os.path.join(DATA, "greek_lexicon.csv")      # lemma-based output

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()

def simple_translit(lemma: str) -> str:
    # Basic Greek -> Latin; good enough for readable display (e.g., ἀρχή -> arche)
    MAP = {
        "Α":"A","α":"a","Β":"B","β":"b","Γ":"G","γ":"g","Δ":"D","δ":"d","Ε":"E","ε":"e",
        "Ζ":"Z","ζ":"z","Η":"E","η":"e","Θ":"Th","θ":"th","Ι":"I","ι":"i","Κ":"K","κ":"k",
        "Λ":"L","λ":"l","Μ":"M","μ":"m","Ν":"N","ν":"n","Ξ":"X","ξ":"x","Ο":"O","ο":"o",
        "Π":"P","π":"p","Ρ":"R","ρ":"r","Σ":"S","σ":"s","ς":"s","Τ":"T","τ":"t","Υ":"Y","υ":"y",
        "Φ":"Ph","φ":"ph","Χ":"Ch","χ":"ch","Ψ":"Ps","ψ":"ps","Ω":"O","ω":"o",
    }
    base = unicodedata.normalize("NFKD", lemma)
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    return "".join(MAP.get(ch, ch) for ch in base)

def pick_better(a, b):
    # Prefer non-empty gloss; if both have gloss pick the shorter; otherwise prefer non-empty translit.
    ga, gb = a["gloss"], b["gloss"]
    if bool(ga) != bool(gb): return a if ga else b
    if ga and gb and len(ga) != len(gb): return a if len(ga) < len(gb) else b
    ta, tb = a["translit"], b["translit"]
    if bool(ta) != bool(tb): return a if ta else b
    return a

assert os.path.isfile(SRC), f"Missing {SRC}. Make sure data/strongs_lexicon.csv exists."

greek_map = {}  # lemma -> {lemma, translit, gloss}

with open(SRC, "r", encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        strong = (row.get("strong") or "").strip().upper()
        if not strong.startswith("G"):  # Greek only
            continue
        lemma = nfc(row.get("lemma") or "")
        if not lemma:
            continue
        translit = (row.get("translit") or "").strip()
        gloss    = (row.get("gloss") or "").strip()
        cand = {"lemma": lemma, "translit": translit, "gloss": gloss}
        greek_map[lemma] = pick_better(greek_map[lemma], cand) if lemma in greek_map else cand

# Ensure every lemma has a transliteration
for v in greek_map.values():
    if not v["translit"]:
        v["translit"] = simple_translit(v["lemma"])

# Write lemma-based file
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["lemma","translit","gloss"])
    w.writeheader()
    for lemma in sorted(greek_map.keys()):
        w.writerow(greek_map[lemma])

print(f"Wrote {len(greek_map)} lemma rows -> {OUT}")
