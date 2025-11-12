# convert_oshb_osis_to_normalized.py
# Converts OSHB OSIS (wlc/*.xml) into your normalized CSV schema.
# Input: folder containing OSIS files (e.g., data/oshb/morphhb-master/wlc)
# Output: data/interlinear_ot.normalized.csv

import os, csv, sys
import xml.etree.ElementTree as ET
from collections import defaultdict

OUT_PATH = os.path.join("data", "interlinear_ot.normalized.csv")

# Map OSIS book IDs -> your 3-letter codes
OSIS_TO_CODE = {
    "Gen":"GEN","Exod":"EXO","Lev":"LEV","Num":"NUM","Deut":"DEU",
    "Josh":"JOS","Judg":"JDG","Ruth":"RUT","1Sam":"1SA","2Sam":"2SA",
    "1Kgs":"1KI","2Kgs":"2KI","1Chr":"1CH","2Chr":"2CH","Ezra":"EZR",
    "Neh":"NEH","Esth":"EST","Job":"JOB","Ps":"PSA","Prov":"PRO",
    "Eccl":"ECC","Song":"SNG","Isa":"ISA","Jer":"JER","Lam":"LAM",
    "Ezek":"EZK","Dan":"DAN","Hos":"HOS","Joel":"JOL","Amos":"AMO",
    "Obad":"OBA","Jonah":"JON","Mic":"MIC","Nah":"NAM","Hab":"HAB",
    "Zeph":"ZEP","Hag":"HAG","Zech":"ZEC","Mal":"MAL"
}

FIELDS = ["book_code","chapter","verse","token_index","surface","lemma","translit","gloss","morph","strong"]

def debug(msg): 
    print(msg, file=sys.stderr)

def parse_osis_file(path, sink):
    """
    Parse a single OSIS XML, accumulate tokens into sink[(code, chap, verse)] = list[...] 
    We look for:
      - verse elements with @osisID like 'Gen.1.1'
      - word elements <w> with attributes lemma="H7225" morph="Ncfsa" and text = surface
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # OSIS uses 'osis' namespace sometimes; we’ll support both
    ns = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    # Find every verse element regardless of prefix
    verses = root.findall(".//{*}verse")
    for v in verses:
        osis_id = v.attrib.get("osisID") or v.attrib.get("{http://www.bibletechnologies.net/2003/OSIS/namespace}osisID")
        if not osis_id:
            continue
        # Examples: Gen.1.1  or Ps.119.1
        parts = osis_id.split(".")
        if len(parts) != 3:
            continue
        osis_book, chap_str, verse_str = parts
        book_code = OSIS_TO_CODE.get(osis_book)
        if not book_code:
            # skip non-OT or unexpected IDs
            continue
        try:
            chap = int(chap_str)
            verse = int(verse_str)
        except:
            continue

        # collect words inside this verse
        key = (book_code, chap, verse)
        # Iterate all descendant <w> nodes under verse
        # Some OSIS uses <w> with namespaces, handle both
        words = v.findall(".//{*}w")
        # token_index per verse
        for w in words:
            surface = (w.text or "").strip()
            if not surface:
                continue
            lemma_attr = (w.attrib.get("lemma") or "").strip()  # often "H1254"
            morph_attr = (w.attrib.get("morph") or "").strip()

            # OSHB puts Strong's in lemma attr (e.g., "H7225"); keep that in strong column.
            strong = lemma_attr if lemma_attr else ""

            sink[key].append({
                "surface": surface,
                "lemma": "",          # OSHB lemma attr is a Strong's key; we store that in 'strong'
                "translit": "",       # not provided by OSIS; can be joined later
                "gloss": "",          # not provided by OSIS; can be joined later
                "morph": morph_attr,
                "strong": strong
            })

def main():
    # Find the extracted folder containing wlc/*.xml
    base = os.path.join("data","oshb")
    # Try common subpaths after unzipping the GitHub archive
    candidates = []
    for root, dirs, files in os.walk(base):
        if os.path.basename(root).lower() == "wlc":
            if any(f.lower().endswith(".xml") for f in files):
                candidates.append(root)
    if not candidates:
        sys.exit("❌ Could not find a 'wlc' folder with OSIS XML under data/oshb. Check your unzip location.")

    wlc_dir = candidates[0]
    print(f"🔎 Using OSIS from: {wlc_dir}")

    sink = defaultdict(list)
    files = [f for f in os.listdir(wlc_dir) if f.lower().endswith(".xml")]
    files.sort()
    total_files = len(files)
    for i, fname in enumerate(files, 1):
        path = os.path.join(wlc_dir, fname)
        parse_osis_file(path, sink)
        if i % 5 == 0 or i == total_files:
            print(f"  … parsed {i}/{total_files} files", file=sys.stderr)

    # Write normalized CSV
    out_path = OUT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_rows = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (code, chap, verse), toks in sorted(sink.items()):
            for idx, t in enumerate(toks, start=1):
                w.writerow({
                    "book_code": code,
                    "chapter": chap,
                    "verse": verse,
                    "token_index": idx,
                    "surface": t["surface"],
                    "lemma": t["lemma"],
                    "translit": t["translit"],
                    "gloss": t["gloss"],
                    "morph": t["morph"],
                    "strong": t["strong"]
                })
                total_rows += 1

    print(f"✅ Wrote normalized OT → {out_path} (rows: {total_rows:,})")

if __name__ == "__main__":
    main()
