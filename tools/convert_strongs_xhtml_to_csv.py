import os, re, csv
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.dirname(__file__))
RAW  = os.path.join(BASE, "raw")
DATA = os.path.join(BASE, "data")
SRC  = os.path.join(RAW, "strongs-dictionary.xhtml")
OUT  = os.path.join(DATA, "strongs_lexicon.csv")
os.makedirs(DATA, exist_ok=True)

def clean(s):
    import re as _re
    return _re.sub(r"\s+", " ", (s or "").strip())

def first_after_punct(txt):
    import re as _re
    txt = clean(txt)
    m = _re.search(r"[—\-:;]\s*(.+)", txt)
    if m: txt = m.group(1)
    return txt.split(".")[0].split(";")[0][:160]

with open(SRC, "rb") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

rows = []

def parse(sec_id, prefix):
    sec = soup.find(id=sec_id)
    if not sec: return 0
    count = 0
    for li in sec.find_all("li"):
        val = li.get("value")
        if not (val and val.isdigit()):
            continue
        strong = f"{prefix}{int(val)}"
        itag = li.find("i")
        lemma = clean(itag.get_text()) if itag else ""
        translit = ""
        if itag and itag.has_attr("title"):
            translit = itag["title"].strip().strip("{}")
        gloss_tag = li.find("span", class_="kjv_def")
        gloss = clean(gloss_tag.get_text()) if gloss_tag else ""
        if not gloss:
            gloss = first_after_punct(li.get_text(" "))
        rows.append({"strong": strong, "lemma": lemma, "translit": translit, "gloss": gloss})
        count += 1
    return count

ot = parse("ot", "H")
nt = parse("nt", "G")

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["strong","lemma","translit","gloss"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

nonempty = sum(1 for r in rows if r["gloss"])
print(f"Wrote {len(rows)} rows (OT: {ot}, NT: {nt}); with gloss: {nonempty}")
print("CSV:", OUT)