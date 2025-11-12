
# Interlinear Bible API (Minimal)

A tiny FastAPI + SQLite service that serves word-by-word (interlinear) tokens for Bible verses.

> **What you get right now:** a working `/interlinear/GEN/1/1` endpoint (Genesis 1:1) using sample Hebrew data, so you can wire your UI while you expand the dataset.

## Endpoints

- `GET /health` — sanity check
- `GET /interlinear/{book}/{chapter}/{verse}` — returns tokens for the verse. `book` can be code (`GEN`) or full name (`Genesis`).

Example:

```
GET /interlinear/GEN/1/1
```

Response:

```json
{
  "reference": "Genesis 1:1",
  "book": "Genesis",
  "book_code": "GEN",
  "chapter": 1,
  "verse": 1,
  "tokens": [
    { "surface": "בְּרֵאשִׁית", "lemma": "רֵאשִׁית", "translit": "berēʾšît", "gloss": "in-beginning", "morph": "Ncfsa", "strong": "H7225", "index": 1 },
    { "surface": "בָּרָא", "lemma": "בָּרָא", "translit": "bārāʾ", "gloss": "created", "morph": "Vqp3ms", "strong": "H1254", "index": 2 },
    { "surface": "אֱלֹהִים", "lemma": "אֱלֹהִים", "translit": "ʾĕlōhîm", "gloss": "God", "morph": "Ncmpa", "strong": "H430", "index": 3 },
    { "surface": "אֵת", "lemma": "אֵת", "translit": "ʾēt", "gloss": "[obj-marker]", "morph": "PAA", "strong": "H853", "index": 4 },
    { "surface": "הַשָּׁמַיִם", "lemma": "שָׁמַיִם", "translit": "haššāmayim", "gloss": "the-heavens", "morph": "Ncmpa+Art", "strong": "H8064", "index": 5 },
    { "surface": "וְ", "lemma": "וְ", "translit": "wə", "gloss": "and", "morph": "Conj", "strong": "H2053", "index": 6 },
    { "surface": "אֵת", "lemma": "אֵת", "translit": "ʾēt", "gloss": "[obj-marker]", "morph": "PAA", "strong": "H853", "index": 7 },
    { "surface": "הָאָרֶץ", "lemma": "אֶרֶץ", "translit": "hāʾāreṣ", "gloss": "the-earth", "morph": "Ncfsa+Art", "strong": "H776", "index": 8 }
  ]
}
```

> **Notes**
> - The morphology/Strong’s here are illustrative; tailor to your dataset conventions.
> - You can expand the dataset by appending more rows to `data/interlinear_tokens.csv` and rerunning `seed.py`.

## Quickstart (local)

```bash
# 1) Create virtual env (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Initialize the DB schema
python db.py

# 4) Seed sample data (Genesis 1:1)
python seed.py

# 5) Run
uvicorn app:app --reload
# or the port Render likes:
uvicorn app:app --host 0.0.0.0 --port 10000
```

Open: `http://127.0.0.1:8000/interlinear/GEN/1/1`

## Deploying to Render

- Create a new **Web Service**.
- Runtime: **Python**.
- Start command: `uvicorn app:app --host 0.0.0.0 --port 10000` (or use `Procfile` with `web:` line).
- Add environment var (optional): `INTERLINEAR_DB=interlinear.sqlite3`

## Data format

Append rows to `data/interlinear_tokens.csv` with columns:

```
book_code,chapter,verse,token_index,surface,lemma,translit,gloss,morph,strong
```

You can preload OSHB (Hebrew) / MorphGNT (Greek) derived exports into this CSV (script that converts their formats into these columns).

> Tip: keep `token_index` sequential per verse so tokens render in order.

## Book codes

Edit `data/book_codes.json` if you want to add/rename codes (full names also work in the endpoint).
