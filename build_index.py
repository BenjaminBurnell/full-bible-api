#!/usr/bin/env python3
import os, json, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss


@dataclass(frozen=True)
class Verse:
    version: str
    book: str
    chapter: int
    verse: int
    text: str


def load_tsv(path: str, want_version: str = "ALL") -> List[Verse]:
    verses: List[Verse] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                raise ValueError(f"TSV parse error line {line_no}: expected 5 cols, got {len(parts)}")
            version = parts[0].strip()
            book = parts[1].strip()
            chap = int(parts[2])
            ver = int(parts[3])
            text = "\t".join(parts[4:]).strip()
            if want_version.upper() != "ALL" and version.upper() != want_version.upper():
                continue
            if not text:
                continue
            verses.append(Verse(version, book, chap, ver, text))
    return verses


def make_ref(v: Verse) -> str:
    return f"{v.version} {v.book} {v.chapter}:{v.verse}"


def build_passage_chunks(verses: List[Verse], chunk_size: int, stride: int) -> Tuple[List[str], List[Dict]]:
    """
    Overlapping chunks within each (version, book, chapter).
    Each chunk concatenates chunk_size consecutive verses, sliding by stride.
    """
    by_chapter: Dict[Tuple[str, str, int], List[Verse]] = {}
    for v in verses:
        key = (v.version, v.book, v.chapter)
        by_chapter.setdefault(key, []).append(v)

    for key in by_chapter:
        by_chapter[key].sort(key=lambda x: x.verse)

    texts: List[str] = []
    meta: List[Dict] = []

    for (version, book, chap), vs in by_chapter.items():
        n = len(vs)
        i = 0
        while i < n:
            window = vs[i:i + chunk_size]
            if not window:
                break
            start = window[0].verse
            end = window[-1].verse
            combined = " ".join([w.text for w in window])

            texts.append(combined)
            meta.append({
                "type": "chunk",
                "version": version,
                "book": book,
                "chapter": chap,
                "start_verse": start,
                "end_verse": end,
                "refs": [make_ref(w) for w in window]
            })

            i += stride
            if i >= n:
                break

    return texts, meta


def build_verse_items(verses: List[Verse]) -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    meta: List[Dict] = []
    for v in verses:
        texts.append(v.text)
        meta.append({
            "type": "verse",
            "version": v.version,
            "book": v.book,
            "chapter": v.chapter,
            "verse": v.verse,
            "ref": make_ref(v)
        })
    return texts, meta


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine similarity via inner product
    )
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TSV file: version<TAB>book<TAB>chapter<TAB>verse<TAB>text")
    ap.add_argument("--out", required=True, help="Output folder for the built index")
    ap.add_argument("--version", default="ALL", help="Filter by one version (KJV/NASB/...) or ALL")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    ap.add_argument("--chunk-size", type=int, default=6, help="Verses per chunk")
    ap.add_argument("--stride", type=int, default=3, help="Slide step between chunks")
    ap.add_argument("--include-verses", action="store_true", help="Also index individual verses")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    verses = load_tsv(args.data, want_version=args.version)
    if not verses:
        raise SystemExit("No verses loaded. Check --data / --version.")

    print(f"Loaded {len(verses):,} verses from {args.data}")

    model = SentenceTransformer(args.model)

    items_texts: List[str] = []
    items_meta: List[Dict] = []

    chunk_texts, chunk_meta = build_passage_chunks(verses, chunk_size=args.chunk_size, stride=args.stride)
    print(f"Built {len(chunk_texts):,} passage chunks (size={args.chunk_size}, stride={args.stride})")
    items_texts.extend(chunk_texts)
    items_meta.extend(chunk_meta)

    if args.include_verses:
        verse_texts, verse_meta = build_verse_items(verses)
        print(f"Adding {len(verse_texts):,} verse items")
        items_texts.extend(verse_texts)
        items_meta.extend(verse_meta)

    emb = embed_texts(model, items_texts)
    dim = emb.shape[1]
    print(f"Embeddings shape: {emb.shape} (dim={dim})")

    index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors = cosine similarity
    index.add(emb)

    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in items_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "count": int(emb.shape[0]),
            "dim": int(dim),
            "chunk_size": args.chunk_size,
            "stride": args.stride,
            "include_verses": bool(args.include_verses),
            "version_filter": args.version,
            "source_tsv": os.path.basename(args.data)
        }, f, indent=2)

    print(f"✅ Done. Index saved to: {args.out}")


if __name__ == "__main__":
    main()