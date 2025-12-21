import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Licenses to allow (edit this list to be stricter/looser).
# You MUST keep attribution for CC-BY / CC-BY-SA.
DEFAULT_ALLOWED_LICENSES = {
    "Public domain",
    "CC0",
    "CC BY 4.0",
    "CC BY-SA 4.0",
    "CC BY 3.0",
    "CC BY-SA 3.0",
    "CC BY 2.0",
    "CC BY-SA 2.0",
    # Some Commons files use variants like:
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-3.0",
    "CC-BY-SA-3.0",
}

def _clean_place_name(name: str) -> str:
    name = (name or "").strip()
    # Remove parenthetical disambiguators that can hurt search sometimes.
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
    # Collapse spaces
    name = re.sub(r"\s+", " ", name)
    return name

def _mw_get(params: Dict[str, Any], session: requests.Session, timeout: int = 30) -> Dict[str, Any]:
    r = session.get(COMMONS_API, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def commons_search_files(query: str, session: requests.Session, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Use MediaWiki search to find File: pages on Commons.
    namespace=6 is File namespace. :contentReference[oaicite:1]{index=1}
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,  # File:
        "srlimit": limit,
    }
    data = _mw_get(params, session)
    return data.get("query", {}).get("search", [])

def commons_imageinfo(titles: List[str], session: requests.Session, iiurlwidth: int = 1200) -> Dict[str, Any]:
    """
    Fetch URL + extmetadata (license/credit/artist/etc.). :contentReference[oaicite:2]{index=2}
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": "|".join(titles),
        "iiprop": "url|extmetadata",
        "iiurlwidth": iiurlwidth,
    }
    return _mw_get(params, session)

def pick_best_candidate(
    pages: Dict[str, Any],
    allowed_licenses: set,
) -> Optional[Dict[str, Any]]:
    """
    Choose the first candidate with an allowed license and a usable URL.
    """
    for _pageid, page in pages.items():
        title = page.get("title", "")
        ii = (page.get("imageinfo") or [])
        if not ii:
            continue
        info = ii[0]
        url = info.get("thumburl") or info.get("url")
        ext = (info.get("extmetadata") or {})

        lic = (ext.get("LicenseShortName") or {}).get("value", "")  # e.g. "CC BY-SA 4.0"
        # Sometimes it's HTML; strip tags lightly
        lic_plain = re.sub(r"<[^>]+>", "", lic).strip()

        if lic_plain and lic_plain in allowed_licenses and url:
            return {
                "title": title,
                "image_url": url,
                "license_short": lic_plain,
                "license_url": (ext.get("LicenseUrl") or {}).get("value", ""),
                "artist": re.sub(r"<[^>]+>", "", (ext.get("Artist") or {}).get("value", "")).strip(),
                "credit": re.sub(r"<[^>]+>", "", (ext.get("Credit") or {}).get("value", "")).strip(),
                "attribution_required": True if "CC BY" in lic_plain or "CC-BY" in lic_plain or "BY" in lic_plain else False,
                "source_file_page": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
            }

    return None

def fetch_image_for_place(
    place_name: str,
    session: requests.Session,
    allowed_licenses: set,
    sleep_s: float = 0.2,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (result_dict_or_None, status_string)
    """
    qname = _clean_place_name(place_name)
    if not qname:
        return None, "empty_place_name"

    # Search strategy: try a few queries.
    # "File:" keyword hint can improve results (Commons search backend behavior varies).
    queries = [
        f'intitle:"{qname}"',
        f'"{qname}"',
        f'File:"{qname}"',  # heuristic
    ]

    for q in queries:
        results = commons_search_files(q, session, limit=8)
        if not results:
            continue

        titles = [r["title"] for r in results if "title" in r]
        if not titles:
            continue

        # Respect rate limits a bit
        time.sleep(sleep_s)

        info = commons_imageinfo(titles, session, iiurlwidth=1200)
        pages = info.get("query", {}).get("pages", {})
        best = pick_best_candidate(pages, allowed_licenses)
        if best:
            return best, "ok"

    return None, "no_allowed_image_found"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places_csv", default="data/metadata_raw/MetaV_Places.csv")
    ap.add_argument("--out_csv", default="data/place_images_commons.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (process all places)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls (seconds)")
    ap.add_argument("--allowed_licenses_json", default="", help="Optional JSON array of allowed LicenseShortName values")
    args = ap.parse_args()

    places_csv = Path(args.places_csv)
    out_csv = Path(args.out_csv)

    df = pd.read_csv(places_csv)
    if "PlaceID" not in df.columns or "PlaceName" not in df.columns:
        raise ValueError("MetaV_Places.csv must include PlaceID and PlaceName columns")

    allowed = set(DEFAULT_ALLOWED_LICENSES)
    if args.allowed_licenses_json.strip():
        allowed = set(json.loads(args.allowed_licenses_json))

    rows_out = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "BibleBoard-PlaceImageFetcher/1.0 (contact: you@example.com)"
    })

    n = 0
    for r in df.itertuples(index=False):
        place_id = int(getattr(r, "PlaceID"))
        place_name = str(getattr(r, "PlaceName") or "")

        n += 1
        if args.limit and n > args.limit:
            break

        result, status = fetch_image_for_place(
            place_name=place_name,
            session=session,
            allowed_licenses=allowed,
            sleep_s=args.sleep,
        )

        out = {
            "place_id": place_id,
            "place_name": place_name,
            "status": status,
            "image_url": "",
            "license_short": "",
            "license_url": "",
            "artist": "",
            "credit": "",
            "attribution_required": "",
            "source_file_page": "",
        }

        if result:
            out.update(result)

        rows_out.append(out)

        if n % 50 == 0:
            print(f"Processed {n} places...")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"✅ Wrote: {out_csv}  (rows: {len(rows_out)})")

if __name__ == "__main__":
    main()