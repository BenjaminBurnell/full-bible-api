# merge_tokens.py
import csv, os, argparse

FIELDS = ["book_code","chapter","verse","token_index","surface","lemma","translit","gloss","morph","strong"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", required=True, help="NT normalized CSV")
    ap.add_argument("--ot", required=True, help="OT normalized CSV")
    ap.add_argument("--out", default=os.path.join("data","all_tokens.csv"))
    args = ap.parse_args()

    seen, total = set(), 0
    with open(args.out, "w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=FIELDS)
        w.writeheader()

        for path in (args.nt, args.ot):
            with open(path, "r", encoding="utf-8-sig", newline="") as f_in:
                r = csv.DictReader(f_in)
                missing = set(FIELDS) - set(r.fieldnames or [])
                if missing:
                    raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
                for row in r:
                    key = (row["book_code"], row["chapter"], row["verse"], row["token_index"])
                    if key in seen:
                        continue
                    seen.add(key)
                    w.writerow({k: row.get(k, "") for k in FIELDS})
                    total += 1

    print(f"✅ Merged → {args.out}  (rows: {total:,})")

if __name__ == "__main__":
    main()