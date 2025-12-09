import csv
import os

path = os.path.join("data", "metadata_raw", "place_verse.csv")

print("Scanning place_verse.csv for 1 Corinthians...")
with open(path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    found_count = 0
    for row in reader:
        ref = row.get('reference_id', '')
        # Check if this row is for 1 Corinthians Chapter 1
        # We look for "1CO", "1 Cor", or "1 Corinthians"
        if "1:2" in ref and ("Cor" in ref or "1CO" in ref):
            print(f"FOUND ROW: {row}")
            found_count += 1
            if found_count > 3: break

if found_count == 0:
    print("❌ No matches found. The book name might be formatted differently.")