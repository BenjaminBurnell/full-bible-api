import csv
import os

RAW_DIR = os.path.join("data", "metadata_raw")
FILES_TO_CHECK = ["books.csv", "person_verse.csv", "place_verse.csv"]

print("--- CHECKING CSV HEADERS ---")
for filename in FILES_TO_CHECK:
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                headers = next(reader)
                print(f"\n📁 {filename}:")
                print(f"   └── Headers: {headers}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    else:
        print(f"\n⚠️  MISSING: {filename}")