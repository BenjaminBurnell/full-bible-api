import os

# Define the folder containing your CSVs
RAW_DIR = os.path.join("data", "metadata_raw")
FILES_TO_CHECK = ["books.csv", "person_verse.csv", "persons.csv"]

print("--- DIAGNOSTIC REPORT ---")

for filename in FILES_TO_CHECK:
    path = os.path.join(RAW_DIR, filename)
    print(f"\nChecking: {filename}...")
    
    if not os.path.exists(path):
        print("❌ FILE MISSING")
        continue
        
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            
            # Check for HTML (Common mistake)
            if "<!DOCTYPE" in first_line or "<html" in first_line:
                print("🚨 ERROR: This is an HTML file, not a CSV! (Redownload 'Raw' file)")
            else:
                print(f"✅ Headers: {first_line}")
                print(f"   Sample:  {second_line}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")