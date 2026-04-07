import os
import json
import pandas as pd

print("\n" + "="*80)
print("MERGING VISAMATE MANUAL EVALUATION BATCHES (JUDGED BY GPT-4o)")
print("="*80)

BASE_DIR = "scripts/manual_evaluation/visamate_judged_by_gpt"

# --------------------------------------------------------------------
# SAFETY: Load batch file that may have formatting issues
# --------------------------------------------------------------------
def load_json_safe(path):
    with open(path, "r") as f:
        raw = f.read()

    raw = raw.strip()

    # Remove accidental trailing commas
    raw = raw.replace(",]", "]").replace(",}", "}")

    try:
        return json.loads(raw)
    except Exception as e:
        print(f"❌ ERROR parsing {path}: {e}")
        raise

# --------------------------------------------------------------------
# Locate all batch files
# --------------------------------------------------------------------
batch_files = sorted([
    os.path.join(BASE_DIR, f)
    for f in os.listdir(BASE_DIR)
    if f.lower().endswith(".json")
])

print(f"\nFound {len(batch_files)} batch files:")
for bf in batch_files:
    print(" •", bf)

if len(batch_files) == 0:
    print("\n❌ No batch files found. Make sure they are in:", BASE_DIR)
    exit()

# --------------------------------------------------------------------
# Merge all batches
# --------------------------------------------------------------------
all_rows = []

for bf in batch_files:
    print(f"\n🔄 Reading {bf}...")
    data = load_json_safe(bf)

    # Support two formats:
    # { "evaluations": [ ... ] }
    # [ ... ]
    if isinstance(data, dict) and "evaluations" in data:
        items = data["evaluations"]
    elif isinstance(data, list):
        items = data
    else:
        print(f"⚠️ Unexpected format in {bf}, skipping.")
        continue

    for item in items:
        row = {
            "qid": item.get("qid"),
            "label": item.get("label"),
            "hallucination_score": item.get("hallucination_score"),
            "correctness_score": item.get("correctness_score"),
            "semantic_similarity": item.get("semantic_similarity"),
            "explanation": item.get("explanation"),
            "model": "VisaMate"
        }
        all_rows.append(row)

# --------------------------------------------------------------------
# Save merged CSV
# --------------------------------------------------------------------
df = pd.DataFrame(all_rows)

output_path = "results/manual_visamate_eval.csv"
df.to_csv(output_path, index=False)

print("\n" + "="*80)
print("✅ MERGE COMPLETE")
print("="*80)
print(f"Saved → {output_path}")
print(f"Total rows: {len(df)}\n")
