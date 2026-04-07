import pandas as pd
import os
import re

print("\n" + "="*80)
print(" CLEAN MERGE: VisaMate + GPT-4o + Claude (600 manual evaluations)")
print("="*80)

# ------------------------------
# Helper: normalize QID
# ------------------------------
def normalize_qid(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()

    if x.startswith("q"):
        x = x[1:]

    digits = re.findall(r"\d+", x)
    if digits:
        return int(digits[0])
    return None

# ------------------------------
# Correct absolute path
# ------------------------------
base = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/"

files = {
    "VisaMate": os.path.join(base, "manual_visamate_eval.csv"),
    "GPT4o":    os.path.join(base, "manual_gpt4o_eval.csv"),
    "Claude":   os.path.join(base, "manual_claude_eval.csv"),
}

required_cols = [
    "qid", "label",
    "hallucination_score",
    "correctness_score",
    "semantic_similarity",
    "explanation"
]

merged_rows = []

# ------------------------------
# Load each model CSV
# ------------------------------
for model, path in files.items():
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    print(f"\n📥 Loading {model} → {path}")
    df = pd.read_csv(path)

    df.columns = [c.lower().strip() for c in df.columns]

    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Missing '{col}' in {model}. Creating blank column.")
            df[col] = None

    df["qid"] = df["qid"].apply(normalize_qid)
    df["model"] = model

    merged_rows.append(df)

if not merged_rows:
    print("\n❌ ERROR: No evaluation files loaded.")
    quit()

merged = pd.concat(merged_rows, ignore_index=True)
merged = merged.dropna(subset=["qid"])
merged["qid"] = merged["qid"].astype(int)
merged = merged.sort_values(["model", "qid"]).reset_index(drop=True)

output_path = os.path.join(base, "manual_all_models_eval_FINAL.csv")
merged.to_csv(output_path, index=False)

print("\n✅ MERGE SUCCESSFUL!")
print(f"📄 Saved → {output_path}")

print("\nPreview:")
print(merged.head(12).to_string(index=False))

print("\n" + "="*80)
print(" MODEL COUNTS (should each be 200)")
print("="*80)
print(merged["model"].value_counts())
