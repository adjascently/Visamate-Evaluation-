import pandas as pd
import re

print("\n" + "="*80)
print("MANUAL vs AUTOMATED METRIC COMPARISON")
print("="*80)

# -----------------------------
# Load files
# -----------------------------
manual = pd.read_csv("results/manual_all_models_eval.csv")
auto   = pd.read_csv("results/quantitative_results.csv")

# -----------------------------
# CLEAN QID COLUMN (Fix “Q1”, “ q12 ”, etc.)
# -----------------------------
def clean_qid(x):
    if pd.isna(x):
        return None
    # Extract only digits
    digits = re.findall(r"\d+", str(x))
    if len(digits) == 0:
        return None
    return int(digits[0])

manual["qid"] = manual["qid"].apply(clean_qid)
auto["qid"]   = auto["qid"].apply(clean_qid)

# Drop rows with missing qid after cleaning
manual = manual.dropna(subset=["qid"])
auto   = auto.dropna(subset=["qid"])

manual["qid"] = manual["qid"].astype(int)
auto["qid"]   = auto["qid"].astype(int)

# -----------------------------
# Normalize model names
# -----------------------------
manual["model"] = manual["model"].str.strip()
auto["model"]   = auto["model"].str.strip()

# -----------------------------
# Merge
# -----------------------------
merged = pd.merge(
    manual,
    auto,
    how="left",
    on=["qid", "model"],
    suffixes=("_manual", "_auto")
)

# -----------------------------
# Identify correct auto hallucination score column
# -----------------------------
auto_col = None
for c in merged.columns:
    if c.startswith("hallucination_score") and c.endswith("_auto"):
        auto_col = c

if auto_col is None:
    raise ValueError("❌ Could not find automatic hallucination score column.")

print(f"\n✅ Using hallucination column: {auto_col}")

# -----------------------------
# Manual correctness → binary
# -----------------------------
merged["manual_correct"] = merged["label"].apply(
    lambda x: 1 if str(x).strip() == "correct" else 0
)

# -----------------------------
# Auto correctness rule
# -----------------------------
merged["auto_correct"] = merged[auto_col].apply(
    lambda x: 1 if x < 0.3 else 0
)

# -----------------------------
# Confusion matrix counts
# -----------------------------
TP = len(merged[(merged.manual_correct == 1) & (merged.auto_correct == 1)])
FN = len(merged[(merged.manual_correct == 1) & (merged.auto_correct == 0)])
FP = len(merged[(merged.manual_correct == 0) & (merged.auto_correct == 1)])
TN = len(merged[(merged.manual_correct == 0) & (merged.auto_correct == 0)])

print("\nCONFUSION MATRIX (Manual vs Automatic)\n")
print(f"Manual Correct & Auto Correct:     {TP}")
print(f"Manual Correct & Auto Incorrect:   {FN}")
print(f"Manual Incorrect & Auto Correct:   {FP}")
print(f"Manual Incorrect & Auto Incorrect: {TN}")

# -----------------------------
# Precision, Recall, F1
# -----------------------------
precision = TP / (TP + FP) if TP+FP > 0 else 0
recall    = TP / (TP + FN) if TP+FN > 0 else 0
f1        = 2*precision*recall / (precision+recall) if precision+recall > 0 else 0

print("\nPRECISION / RECALL / F1")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# -----------------------------
# Save merged result
# -----------------------------
merged.to_csv("results/manual_vs_auto_merged.csv", index=False)
print("\n📄 Saved → results/manual_vs_auto_merged.csv\n")
