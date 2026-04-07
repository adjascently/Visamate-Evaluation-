import pandas as pd

print("\n" + "="*80)
print("GENERATING FINAL SUMMARY METRICS (WITH CORRECTED HALLUCINATIONS)")
print("="*80)

# ------------------------------------------------------------
# Load the corrected manual evaluation file
# ------------------------------------------------------------
df = pd.read_csv(
    "/Users/jasmainekhale/Desktop/visamate-evaluation/results/manual_all_models_eval_corrected.csv"
)

# Normalize labels
df["label"] = df["label"].str.lower().str.strip()

# Manual correctness weighting
label_to_accuracy = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
    "hallucination": 0.0
}

# Compute manual accuracy score
df["manual_accuracy"] = df["label"].map(label_to_accuracy)

# ------------------------------------------------------------
# Summary per model
# ------------------------------------------------------------
summary = df.groupby("model").agg(
    total_questions=("qid", "count"),
    correct=("label", lambda x: (x == "correct").sum()),
    partially_correct=("label", lambda x: (x == "partially_correct").sum()),
    incorrect=("label", lambda x: (x == "incorrect").sum()),
    manual_hallucinations=("label", lambda x: (x == "hallucination").sum()),
    corrected_hallucinations=("hallucination_corrected", "sum"),
    avg_correctness_score=("correctness_score", "mean"),
    avg_hallucination_score=("hallucination_score", "mean"),
    avg_semantic_similarity=("semantic_similarity", "mean"),
    overall_manual_accuracy=("manual_accuracy", "mean")
).reset_index()

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------
output_path = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/manual_eval_summary.csv"
summary.to_csv(output_path, index=False)

print("\n✅ Summary saved to:", output_path)
print("\nSummary preview:\n")
print(summary)

print("\n" + "="*80)
print("DONE")
print("="*80)
