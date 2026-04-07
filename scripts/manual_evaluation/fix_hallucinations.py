import pandas as pd

# ================================================================
# Load NEW merged manual evaluation + cleaned typology
# ================================================================
manual = pd.read_csv(
    "/Users/jasmainekhale/Desktop/visamate-evaluation/results/manual_all_models_eval_FINAL.csv"
)
typology = pd.read_csv(
    "/Users/jasmainekhale/Desktop/visamate-evaluation/results/error_typology_cleaned.csv"
)

# ================================================================
# Normalize QID
# ================================================================
manual["qid"] = manual["qid"].astype(str).str.replace("Q", "").str.strip()
typology["qid"] = typology["qid"].astype(str).str.replace("Q", "").str.strip()

# ================================================================
# Merge manual evaluation with error typology
# ================================================================
merged = pd.merge(
    manual,
    typology[["qid", "model", "error_type"]],
    on=["qid", "model"],
    how="left"
)

# ================================================================
# STRONGER hallucination detection logic
# ================================================================
def compute_hallucination(row):
    label = str(row["label"]).lower().strip()
    error = str(row["error_type"]).strip()
    correctness = float(row["correctness_score"])
    similarity = float(row["semantic_similarity"])

    # ------------------------------------------------------------
    # 1. Explicit manual hallucination
    # ------------------------------------------------------------
    if label == "hallucination":
        return 1

    # ------------------------------------------------------------
    # 2. Strong hallucination error types
    # ------------------------------------------------------------
    STRONG_ERRORS = {
        "fabricated_rule",
        "wrong_number",
        "contradiction",
        "incorrect_process",
        "overconfident_unsafe",
        "unsafe_overconfident"
    }

    if error in STRONG_ERRORS:
        return 1

    # ------------------------------------------------------------
    # 3. Missing requirement ALWAYS hallucination in F-1 context
    # ------------------------------------------------------------
    if error == "missing_requirement":
        return 1

    # ------------------------------------------------------------
    # 4. Incorrect USCIS process ALWAYS hallucination
    # ------------------------------------------------------------
    if error == "incorrect_process":
        return 1

    # ------------------------------------------------------------
    # 5. Vague/generic + low-ish similarity (< 0.7)
    # ------------------------------------------------------------
    if error == "vague_or_generic" and similarity < 0.7:
        return 1

    # ------------------------------------------------------------
    # 6. correctness_score < 0.65 → factually unreliable
    # ------------------------------------------------------------
    if correctness < 0.65:
        return 1

    # ------------------------------------------------------------
    # 7. partially_correct + strong error → hallucination
    # ------------------------------------------------------------
    if label == "partially_correct" and error in STRONG_ERRORS:
        return 1

    return 0

# ================================================================
# Compute corrected hallucination values
# ================================================================
merged["hallucination_corrected"] = merged.apply(compute_hallucination, axis=1)

# ================================================================
# Save corrected output
# ================================================================
output_path = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/manual_all_models_eval_corrected.csv"
merged.to_csv(output_path, index=False)

print("\n========================================================")
print(" CORRECTED HALLUCINATION COUNTS (STRONGER METRIC)")
print("========================================================\n")

summary = merged.groupby("model")["hallucination_corrected"].sum()
print(summary)

print("\nSaved corrected file to:")
print(output_path)
