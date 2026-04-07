import pandas as pd
import re

# ---------------------------------------------------------
# LOAD QUANTITATIVE RESULTS
# ---------------------------------------------------------
df = pd.read_csv("../results/quantitative_results.csv")

# ---------------------------------------------------------
# STEP 1 — Normalize QID (CRITICAL FIX)
# Converts Q12 → 12, q88 → 88, 21.0 → 21
# ---------------------------------------------------------
def normalize_qid(val):
    try:
        if isinstance(val, str):
            digits = re.findall(r"\d+", val)
            if digits:
                return int(digits[0])
        return int(float(val))
    except:
        return None

df["qid"] = df["qid"].apply(normalize_qid)

# Remove old incorrect category column (if present)
if "category" in df.columns:
    df = df.drop(columns=["category"])

# ---------------------------------------------------------
# STEP 2 — Define Category Ranges
# ---------------------------------------------------------
categories = {
    "Initial F-1 Entry": range(1, 11),
    "CPT": range(11, 21),
    "OPT Application": range(21, 31),
    "STEM OPT": range(31, 41),
    "Job Loss & Unemployment": range(41, 51),
    "Travel & Re-entry": range(51, 61),
    "H-1B Transition": range(61, 71),
    "Remote Work": range(71, 81),
    "Multiple Jobs": range(81, 91),
    "Grace Period": range(91, 101),
    "Administrative Issues": range(101, 121),
    "Academic Issues": range(121, 141),
    "F-2 Dependents": range(141, 161),
    "Tax & Financial": range(161, 181),
    "Long-term Planning": range(181, 201),
}

def assign_category(qid):
    for cat, r in categories.items():
        if qid in r:
            return cat
    return "Unknown"

df["category"] = df["qid"].apply(assign_category)

# ---------------------------------------------------------
# STEP 3 — Compute Category-Level Averages
# ---------------------------------------------------------
results = []

for model in ["VisaMate", "GPT4o", "Claude"]:
    model_rows = df[df["model"] == model]

    for cat in categories.keys():
        subset = model_rows[model_rows["category"] == cat]
        if len(subset) == 0:
            continue

        results.append({
            "model": model,
            "category": cat,
            "question_count": len(subset),
            "hallucination_score_avg": subset["hallucination_score"].mean(),
            "correctness_avg": subset["correctness_score"].mean(),
            "completeness_avg": subset["completeness"].mean(),
            "relevance_avg": subset["relevance"].mean(),
            "regulatory_avg": subset["regulatory_consistency"].mean(),
            "semantic_similarity_avg": subset["semantic_similarity"].mean(),
        })

category_df = pd.DataFrame(results)

# Save output
category_df.to_csv("../results/category_breakdown.csv", index=False)

# ---------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------
print("\n" + "="*80)
print("CATEGORY-LEVEL PERFORMANCE BREAKDOWN (UPDATED)")
print("="*80)

for cat in categories.keys():
    subset = category_df[category_df["category"] == cat]
    if len(subset) == 0:
        continue

    print(f"\n{cat}")
    print("-" * 60)
    for _, row in subset.iterrows():
        print(
            f"  {row['model']:12s} | "
            f"H: {row['hallucination_score_avg']:.2f} | "
            f"C: {row['completeness_avg']:.3f} | "
            f"R: {row['relevance_avg']:.3f} | "
            f"Reg: {row['regulatory_avg']:.3f}"
        )

print("\n" + "="*80)
print("Updated category breakdown saved → ../results/category_breakdown.csv")
print("="*80 + "\n")
