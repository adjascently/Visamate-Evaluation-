import pandas as pd
import os

print("\n" + "="*80)
print("GENERATING ERROR TYPOLOGY CANDIDATE TABLE")
print("="*80)

INPUT_FILE = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/manual_all_models_eval_FINAL.csv"



OUTPUT_FILE = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/error_typology_candidates.csv"


if not os.path.exists(INPUT_FILE):
    print(f"\n❌ Missing file: {INPUT_FILE}")
    print("Run merge_all_manual_eval.py first.")
    exit(1)

# Load manual evaluation
df = pd.read_csv(INPUT_FILE)

# Keep only incorrect or partially correct
error_df = df[df['label'].isin(['incorrect', 'partially_correct', 'hallucination'])].copy()

# Sort by severity (worst first)
error_df = error_df.sort_values(by=['hallucination_score', 'correctness_score'])

# Add empty column for human tagging
error_df['error_type'] = ""   # you will fill this
error_df['summary'] = ""      # optional short summary

# Select important columns
cols = [
    "qid", "model", "label", "hallucination_score", "correctness_score",
    "semantic_similarity", "explanation", "error_type", "summary"
]

error_df = error_df[cols]

# Save
error_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Saved candidate file → {OUTPUT_FILE}")
print("\nNext steps:")
print("1. Open the CSV file.")
print("2. Fill in error_type with one of the following:")
print("   - wrong_number")
print("   - fabricated_rule")
print("   - missing_restriction")
print("   - incorrect_uscis_process")
print("   - contradiction_sevp")
print("   - vague_generic")
print("   - unsafe_overconfident")
print("3. Add a 1–2 line summary of the error.")
print("4. Save as error_typology_filled.csv\n")

print("="*80)
print("READY FOR HUMAN TAGGING")
print("="*80)
