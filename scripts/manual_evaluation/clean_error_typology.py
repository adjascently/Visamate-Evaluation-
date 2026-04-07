import pandas as pd
import re
import os

INPUT_FILE = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/error_typology_candidates.csv"
OUTPUT_FILE = "/Users/jasmainekhale/Desktop/visamate-evaluation/results/error_typology_cleaned.csv"


print("\n=====================================================================")
print(" CLEANING ERROR TYPOLOGY FILE")
print("=====================================================================\n")

# ---------------------------------------------------------
# LOAD CSV
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# ---------------------------------------------------------
# FIX QID FORMAT (“Q98” → 98)
# ---------------------------------------------------------
def clean_qid(x):
    if isinstance(x, str):
        num = re.findall(r'\d+', x)
        return int(num[0]) if num else None
    return int(x)

df["qid"] = df["qid"].apply(clean_qid)

# ---------------------------------------------------------
# AUTO-DETECT ERROR TYPE
# ---------------------------------------------------------

def classify_error(explanation):
    text = explanation.lower()

    # Wrong number (days, hours, limits)
    if any(word in text for word in ["90 days", "60 days", "20 hours", "incorrect duration", "wrong number"]):
        return "wrong_number"

    # Fabricated rule
    if any(word in text for word in ["made up", "fabricated", "not in regulation", "invented"]):
        return "fabricated_rule"

    # Missing restriction or missing requirement
    if "missing" in text or "does not mention" in text or "omits" in text:
        return "missing_requirement"

    # Incorrect USCIS process
    if any(word in text for word in ["incorrect process", "wrong procedure", "misstates process", "incorrectly explains"]):
        return "incorrect_process"

    # Contradiction with SEVP or policy
    if "contradiction" in text or "opposite" in text or "misaligned with SEVP" in text:
        return "contradiction"

    # Vague / generic answer
    if "generic" in text or "not specific" in text or "lacks detail" in text:
        return "vague_or_generic"

    # Overconfident unsafe guidance
    if "unsafe" in text or "confidently wrong" in text:
        return "overconfident_unsafe"

    # Default fallback
    return "other"

df["error_type"] = df["explanation"].apply(classify_error)

# ---------------------------------------------------------
# CREATE CLEAN SUMMARY
# ---------------------------------------------------------
def summarize(row):
    return f"Q{row['qid']} - {row['model']} - {row['error_type']}"

df["summary"] = df.apply(summarize, axis=1)

# ---------------------------------------------------------
# SAVE FILE
# ---------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved cleaned file → {OUTPUT_FILE}\n")

print("Preview:")
print(df.head(10).to_string(index=False))
