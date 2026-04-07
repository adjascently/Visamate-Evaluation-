import os
import json
import pandas as pd

print("\n" + "="*80)
print("MERGING GPT-4o MANUAL EVALUATION BATCHES (JUDGED BY CLAUDE)")
print("="*80 + "\n")

# Folder containing GPT-4o evaluated JSON batches
BASE_DIR = "scripts/manual_evaluation/gpt_judged_by_claude"

batch_files = sorted([
    f for f in os.listdir(BASE_DIR)
    if f.lower().endswith(".json")
])

print(f"Found {len(batch_files)} batch files:")
for bf in batch_files:
    print(" -", os.path.join(BASE_DIR, bf))

all_rows = []


def load_any_format(path):
    """Handles both list-based JSON and dict-with-key JSON."""
    with open(path, "r") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
    except:
        raise ValueError(f"❌ Could not parse JSON in {path}")

    # CASE 1 — list of evaluation objects
    if isinstance(data, list):
        return data

    # CASE 2 — dict containing evaluations (common)
    if isinstance(data, dict) and "evaluations" in data:
        return data["evaluations"]

    # CASE 3 — unexpected dict structure
    raise ValueError(f"❌ JSON format not recognized in {path}")


for bf in batch_files:
    path = os.path.join(BASE_DIR, bf)
    print(f"\n🔄 Reading {path} ...")

    data_list = load_any_format(path)

    for item in data_list:
        # Normalize potential variations from Claude outputs
        qid = item.get("qid")
        label = item.get("label") or item.get("classification")
        halluc = item.get("hallucination_score")
        correct = item.get("correctness_score")
        sim = item.get("semantic_similarity")
        explanation = item.get("explanation") or item.get("reasoning")

        all_rows.append({
            "qid": qid,
            "model": "GPT4o",
            "label": label,
            "hallucination_score": halluc,
            "correctness_score": correct,
            "semantic_similarity": sim,
            "explanation": explanation
        })

df = pd.DataFrame(all_rows)

OUTPUT = "results/manual_gpt4o_eval.csv"
df.to_csv(OUTPUT, index=False)

print("\n" + "="*80)
print("✅ MERGE COMPLETE")
print("="*80)
print(f"Saved → {OUTPUT}\n")
