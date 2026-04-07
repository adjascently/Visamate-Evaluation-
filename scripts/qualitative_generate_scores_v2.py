import json
import csv
import os
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("\n" + "="*80)
print(" AUTOMATED QUALITATIVE EVALUATION (GPT-4o as Judge — Immigration Rubric v2)")
print("="*80)

# ============================================================
# LOAD DATA
# ============================================================
with open("data/questions.json") as f:
    QUESTIONS = json.load(f)

with open("data/gold.json") as f:
    GOLD = {item["qid"]: item["answer"] for item in json.load(f)}

models = {
    "VisaMate": "results/visamate_results.json",
    "GPT4o": "results/gpt4o_results.json",
    "Claude": "results/claude_results.json",
}

model_answers = {}
for name, path in models.items():
    with open(path) as f:
        arr = json.load(f)
        model_answers[name] = {x["qid"]: x["answer"] for x in arr}

# ============================================================
# SAMPLE QIDS FOR QUALITATIVE EVAL
# ============================================================
sample_qids = [
    1, 5, 12, 18,         # Initial F-1 Entry
    22, 28,               # CPT
    35, 42, 48,           # OPT
    55, 60,               # STEM OPT
    75, 78,               # Travel
    95, 98,               # Grace Period
    105, 110,             # Multiple Jobs
    125, 130,             # Tax
    145                   # Admin Issues
]

print(f"\n📋 Evaluating {len(sample_qids)} questions × 3 models = {len(sample_qids)*3} responses.\n")

# ============================================================
# IMMIGRATION-ADVISOR RUBRIC
# ============================================================
rubric = """
Evaluate the following answer strictly as an F-1 immigration advisor.
Score each criterion from 1–5.

1. FACTUAL ACCURACY & REGULATORY ALIGNMENT
   5 = Fully consistent with USCIS/SEVP regulations, no errors
   3 = Mostly correct with mild imprecision
   1 = Incorrect or contradictory

2. COMPLETENESS OF COMPLIANCE REQUIREMENTS
   5 = Covers all required steps, deadlines, restrictions
   3 = Covers main points but missing some details
   1 = Major omissions

3. CITATION PRECISION & SOURCE GROUNDING
   5 = References correct forms/regulations (I-20, I-94, 8 CFR …)
   3 = Mentions general sources
   1 = No citations or vague

4. CLARITY & CONCISENESS
   5 = Very clear and concise
   3 = Understandable but wordy
   1 = Hard to follow or verbose

5. SAFETY, RISK AWARENESS, DISCRETION
   5 = Includes appropriate cautions (DSO variation, policy risk)
   3 = Partially cautions
   1 = Overconfident or risky

6. RELEVANCE TO QUESTION
   5 = Direct, no tangents
   3 = Mostly relevant
   1 = Off-topic

7. ACTIONABILITY & NEXT-STEP GUIDANCE
   5 = Clear, correct next steps
   3 = Partial guidance
   1 = No actionable steps

Respond ONLY with JSON in this exact format:

{
  "factual_accuracy": 5,
  "completeness": 5,
  "citation_precision": 5,
  "clarity": 4,
  "safety": 5,
  "relevance": 4,
  "actionability": 5,
  "brief_reasoning": "One short sentence explanation."
}
"""

# ============================================================
# GPT-BASED JUDGE CALL
# ============================================================
def score_with_gpt(question, model_answer, gold_answer):
    prompt = f"""
You are an expert U.S. immigration advisor evaluating AI-generated guidance.

QUESTION:
{question}

GOLD-STANDARD (USCIS-based) REFERENCE:
{gold_answer}

MODEL ANSWER TO EVALUATE:
{model_answer}

{rubric}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.choices[0].message.content.strip()

        # Strip code fences
        text = re.sub(r"```json|```", "", text).strip()

        # Extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            print("⚠️ Failed to parse JSON, raw output:", text)
            return None

        return json.loads(match.group())

    except Exception as e:
        print("❌ GPT error:", str(e))
        return None


# ============================================================
# RUN SCORING
# ============================================================
all_rows = []

for qid in tqdm(sample_qids, desc="Scoring"):
    q = QUESTIONS[qid-1]
    gold = GOLD[qid]

    for model in models.keys():
        ans = model_answers[model][qid]

        scores = score_with_gpt(q, ans, gold)
        if not scores:
            continue

        all_rows.append({
            "qid": qid,
            "model": model,
            **scores
        })

# ============================================================
# SAVE RESULT CSV
# ============================================================
out_path = "results/qualitative_evaluation_v2.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n✅ Saved detailed scores → {out_path}")

# ============================================================
# SUMMARY
# ============================================================
df = pd.DataFrame(all_rows)
criteria = [
    "factual_accuracy",
    "completeness",
    "citation_precision",
    "clarity",
    "safety",
    "relevance",
    "actionability",
]

summary = df.groupby("model")[criteria].mean()
summary["overall"] = summary.mean(axis=1)

summary_path = "results/qualitative_summary_v2.csv"
summary.to_csv(summary_path)

print(f"✅ Saved model summary → {summary_path}\n")

print("="*80)
print(" QUALITATIVE EVALUATION COMPLETE (IMMIGRATION RUBRIC v2)")
print("="*80)
