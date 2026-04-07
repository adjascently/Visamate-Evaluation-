import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from openai import OpenAI
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

# Category mapping (based on qid ranges)
CATEGORIES = {
    "Initial F-1 Entry": range(1, 11),
    "CPT": range(11, 21),
    "OPT": range(21, 31),
    "STEM OPT": range(31, 41),
    "Job Loss": range(41, 51),
    "Travel": range(51, 61),
    "H-1B Transition": range(61, 71),
    "Remote Work": range(71, 81),
    "Multiple Jobs": range(81, 91),
    "Grace Period": range(91, 101),
    "Admin Issues": range(101, 121),
    "Academic Issues": range(121, 141),
    "Dependents (F-2)": range(141, 161),
    "Tax": range(161, 181),
    "Long-term Planning": range(181, 201),
}

def get_category(qid):
    for category, q_range in CATEGORIES.items():
        if qid in q_range:
            return category
    return "Unknown"


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------
def get_embedding(text):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return np.array(resp.data[0].embedding)

def cosine_sim(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))


# ------------------------------------------------------------
# Regulatory consistency
# ------------------------------------------------------------
KEYWORDS = ["I-20", "I-94", "USCIS", "SEVP", "DHS", "CBP", "DSO", "CPT", "OPT", "STEM OPT"]

def regulatory_consistency(text):
    t = text.upper()
    hits = sum(1 for kw in KEYWORDS if kw in t)
    return hits / len(KEYWORDS)


# ------------------------------------------------------------
# Load evaluation results (from hallucination script)
# ------------------------------------------------------------
df = pd.read_csv("results/evaluation_results.csv")

with open("data/questions.json") as f:
    QUESTIONS = json.load(f)

with open("data/gold.json") as f:
    GOLD = {x["qid"]: x["answer"] for x in json.load(f)}

model_answers = {}
for model in ["VisaMate", "GPT4o", "Claude"]:
    with open(f"results/{model.lower()}_results.json") as f:
        model_answers[model] = {d["qid"]: d["answer"] for d in json.load(f)}


# ------------------------------------------------------------
# COMPUTE METRICS
# ------------------------------------------------------------
rows = []

print("\nComputing quantitative metrics...\n")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    qid = int(row["qid"])
    model = row["model"]

    model_answer = model_answers[model][qid]
    gold_answer = GOLD[qid]
    question = QUESTIONS[qid - 1]

    # Embeddings
    emb_model = get_embedding(model_answer)
    emb_gold = get_embedding(gold_answer)
    emb_question = get_embedding(question)

    # Metrics
    completeness = cosine_sim(emb_model, emb_gold)            # model → gold
    relevance = cosine_sim(emb_model, emb_question)           # model → question
    semantic_sim = completeness                               # SAME SCORE (semantic vs gold)
    reg = regulatory_consistency(model_answer)

    correctness_score = 1 - row["hallucination_score"]

    rows.append([
        model,
        qid,
        row["hallucination_score"],
        correctness_score,
        completeness,
        relevance,
        reg,
        semantic_sim,
        get_category(qid)
    ])


# ------------------------------------------------------------
# SAVE FINAL RESULTS
# ------------------------------------------------------------
out_df = pd.DataFrame(rows, columns=[
    "model", "qid",
    "hallucination_score", "correctness_score",
    "completeness", "relevance",
    "regulatory_consistency",
    "semantic_similarity",
    "category"
])

out_df.to_csv("results/quantitative_results.csv", index=False)
print("\n✅ Saved: results/quantitative_results.csv")

# Model summary
model_summary = out_df.groupby("model").mean(numeric_only=True)
model_summary.to_csv("results/model_summary.csv")
print("✅ Saved: results/model_summary.csv")

# Category breakdown
cat_breakdown = out_df.groupby(["category", "model"]).mean(numeric_only=True)
cat_breakdown.to_csv("results/category_breakdown.csv")
print("✅ Saved: results/category_breakdown.csv")

# Statistical tests
tests = []
for metric in ["hallucination_score", "correctness_score",
               "completeness", "relevance", "semantic_similarity"]:
    v = out_df[out_df["model"] == "VisaMate"][metric]
    g = out_df[out_df["model"] == "GPT4o"][metric]
    c = out_df[out_df["model"] == "Claude"][metric]

    u1, p1 = mannwhitneyu(v, g, alternative="less")
    u2, p2 = mannwhitneyu(v, c, alternative="less")

    tests.append([metric, float(p1), float(p2)])

pd.DataFrame(tests, columns=[
    "metric", "p_value_vs_gpt4o", "p_value_vs_claude"
]).to_csv("results/statistical_tests.csv", index=False)

print("✅ Saved: results/statistical_tests.csv")
print("\n🎉 QUANTITATIVE ANALYSIS COMPLETE!\n")
