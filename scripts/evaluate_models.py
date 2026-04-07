import json
import csv
import re
from tqdm import tqdm
import numpy as np
import openai
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
with open("data/questions.json") as f:
    QUESTIONS = json.load(f)

with open("data/gold.json") as f:
    GOLD = {entry["qid"]: entry["answer"] for entry in json.load(f)}

models = {
    "VisaMate": "results/visamate_results.json",
    "GPT4o": "results/gpt4o_results.json",
    "Claude": "results/claude_results.json"
}

def load_answers(path):
    with open(path) as f:
        data = json.load(f)
    return {item["qid"]: item["answer"] for item in data}

# ---------------------------------------------------------
# Embeddings (OpenAI new API)
# ---------------------------------------------------------
def get_embedding(text):
    resp = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp["data"][0]["embedding"])

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------
def extract_facts(text):
    facts = {
        "numbers": set(re.findall(r"\b\d+\b", text)),
        "forms": set(re.findall(r"\bI-\d+\b", text.upper())),
        "terms": set()
    }

    terms = [
        "F-1", "F-2", "H-1B", "CPT", "OPT", "STEM OPT", "DSO",
        "SEVIS", "USCIS", "SEVP", "DHS", "CBP", "EAD"
    ]

    T = text.upper()
    for term in terms:
        if term.upper() in T:
            facts["terms"].add(term)

    return facts

def fact_containment(model_ans, gold_ans):
    m = extract_facts(model_ans)
    g = extract_facts(gold_ans)

    scores = []

    # number containment
    if g["numbers"]:
        s = len(g["numbers"] & m["numbers"]) / len(g["numbers"])
        scores.append(s)

    # forms
    if g["forms"]:
        s = len(g["forms"] & m["forms"]) / len(g["forms"])
        scores.append(s)

    # terms
    if g["terms"]:
        s = len(g["terms"] & m["terms"]) / len(g["terms"])
        scores.append(s)

    # if no facts at all, fallback to 0.4
    if not scores:
        return 0.4

    return sum(scores) / len(scores)

# ---------------------------------------------------------
# Hallucination scoring (0–1)
# ---------------------------------------------------------
def hallucination_score(model_ans, gold_ans):
    # fact containment (strong signal)
    fc = fact_containment(model_ans, gold_ans)

    # embedding similarity (weak signal)
    e_model = get_embedding(model_ans)
    e_gold = get_embedding(gold_ans)
    sim = cosine_sim(e_model, e_gold)

    # final score (higher = more hallucination)
    score = (1 - fc) * 0.7 + (1 - sim) * 0.3
    return min(max(score, 0), 1)

# ---------------------------------------------------------
# Evaluate models
# ---------------------------------------------------------
rows = []

print("\nEvaluating models...\n")

for model_name, path in models.items():
    answers = load_answers(path)

    for qid, model_answer in tqdm(answers.items(), desc=f"Evaluating {model_name}"):

        gold_answer = GOLD[qid]
        question = QUESTIONS[qid - 1]

        # hallucination (0–1)
        h_score = hallucination_score(model_answer, gold_answer)

        # completeness via embeddings
        embed_model = get_embedding(model_answer)
        embed_gold = get_embedding(gold_answer)
        completeness = cosine_sim(embed_model, embed_gold)

        # relevance via embeddings
        embed_q = get_embedding(question)
        relevance = cosine_sim(embed_model, embed_q)

        # regulatory consistency
        reg_keywords = ["USCIS", "SEVP", "I-20", "I-94", "F-1", "OPT", "STEM"]
        reg = sum(kw in model_answer.upper() for kw in reg_keywords) / len(reg_keywords)

        rows.append([
            model_name,
            qid,
            h_score,
            completeness,
            relevance,
            reg
        ])

# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------
with open("results/evaluation_results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "qid", "hallucination_score", "completeness", "relevance", "regulatory_consistency"])
    writer.writerows(rows)

print("\n✅ Saved: results/evaluation_results.csv\n")
