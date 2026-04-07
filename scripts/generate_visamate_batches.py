import json
import os

# ------------------------------
# Load data
# ------------------------------

# Get project root (one level up from scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load questions
with open(os.path.join(ROOT, "data/questions.json")) as f:
    QUESTIONS = json.load(f)

# Load gold answers
with open(os.path.join(ROOT, "data/gold.json")) as f:
    GOLD = {item["qid"]: item["answer"] for item in json.load(f)}

# Load VisaMate results
with open(os.path.join(ROOT, "results/visamate_results.json")) as f:
    VISAMATE = {item["qid"]: item["answer"] for item in json.load(f)}


# ------------------------------
# Generate batch
# ------------------------------
def generate_batch(batch_num, batch_size=20):
    start = (batch_num - 1) * batch_size + 1
    end = start + batch_size - 1

    print(f"\n================ VISAMATE BATCH {batch_num} (Q{start}–Q{end}) ================\n")

    for qid in range(start, end + 1):
        question = QUESTIONS[qid - 1]
        gold = GOLD[qid]
        visamate = VISAMATE[qid]

        print(f"---------------- Q{qid} ----------------\n")
        print(f"QUESTION:\n{question}\n")
        print(f"GOLD ANSWER:\n{gold}\n")
        print(f"VISAMATE ANSWER:\n{visamate}\n")
        print("========================================\n")


# ------------------------------
# Run script
# ------------------------------
if __name__ == "__main__":
    print("Enter batch number (1–10):")
    batch = int(input().strip())
    generate_batch(batch)
