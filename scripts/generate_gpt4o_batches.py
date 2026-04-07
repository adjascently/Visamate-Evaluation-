import json

# ------------------------------
# Load data
# ------------------------------
with open("data/questions.json") as f:
    QUESTIONS = json.load(f)

with open("data/gold.json") as f:
    GOLD = {item["qid"]: item["answer"] for item in json.load(f)}

with open("results/gpt4o_results.json") as f:
    GPT4O = {item["qid"]: item["answer"] for item in json.load(f)}

# ------------------------------
# Generate batch
# ------------------------------
def generate_batch(batch_num, batch_size=20):
    start = (batch_num - 1) * batch_size + 1
    end = start + batch_size - 1

    print(f"\n================ BATCH {batch_num} (Q{start}–Q{end}) ================\n")

    for qid in range(start, end + 1):
        question = QUESTIONS[qid - 1]
        gold = GOLD[qid]
        gpt = GPT4O[qid]

        print(f"---------------- Q{qid} ----------------\n")
        print(f"QUESTION:\n{question}\n")
        print(f"GOLD ANSWER:\n{gold}\n")
        print(f"GPT-4o ANSWER:\n{gpt}\n")
        print("========================================\n")


# ------------------------------
# Run script
# ------------------------------
if __name__ == "__main__":
    print("Enter batch number (1–10):")
    batch = int(input().strip())
    generate_batch(batch)
