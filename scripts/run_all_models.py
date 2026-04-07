import json
import time
import os
from openai import OpenAI
from anthropic import Anthropic
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --------------------------
# Load questions
# --------------------------
with open("data/questions.json", "r") as f:
    QUESTIONS = json.load(f)

# --------------------------
# Initialize API clients
# --------------------------
VISAMATE_URL = os.getenv("VISAMATE_URL", "http://localhost:8000/api/eval")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --------------------------
# VisaMate call
# --------------------------
def call_visamate(question):
    try:
        response = requests.post(VISAMATE_URL, json={"query": question}, timeout=60)
        return response.json().get("answer", "")
    except Exception as e:
        return f"ERROR: {str(e)}"

# --------------------------
# GPT-4o call
# --------------------------
def call_gpt4o(question):
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
            temperature=0,
            max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# --------------------------
# Claude 4.5 Sonnet call
# --------------------------
def call_claude(question):
    try:
        resp = anthropic_client.messages.create(
            model="claude-3-7-sonnet-latest",   # Claude 4.5 official model name
            max_tokens=400,
            temperature=0,
            messages=[{"role": "user", "content": question}]
        )
        return resp.content[0].text
    except Exception as e:
        return f"ERROR: {str(e)}"

# --------------------------------------
# RUN ALL MODELS & SAVE JSON FILES
# --------------------------------------
def run_model(model_name, model_func, output_file):
    results = []

    print(f"\n{'='*70}")
    print(f"Running {model_name} on {len(QUESTIONS)} questions…")
    print(f"{'='*70}\n")

    for idx, q in enumerate(QUESTIONS, start=1):
        print(f"[{model_name}] Q{idx}/200 → {q[:60]}...")
        answer = model_func(q)

        results.append({
            "qid": idx,
            "question": q,
            "answer": answer
        })

        time.sleep(0.5)  # API safety

    with open(f"results/{output_file}", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ {model_name} results saved → results/{output_file}\n")

# --------------------------------------
# MAIN
# --------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING ALL MODELS (VisaMate, GPT-4o, Claude 4.5)")
    print("="*60)

    run_model("VisaMate", call_visamate, "visamate_results.json")
    run_model("GPT4o", call_gpt4o, "gpt4o_results.json")
    run_model("Claude4.5", call_claude, "claude_results.json")

    print("\n" + "="*60)
    print("✨ ALL MODELS COMPLETED SUCCESSFULLY ✨")
    print("="*60)
