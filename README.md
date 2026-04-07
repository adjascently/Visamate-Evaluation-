# VisaMate Evaluation  
**Reliability, Hallucination, and Regulatory Correctness Benchmark for Immigration AI**

---

##  📌 Overview

This repository contains the complete evaluation framework for **VisaMate**, a domain-specific AI assistant designed for U.S. immigration guidance.

Unlike general-purpose LLM benchmarks, this evaluation focuses on a **high-stakes, compliance-critical domain**, where even small errors can lead to:

- SEVIS termination  
- Loss of OPT/STEM OPT authorization  
- Visa status violations  
- Inadmissibility or re-entry denial  

This project evaluates whether models are not just **fluent**, but **factually correct, policy-aligned, and safe**.

---

##  Why This Evaluation Matters

### 1. Similarity ≠ Regulatory Correctness

Immigration rules are **discrete and rule-based**. Even minor deviations can invalidate an answer:

- “90 days unemployment” vs “120 days unemployment”  
- Missing SEVIS reporting requirements  
- Incorrect filing timelines  

Despite this, semantic similarity metrics still score such answers highly (~0.7–0.9), making them misleading.

---

### 2. Fluent Answers Can Still Be Unsafe

LLMs often generate:

- Confident tone  
- Detailed explanations  
- Step-by-step reasoning  

But may include:

- Fabricated USCIS rules  
- Incorrect eligibility criteria  
- Nonexistent procedures  

These hallucinations are especially dangerous in immigration contexts.

---

### 3. Automated Metrics Are Not Reliable

Validation results show:

| Metric | Value |
|------|------|
| Precision | 0.789 |
| Recall | 0.623 |
| F1 Score | 0.697 |

👉 This is insufficient for high-stakes domains where ≥ 0.90 reliability is required.

---

## 🎯 Project Goals

This evaluation framework aims to:

- Measure **regulatory correctness** of model outputs  
- Detect and classify **hallucinations**  
- Evaluate **grounding and policy alignment**  
- Compare **domain-specific vs general LLMs**  
- Demonstrate the limitations of **similarity-based evaluation**  

---

## 🧪 Dataset & Benchmark

###  VisaMate F-1 Compliance Benchmark

- **200 curated immigration questions**
- Derived from:
  - DHS
  - USCIS Policy Manual
  - SEVP guidance
  - IRS nonresident rules
  - DSO documentation

###  Total Evaluations

- 3 Models × 200 Questions = **600 responses**

###  Categories (15 Domains)

- CPT  
- OPT  
- STEM OPT  
- Job Loss & Unemployment  
- Travel & Re-entry  
- H-1B Transition  
- SEVIS / Administrative Issues  
- Academic Issues  
- F-2 Dependents  
- Tax & Financial  
- Grace Period  
- Remote Work  
- Multiple Employment  
- Initial Entry  
- Long-term Planning  

---

## 🤖 Models Evaluated

| Model | Description |
|------|------------|
| **VisaMate** | Domain-specific, retrieval-grounded immigration assistant |
| **GPT-4o** | General-purpose LLM baseline |
| **Claude Sonnet 4.5** | General reasoning model with high fluency |

---

## 🧠 Evaluation Methodology

The evaluation uses **four complementary layers**:

### 1. Manual Meaning-Based Evaluation (Primary Ground Truth)

Each response is labeled as:

- **Correct** → Fully policy-aligned  
- **Partially Correct** → Missing key constraints  
- **Incorrect** → Misinterprets regulation  
- **Hallucination** → Fabricates or contradicts policy  

👉 Manual evaluation is required because correctness depends on **legal reasoning**, not text similarity.

---

### 2. Automated Quantitative Metrics (Secondary Signals)

- Semantic Similarity  
- Correctness Score  
- Hallucination Score  
- Completeness  
- Relevance  
- Regulatory Consistency  

⚠️ These are used for **analysis only**, not correctness decisions.

---

### 3. Qualitative Human Evaluation

Each response is rated (1–5 scale) on:

- Clarity  
- Specificity  
- Safety  
- Structure  
- Tone  
- Policy Consistency  

---

### 4. Statistical Testing

- χ² Test  
- Mann–Whitney U  
- ANOVA / Kruskal-Wallis  
- Effect Sizes  

👉 All results are statistically significant (**p < 0.001**)

---

## 📊 Results

### 🧾 Manual Evaluation

| Model | Correct | Partial | Incorrect | Hallucinations | Accuracy |
|------|--------|--------|----------|---------------|---------|
| **VisaMate** | 155 | 32 | 13 | 25 | 0.855 |
| **GPT-4o** | 165 | 27 | 8 | 20 | 0.892 |
| **Claude** | 89 | 50 | 48 | 67 | 0.570 |

---

### 📈 Quantitative Metrics (Averages)

| Metric | VisaMate | GPT-4o | Claude |
|------|----------|--------|--------|
| Correctness | 0.9168 | 0.9280 | 0.7145 |
| Hallucination Score | 0.0798 | 0.0390 | 0.2521 |
| Semantic Similarity | 0.8986 | 0.8305 | 0.7027 |
| Completeness | High | High (uneven) | Low |
| Regulatory Consistency | Highest | Strong | Weak |

---

### 🧠 Qualitative Evaluation

| Dimension | VisaMate | GPT-4o | Claude |
|----------|----------|--------|--------|
| Factual Accuracy | 4.95 | 4.30 | 4.40 |
| Completeness | 4.95 | 4.35 | 4.50 |
| Safety | 5.00 | 4.60 | 4.65 |
| Relevance | 4.75 | 4.35 | 4.30 |
| Overall | **4.65** | 4.19 | 4.33 |

---

## ⚠️ Hallucination Framework

### 🔍 Hallucination Detection Logic

A response is classified as hallucination if:

- Manually labeled as hallucination  
- Contains:
  - fabricated rules  
  - incorrect numbers  
  - contradictions  
  - unsafe overconfidence  
- Missing critical requirements with low correctness  
- Vague + low semantic alignment  

---

### 🧩 Error Taxonomy

Each incorrect response is categorized into:

1. `wrong_number`  
2. `fabricated_rule`  
3. `missing_requirement`  
4. `incorrect_process`  
5. `contradiction`  
6. `vague_generic`  
7. `unsafe_overconfident`  

---

## 🔥 Key Insights

### ✅ VisaMate

- Lowest hallucination rate  
- Highest regulatory consistency  
- Most stable across all categories  
- Errors are **low-risk (mostly omissions)**  

---

### ⚖️ GPT-4o

- Highest raw correctness  
- Strong fluency  
- Frequent **missing constraints**  
- Risk: **appears correct but incomplete**

---

### ❌ Claude

- Highest hallucination rate (~33.5%)  
- Fabricates rules and processes  
- Contradicts DHS/USCIS guidance  
- **Unsafe for immigration advising**

---

## 🧨 Why Automated Metrics Fail

- Reward verbosity over correctness  
- Cannot detect:
  - fabricated policies  
  - incorrect numbers  
  - missing requirements  
- Treat critical errors as minor differences  

👉 Underestimate hallucinations by **2–5×**

---

## 🧪 Evaluation Pipeline

```bash
1. Data Preparation
2. Model Inference (600 outputs)
3. Manual Annotation (Gold Standard)
4. Automated Metric Computation
5. Hallucination Correction
6. Statistical Analysis
7. Result Aggregation
