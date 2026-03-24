# databricks_demo
# 🇮🇳 GramSeva — Apni Yojana Khojein

> Find every government scheme you qualify for — in 30 seconds.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-red)
![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚨 The Problem

India has **500+ Central and thousands of State government schemes** worth lakhs of crores.  
Yet most eligible citizens **never claim them** — not because they don't deserve them,  
but because the information is buried across dozens of government portals in bureaucratic language.

A farmer in Uttar Pradesh may qualify for **12 schemes worth ₹8,00,000+/year**  
and walk away with nothing.

**GramSeva fixes this.**

---

## 💡 What It Does

A citizen enters their basic profile:
- Name, Age, Occupation
- State, Annual Income, Category (General / OBC / SC / ST)

And gets a **ranked list of every scheme they qualify for** — with:
- ✅ Why they qualify (personalised to their profile)
- 💰 Exact benefit amount in ₹
- 📋 Step-by-step application instructions
- 📊 Total estimated annual benefit

---

## 🏗️ Architecture
```
User Profile Input
       ↓
PySpark Query Builder (Databricks)
       ↓
Semantic Search over Delta Lake Embeddings
(3,400 schemes × 384-dim vectors)
       ↓
Cosine Similarity — Top-K Retrieval
       ↓
Gemini 1.5 Flash — RAG Reasoning Layer
       ↓
Ranked Eligible Schemes + Total ₹ Benefit
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Platform | Databricks (Serverless) |
| Data Storage | Delta Lake |
| Data Processing | PySpark |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | Cosine Similarity (scikit-learn) |
| LLM | Gemini 1.5 Flash API |
| Experiment Tracking | MLflow |
| UI | Gradio |
| Dataset | 3,400 real Indian government schemes |

---

## 📦 Dataset

- **Source**: [myscheme.gov.in](https://www.myscheme.gov.in) + [data.gov.in](https://data.gov.in)
- **Size**: 3,400 schemes
- **Coverage**: 19 categories — Agriculture, Health, Education, Housing, Women & Child, Business, and more
- **Includes**: Eligibility criteria, benefits, application process, required documents
- **Split**: 541 Central schemes + 2,859 State schemes

---

## 🚀 How to Run

### Prerequisites
- Databricks workspace (Free Edition works)
- Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Step 1 — Upload Dataset
Upload `updated_data.csv` to Databricks via Data Ingestion → DBFS → `/FileStore/gramseva/`

### Step 2 — Run Notebooks in Order
```bash
Notebook 1 — gramseva_01_setup.py       # Load data → Delta Lake
Notebook 2 — gramseva_02_embeddings.py  # Generate embeddings → Delta Lake
Notebook 3 — gramseva_03_rag.py         # RAG pipeline + Gemini
Notebook 4 — gramseva_04_ui.py          # Gradio UI
```

### Step 3 — Launch UI
```python
demo.launch(share=True)
# Opens public Gradio URL
```

---

## 📁 Project Structure
```
gramseva/
├── notebooks/
│   ├── gramseva_01_setup.py
│   ├── gramseva_02_embeddings.py
│   ├── gramseva_03_rag.py
│   └── gramseva_04_ui.py
├── data/
│   └── updated_data.csv
└── README.md
```

---

## 🎯 Demo Profiles

| Name | Occupation | State | Income | Result |
|---|---|---|---|---|
| Ramesh Yadav | Farmer | Uttar Pradesh | ₹80,000 | 8+ schemes, ₹5L+ benefit |
| Priya Sharma | Student | Karnataka | ₹1,50,000 | 6+ schemes, scholarships |
| Sunita Devi | Woman entrepreneur | Bihar | ₹60,000 | 10+ schemes, ₹3L+ benefit |

---

## 📈 Impact

If just **1% of eligible Indians** use GramSeva to claim one additional scheme worth ₹6,000/year:

> **₹840 crore flows back to people who need it most.**

---

## 👥 Team

Built at **Databricks Hackathon 2026**

---

## 📄 License

MIT License — free to use, modify, and distribute.
