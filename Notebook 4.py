# Databricks notebook source
# MAGIC %pip install sentence-transformers google-generativeai scikit-learn gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 2
from sentence_transformers import SentenceTransformer
import numpy as np, json
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import gradio as gr

# Load data
pdf = spark.table("default.gramseva_embeddings").toPandas().fillna("")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_matrix = np.array([json.loads(e) for e in pdf["embedding"]])

genai.configure(api_key="AIzaSyDtpOHy0809gFeqHDhBrGLiB69UKRJRstk")
gemini = genai.GenerativeModel("gemini-1.5-flash")

def gramseva(name, occupation, state, income, category, age):
    query = f"{occupation} {state} government scheme"
    q_emb = model.encode([query])
    scores = cosine_similarity(q_emb, embeddings_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:8]
    
    context = ""
    for i in top_idx:
        r = pdf.iloc[i]
        context += f"\n- {r['scheme_name']}: Eligibility: {str(r['eligibility'])[:200]} | Benefits: {str(r['benefits'])[:200]}"
    
    prompt = f"""You are GramSeva. Find schemes for:
Name: {name}, Occupation: {occupation}, State: {state}, 
Income: ₹{income}, Category: {category}, Age: {age}

Schemes:
{context}

List eligible schemes with name, why they qualify, benefit amount, how to apply.
End with TOTAL estimated annual benefit in ₹."""

    return gemini.generate_content(prompt).text

demo = gr.Interface(
    fn=gramseva,
    inputs=[
        gr.Textbox(label="Full Name", placeholder="e.g. Ramesh Yadav"),
        gr.Dropdown(["farmer", "student", "daily wage worker", "small business owner", "unemployed", "woman entrepreneur"], label="Occupation"),
        gr.Dropdown(["Uttar Pradesh", "Maharashtra", "Karnataka", "Bihar", "Rajasthan", "Tamil Nadu", "West Bengal", "Gujarat", "Madhya Pradesh"], label="State"),
        gr.Number(label="Annual Income (₹)", value=80000),
        gr.Dropdown(["General", "OBC", "SC", "ST"], label="Category"),
        gr.Number(label="Age", value=30)
    ],
    outputs=gr.Textbox(label="🇮🇳 Your Eligible Government Schemes", lines=25),
    title="🇮🇳 GramSeva — Apni Yojana Khojein",
    description="### Find all Central & State government schemes you qualify for — in seconds.\nPowered by AI + 3,400 real government schemes.",
    examples=[
        ["Ramesh Yadav", "farmer", "Uttar Pradesh", 80000, "OBC", 42],
        ["Priya Sharma", "student", "Karnataka", 150000, "General", 20],
        ["Sunita Devi", "woman entrepreneur", "Bihar", 60000, "SC", 35]
    ]
)

demo.launch()

# COMMAND ----------

