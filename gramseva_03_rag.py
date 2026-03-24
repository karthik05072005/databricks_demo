# Databricks notebook source
# MAGIC %pip install sentence-transformers google-generativeai

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load the embeddings table
pdf = spark.table("default.gramseva_embeddings").toPandas()
pdf = pdf.fillna("")

# Load model (same one used in Notebook 2)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert stored embeddings back to numpy arrays
embeddings_matrix = np.array([json.loads(e) for e in pdf["embedding"]])

print(f"✅ Loaded {len(pdf)} schemes with embeddings")
print(f"Embedding shape: {embeddings_matrix.shape}")

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity

def find_relevant_schemes(user_query, top_k=10):
    # Embed the user's query
    query_embedding = model.encode([user_query])
    
    # Cosine similarity against all schemes
    similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        row = pdf.iloc[idx]
        results.append({
            "scheme_name": row["scheme_name"],
            "category": row["category"],
            "level": row["level"],
            "eligibility": row["eligibility"],
            "benefits": row["benefits"],
            "application": row["application"],
            "documents": row["documents"],
            "score": float(similarities[idx])
        })
    
    return results

# Quick test
test = find_relevant_schemes("farmer income support agriculture")
print(f"Top result: {test[0]['scheme_name']} (score: {test[0]['score']:.3f})")

# COMMAND ----------

def filter_by_profile(schemes, profile):
    """
    Basic keyword filter to narrow down schemes
    based on user profile before sending to LLM
    """
    filtered = []
    
    profile_text = " ".join([
        profile.get("occupation", ""),
        profile.get("state", ""),
        profile.get("category", ""),
        profile.get("gender", ""),
        str(profile.get("income", ""))
    ]).lower()
    
    for s in schemes:
        eligibility_text = (s["eligibility"] or "").lower()
        category_text = (s["category"] or "").lower()
        
        # Always include if high similarity score
        if s["score"] > 0.4:
            filtered.append(s)
            continue
            
        # Include if profile keywords match eligibility
        profile_words = [w for w in profile_text.split() if len(w) > 3]
        if any(word in eligibility_text or word in category_text 
               for word in profile_words):
            filtered.append(s)
    
    return filtered[:8]  # Max 8 schemes to LLM

# Test the filter
test_profile = {
    "occupation": "farmer",
    "state": "Maharashtra",
    "income": 50000,
    "category": "general",
    "gender": "male"
}

relevant = find_relevant_schemes("schemes for farmer in Maharashtra")
filtered = filter_by_profile(relevant, test_profile)
print(f"Filtered to {len(filtered)} schemes for LLM")
for s in filtered:
    print(f"  - {s['scheme_name']} ({s['category']})")

# COMMAND ----------

import google.generativeai as genai

genai.configure(api_key="AIzaSyDtpOHy0809gFeqHDhBrGLiB69UKRJRstk")
gemini = genai.GenerativeModel("gemini-2.5-flash-lite")

def gramseva_rag(user_profile):
    # Step 1 — Build search query
    query = f"{user_profile.get('occupation','')} {user_profile.get('state','')} government scheme benefits"
    
    # Step 2 — Retrieve top schemes
    top_schemes = find_relevant_schemes(query, top_k=15)
    
    # Step 3 — Filter
    filtered_schemes = filter_by_profile(top_schemes, user_profile)
    
    # Step 4 — Build context
    schemes_context = ""
    for i, s in enumerate(filtered_schemes, 1):
        schemes_context += f"""
Scheme {i}: {s['scheme_name']}
Category: {s['category']} | Level: {s['level']}
Eligibility: {s['eligibility'][:300]}
Benefits: {s['benefits'][:300]}
How to Apply: {s['application'][:200]}
---"""

    # Step 5 — Prompt Gemini
    prompt = f"""You are GramSeva, an AI assistant helping Indian citizens find government schemes they qualify for.

User Profile:
- Name: {user_profile.get('name', 'User')}
- Occupation: {user_profile.get('occupation')}
- State: {user_profile.get('state')}
- Annual Income: ₹{user_profile.get('income')}
- Category: {user_profile.get('category')}
- Gender: {user_profile.get('gender')}
- Age: {user_profile.get('age')}

Relevant Government Schemes from Database:
{schemes_context}

Based on this user's profile, identify which schemes they are ELIGIBLE for.
For each eligible scheme provide:
1. Scheme name
2. Why they qualify (specific to their profile)
3. Key benefit in rupees or concrete terms
4. One-line application step

Format as a clean numbered list. Be specific and encouraging.
End with total estimated annual benefit in rupees."""

    response = gemini.generate_content(prompt)
    return response.text

# Test it!
test_profile = {
    "name": "Ramesh Yadav",
    "occupation": "farmer",
    "state": "Uttar Pradesh",
    "income": 80000,
    "category": "OBC",
    "gender": "male",
    "age": 42
}

result = gramseva_rag(test_profile)
print(result)

# COMMAND ----------

