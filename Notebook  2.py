# Databricks notebook source
df = spark.table("default.gramseva_schemes")
print(f"Total schemes: {df.count()}")
df.printSchema()
df.show(3, truncate=True)

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, lower, trim

# Combine key fields into one searchable text column for embeddings
df_clean = df.select(
    col("scheme_name"),
    col("schemeCategory").alias("category"),
    col("level"),           # Central / State
    col("details"),
    col("eligibility"),
    col("benefits"),
    col("application"),
    col("documents")
).dropna(subset=["scheme_name", "eligibility"])

# Create a combined text field for embedding
df_clean = df_clean.withColumn(
    "combined_text",
    concat_ws(" | ",
        col("scheme_name"),
        col("category"),
        col("eligibility"),
        col("benefits")
    )
)

print(f"Clean schemes: {df_clean.count()}")

# Save cleaned version to Delta
df_clean.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.gramseva_clean")

print("✅ gramseva_clean table saved!")

# COMMAND ----------

# MAGIC %pip install sentence-transformers

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# Load model (lightweight, fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pull to Pandas for embedding (3400 rows is fine)
pdf = df.toPandas()
pdf = pdf.fillna("")

print(f"Generating embeddings for {len(pdf)} schemes...")

# Generate embeddings
embeddings = model.encode(
    pdf["combined_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# Store embeddings as JSON strings (Delta-compatible)
pdf["embedding"] = [json.dumps(e.tolist()) for e in embeddings]

print("✅ Embeddings generated!")

# Convert back to Spark and save
df_with_embeddings = spark.createDataFrame(pdf)
df_with_embeddings.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.gramseva_embeddings")

print("✅ gramseva_embeddings table saved to Delta Lake!")

# COMMAND ----------

result = spark.table("default.gramseva_embeddings")
print(f"Total rows with embeddings: {result.count()}")
result.select("scheme_name", "category", "level", "embedding").show(5, truncate=50)