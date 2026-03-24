# Databricks notebook source
# Create a clean GramSeva table from the uploaded data
spark.sql("""
  CREATE TABLE IF NOT EXISTS default.gramseva_schemes 
  AS SELECT * FROM default.updated_data
""")

print("gramseva_schemes table ready!")

# COMMAND ----------

# Verify your table loaded correctly
df = spark.table("default.updated_data")
print(f"Total schemes: {df.count()}")
df.printSchema()
df.show(3)