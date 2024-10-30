# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Explor the data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries and define schema
# MAGIC

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_path = "dbfs:/FileStore/housing.csv"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField

schema = StructType([
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("housing_median_age", DoubleType(), True),
    StructField("total_rooms", DoubleType(), True),
    StructField("total_bedrooms", DoubleType(), True),
    StructField("total_bedrooms", DoubleType(), True),
    
    StructField("households", DoubleType(), True),
    StructField("median_income", DoubleType(), True),
    StructField("median_house_value", DoubleType(), True),
    StructField("ocean_proximity", StringType(), True),
])


housing_df = spark.read.format("csv").schema(schema).option("header", "true").load(df_path)
display(housing_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dispaly the result of Dataframe

# COMMAND ----------

pd_df = housing_df.toPandas()
pd_df.head()

# COMMAND ----------

print(type(pd_df))
print(pd_df.shape)

pd_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of non-null values in each column

# COMMAND ----------

pd_df.count()

# COMMAND ----------

pd_df.count(axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve the column names and types

# COMMAND ----------

print(pd_df.columns)

# COMMAND ----------

print(type(pd_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute the pairwise correlation of columns

# COMMAND ----------

pd_df.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove duplicate rows

# COMMAND ----------

original_shape = df_1.shape

#drop duplicate rows
df_1 = df_1.drop_duplicates()

num_duplicates_rows = original_shape[0] - df_1.shape[0]
print("num_duplicates_rows : ",num_duplicates_rows)

# COMMAND ----------

data = {
    "Name" : ["John","Alice","John","Bob","Alice"],
    "Age" : [25,30,25,35,30],
    "City" : ["New York","London","New York","Paris","London"],
    "Age" : [25,30,25,35,30],
    "City" : ["New York","London","New York","Paris","London"],
    
    }

df_1 = pd.DataFrame(data)
print(df_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary statistics of numerical columns

# COMMAND ----------

pd_df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for missing values
# MAGIC

# COMMAND ----------

pd_df.isnull().values.any()

# COMMAND ----------

# check how many missing values in df
pd_df.isnull().values.sum()

# COMMAND ----------

pd_df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Visualize the Data

# COMMAND ----------

pd_df.head()

# COMMAND ----------

df_2 = pd_df.drop(columns = "median_house_value")
df_2.head()

# COMMAND ----------

correlation_values = df_2.corrwith(pd_df["median_house_value"])
sorted_correlation_values = correlation_values.sort_values(ascending = False)

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(sorted_correlation_values.index,
       sorted_correlation_values.values,
       color= ['#1f77b4' if x > 0 else '#ff7f0e' for x in sorted_correlation_values.values])
ax.set_xlabel('Features')
ax.set_ylabel('Correclation')
ax.set_title("Correlation with median hourse value")

plt.xticks(rotation = 45)
ax.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### data correlation with headmap

# COMMAND ----------

corr = pd_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr,annot=True, cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Histogram of a numberical variable

# COMMAND ----------

plt.figure(figsize=(12, 8))

plt.hist(pd_df["median_house_value"], bins=20, color="skyblue", edgecolor="black"  )

plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.title("Histogram of Median House Value")

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scatter plot of two numberical variables

# COMMAND ----------

plt.figure(figsize=(12, 8))

plt.scatter(pd_df["median_house_value"], pd_df["median_income"], color="skyblue", edgecolor="black"  )

plt.xlabel("Median House Value")
plt.ylabel("Median Income")
plt.title("Scatter of Median House Value vs Median Income")

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Pandas Profilling 

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(pd_df,
                           correlations={
                           "auto" :    {"calculate" : True},
                           "pearson" : {"calculate" : True},
                           "spearman": {"calculate" : True},
                           "kendall" :  {"calculate" : True},
                           "phi_k":    {"calculate" : True},
                           "cramers" : {"calculate" : True}
                           },  title = "Profiling Report", progress_bar = False, infer_dtypes = False)
                           
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------


