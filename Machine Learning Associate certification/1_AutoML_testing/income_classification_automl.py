# Databricks notebook source
from pyspark.sql.types import DoubleType, StringType, StructType, StructField


schema = StructType([
    StructField("age", DoubleType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", DoubleType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", DoubleType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", DoubleType(), True),
    StructField("capital_loss", DoubleType(), True),
    StructField("hours_per_week", DoubleType(), True),
    StructField("native_country", StringType(), True),
    StructField("income", StringType(), True),
])

df_path = "dbfs:/FileStore/Adult_Census_Income.csv"
census_df = spark.read.format("csv").schema(schema).load(df_path)
census_df.head()
display(census_df)

# COMMAND ----------

census_df.write.saveAsTable("default.census_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT* FROM census_t
# MAGIC

# COMMAND ----------

print(census_df.count())
census_df.display()

# COMMAND ----------

train_df, test_df = census_df.randomSplit([0.99,0.01],seed = 42)
display(train_df)

# COMMAND ----------

from databricks import automl
summary = automl.classify(train_df,target_col='income',timeout_minutes=5)

# COMMAND ----------

print(summary)

# COMMAND ----------

import mlflow


model_uri = summary.best_trial.model_path

y_test = test_df["income"].tolist()
x_test = test_df.drop(["income"], axis=1)

model = mlflow.pyfunc.load_model(model_uri)
y_pred = model.predict(x_test)
test_df["income_predicted"] = y_pred
display(test_df)

# COMMAND ----------

type(test_df)

# COMMAND ----------

test_df["income"]

# COMMAND ----------

import sklearn.metrics as metrics
model = mlflow.sklearn.load_model(model_uri)
print(metrics.classification_report(test_df["income"],test_df["income_predicted"]))

# COMMAND ----------


