# Databricks notebook source
from pyspark.sql.types import StringType, DoubleType,StructType,StructField 

# COMMAND ----------

schema = StructType([
    StructField("date", StringType(), True),
    StructField("county", StringType(), True),
    StructField("state", StringType(), True),
    StructField("fips", DoubleType(), True),
    StructField("cases", DoubleType(), True),
    StructField("deaths", DoubleType(), True),
    
])


"""
df_path ="dbfs:/FileStore/us_counties_covid_19.csv"


covid_df = spark.read.format("csv").schema(schema).option("header","true").load(df_path)
display(covid_df)

"""
import pandas as pd
df_path = "/Workspace/Users/tharhtetsan.ai@gmail.com/DataBricks_Certification/1_AutoML_testing/us-counties-2020.csv"
covid_df = pd.read_csv(df_path)
spark_df = spark.createDataFrame(covid_df)
display(spark_df)


# COMMAND ----------

spark_df.write.saveAsTable("default.covid_df")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from default.covid_df
# MAGIC
# MAGIC

# COMMAND ----------

print(spark_df.columns)
print(type(spark_df))

# COMMAND ----------

train_df, test_df = spark_df.randomSplit([0.99, 0.01], seed=12345)



# COMMAND ----------

from databricks import  automl
import logging

logging.getLogger("py4j").setLevel(logging.WARNING)



# COMMAND ----------

summary = automl.forecast(train_df, time_col='date', horizon=30, frequency="d", primary_metric='mdape', output_database="default", target_col='deaths', timeout_minutes=15)

# COMMAND ----------

print(summary.output_table_name)

# COMMAND ----------

forcast_pd = spark.table(summary.output_table_name)
display(forcast_pd)

# COMMAND ----------

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id

model_uri = f"runs:/{trial_id}/model"
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

forcests = pyfunc_model._model_impl.python_model.predict_timeseries()
display(forcests)

# COMMAND ----------


