# Databricks notebook source
# MAGIC %md
# MAGIC 3. Finetune parameter search

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# Initialize the FeatureStoreClient
fs = FeatureStoreClient()

# Define the table name
table_name = "wine_data_2024_09_12_07_13"

# Load the feature table into a DataFrame
feature_table_df = fs.read_table(table_name)

# Display the DataFrame
display(feature_table_df)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature


data = feature_table_df.toPandas()
X = data.drop(['quality'],axis=1)
y = data.quality


X_train, X_rem , y_train, y_rem = train_test_split(X,y,train_size=0.8,random_state=123)

# Split the remaining data euqally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem,y_rem,test_size=0.5,random_state=123)

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

# COMMAND ----------

search_space = {
    "max_depth" : scope.int(hp.quniform('max_depth',4,100,1)),
    "learning_rate" : hp.loguniform("learning_rate",-3,0),
    "reg_alpha" : hp.loguniform("reg_alpha",-5,-1),
    "min_child_weight" : hp.loguniform("min_child_weight",-1,3),
    "objective" : 'binary:logistic',
    "seed":123
}

# COMMAND ----------

def train_model(params):
    mlflow.xgboost.autolog()
    with mlflow.start_run(nested=True) as run:
        train = xgb.DMatrix(data=X_train,label=y_train)
        validation =  xgb.DMatrix(data=X_val,label=y_val)

        booster = xgb.train(params, dtrain=train, num_boost_round=1000, evals=[(validation, "validation")], early_stopping_rounds=50)
        validation_score = booster.predict(validation)

        auc_score = roc_auc_score(y_val,validation_score)
        mlflow.log_metric("auc",auc_score)

        signature = infer_signature(X_train,booster.predict(train))
        mlflow.xgboost.log_model(booster, "model", signature=signature)

    
        return {"status": STATUS_OK, "loss": -1*auc_score, "booster": booster.attributes()}

    

# COMMAND ----------

tpe

# COMMAND ----------

algo = tpe.suggest
algo

# COMMAND ----------

from hyperopt import SparkTrials
# Greater parallelism will lead to speedups, but a less optimal hpyerparameter sweep.
# A reasonable value for parallelism is the square root of max_evals.

spark_trainals = SparkTrials(parallelism=10)

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_models"):
    best_params = fmin(
        fn = train_model,
        space= search_space,
        algo= algo,
        max_evals= 12,
        trials= spark_trainals
    )

# COMMAND ----------

best_params

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.acuracy DESC']).iloc[0]
print(f"AUC of Best Run : {best_run['metrics.auc']}")

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", "wine_quality")



# COMMAND ----------

new_model_version

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

model_name = "ths_lab_databricks.default.wine_quality"

alias = "Production"

# Get the model version information using the alias
model_version_info = client.get_model_version_by_alias(name=model_name, alias=alias)

print(f"Current production model version: {model_version_info.version}")


# COMMAND ----------

## set current model version to Archived

client.set_registered_model_alias(name=model_name,
                                  version=model_version_info.version, alias="Archived")
## set this model version to production
client.set_registered_model_alias(name=model_name,
                                       version=new_model_version.version, alias="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Batch Inference

# COMMAND ----------

spark_df = spark.createDataFrame(X_train)
table_path = "dbfs:/delta/wine_data"

spark_df.write.format("delta").mode("overwrite").save(table_path)

# COMMAND ----------

import mlflow.pyfunc
model_uri = f"models:/{model_name}@production"
apply_model_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

# COMMAND ----------

# Read the "new_data" from Delta
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

from pyspark.sql.functions import struct

#Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))
new_data = new_data.withColumn("prediction", apply_model_udf(udf_inputs))
display(new_data)

# COMMAND ----------


