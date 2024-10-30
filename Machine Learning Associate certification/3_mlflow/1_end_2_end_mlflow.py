# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. Data Preparation

# COMMAND ----------

import os
import pandas as pd
root_path = os.getcwd()
white_wine = pd.read_csv(root_path+"/winequality-white.csv",sep=";")
red_wine = pd.read_csv(root_path+"/winequality-red.csv",sep=";")

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0


# COMMAND ----------

data = pd.concat([red_wine,white_wine],axis=0)
print("data shape: ",data.shape)

# COMMAND ----------

data.rename(columns=lambda x: x.replace(' ','_'),inplace=True)
data.corr()

# COMMAND ----------

import seaborn as sns
sns.displot(data.quality)

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality= high_quality


# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3,4)
f,axes  = plt.subplots(dims[0],dims[1],figsize=(25,15))
axis_i, axis_j = 0,0
for col in data.columns:
    if col == 'is_red' or col == 'quality':
        continue
    sns.boxplot(x=high_quality,y=data[col],ax=axes[axis_i,axis_j])
    axis_j += 1
    if axis_j   == dims[1]:
        axis_i += 1
        axis_j = 0

# COMMAND ----------

# Check Null values
data.isna().any()

# COMMAND ----------

data = spark.createDataFrame(data)
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add to Feature Store

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id,expr, rand, col

def addIdColumn(dataframe, id_column_name):
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name]+columns]


fs_data = addIdColumn(data, "wine_id")

# COMMAND ----------

fs_data.schema

# COMMAND ----------

from databricks import feature_store
from datetime import datetime
spark.sql(f"CREATE DATABASE IF NOT EXISTS wind_db")

table_name = f"wine_data_"+str(datetime.now().strftime("%Y_%m_%d_%H_%M"))

fs = feature_store.FeatureStoreClient()
fs.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=fs_data,
    schema=fs_data.schema,
    description="wine features by ths"
)

# COMMAND ----------

from sklearn.model_selection import train_test_split

data = data.toPandas()
X = data.drop(['quality'],axis=1)
y = data.quality


X_train, X_rem , y_train, y_rem = train_test_split(X,y,train_size=0.8,random_state=123)

# Split the remaining data euqally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem,y_rem,test_size=0.5,random_state=123)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Build model
# MAGIC

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self,context, model_input):
        return self.model.predict_proba(model_input)[:,0]
    

# COMMAND ----------

with mlflow.start_run(run_name="untuned_random_forest"):
    n_estimators = 110
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
    model.fit(X_train, y_train)

    # 
    predictions_test = model.predict_proba(X_test)[:,0]

    print("predictions_test : ",predictions_test.shape)
    print("y_test : ",y_test.shape)

    # Use the area under the ROC curve
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric("auc", auc_score)
    wrappedModel = SklearnModelWrapper(model)
    
    signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
    
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__),
                             "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None
    )
    mlflow.pyfunc.log_model("random_forest_model",python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=["importance"])
feature_importances.sort_values("importance", ascending=False)

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
print("run_id : ",run_id)

# COMMAND ----------

model_name = "wine_quality"
model_uri = f"runs:/{run_id}/random_forest_model"
model_version = mlflow.register_model(model_uri, model_name)


# COMMAND ----------

model_version.version

# COMMAND ----------

# Set model to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(name=model_name,
                                       version=model_version.version, alias="Production")


# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
print(f"AUC:{roc_auc_score(y_test, model.predict(X_test))}")

# COMMAND ----------


