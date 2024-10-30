# Databricks notebook source
from databricks import feature_store
from databricks.feature_store import feature_table,FeatureLookup

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id,expr, rand
import uuid
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# COMMAND ----------

df_path = os.getcwd()+"/winequality-red.csv"
pd_df = pd.read_csv(df_path)

raw_data = spark.createDataFrame(pd_df)
display(raw_data)


# COMMAND ----------

def addIdColumn(dataframe, id_column_name):
    columns = dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name]+columns]


def renameColumns(df):
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(" ", "_"))
    return renamed_df

# COMMAND ----------

renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, "wine_id")


# COMMAND ----------

feature_df = df.drop('quality')
display(feature_df)

# COMMAND ----------

from datetime import datetime
spark.sql(f"CREATE DATABASE IF NOT EXISTS wind_db")

table_name = f"wine_db_"+str(datetime.now().strftime("%Y_%m_%d_%H_%M"))
print(table_name)

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
fs.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=feature_df,
    schema=feature_df.schema,
    description="wine features by ths"
)

# COMMAND ----------

inference_data_df = df.select("wine_id","quality",(10*rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you don't provide the `feature_names` parameter, all features expect primary keys are returned.

    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
    
    # fs.create_training_set looks up features in model_feature_lookups that match the priamry key from inference_date_df

    training_set = fs.create_training_set(inference_data_df,model_feature_lookups,label="quality",
                                          exclude_columns="wine_id")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datesets
    X = training_pd.drop("quality",axis=1)
    y = training_pd["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, training_set

# COMMAND ----------

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

# delete wine model if exist

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

try:
    client.delete_registered_model("wine_model")
except:
    None


# COMMAND ----------

import mlflow
# Disable MLflow autologging and instead log the model using Feature Store
mlflow.sklearn.autolog(log_models = False)

def train_model(X_train,X_test, y_train, y_test, training_set,fs):



    with mlflow.start_run() as run:
        rf = RandomForestRegressor(max_depth=3,n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_absolute_error(y_test,y_pred))
        mlflow.log_metric("test_r2_score",r2_score(y_test,y_pred))

        fs.log_model(model=rf,
                     artifact_path="wine_quality_prediction",
                     flavor=mlflow.sklearn,
                     training_set= training_set,
                     registered_model_name= "wine_model")
        
train_model(X_train,X_test, y_train, y_test,training_set,fs)

# COMMAND ----------

batch_input_df = inference_data_df.drop("quality")
print(batch_input_df)

model = client.get_registered_model("wine_model")
aliases = model.latest_versions
print(aliases)



# COMMAND ----------

# Must be run on a cluster running Databricks Runtime for Machine Learning.
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
model_name = "wine_model"
model_stage = "1"
model_uri = f"models:/{model_name}/{model_stage}"
# This model was packaged by Feature Store.
# To retrieve features prior to scoring, call FeatureStoreClient.score_batch.




predictions_df = fs.score_batch(model_uri, batch_input_df)
display(predictions_df["wine_id","prediction"])

# COMMAND ----------

so2_cols = ["free_sulfur_dioxide", "total_sulfur_dioxide"]
new_features_df = (feature_df.withColumn("average_so2_2", expr("+".join(so2_cols)) / 2))

# COMMAND ----------

#display(new_features_df)
fs.write_table(
    name=table_name, 
    df=new_features_df,
    mode="merge",

)

# COMMAND ----------

display(fs.read_table(table_name))

# COMMAND ----------

fs = FeatureStoreClient()
model_name = "wine_model"
model_stage = "1"
model_uri = f"models:/{model_name}/{model_stage}"
# This model was packaged by Feature Store.
# To retrieve features prior to scoring, call FeatureStoreClient.score_batch.




predictions_df = fs.score_batch(model_uri, new_features_df)
display(predictions_df["wine_id","prediction"])

# COMMAND ----------


