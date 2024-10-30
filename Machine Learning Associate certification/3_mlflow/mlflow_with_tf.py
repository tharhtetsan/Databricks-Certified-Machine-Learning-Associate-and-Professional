# Databricks notebook source
import pandas as pd
import os
root_path = os.getcwd()
wind_farm_data = pd.read_csv(root_path+"/windfarm_data.csv",index_col=0)
wind_farm_data.head()

# COMMAND ----------

def get_training_data(wind_farm_data):
    training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
    X = training_data.drop(columns="power")
    y = training_data["power"]    
    return X,y

def get_validation_data(wind_farm_data):
    validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
    X = validation_data.drop(columns="power")
    y = validation_data["power"]
    return X,y


def get_weather_and_forecast():
    format_date = lambda pd_date : pd_date.strftime("%Y-%m-%d")
    today = pd.Timedelta('today').normalize()
    week_ago = today - pd.Timedelta(days=5)
    week_later = today + pd.Timedelta(days=5)


    past_power_output = pd.DataFrame(wind_farm_data[format_date(week_ago):format_date(today)])
    weather_and_forecast = pd.DataFrame(wind_farm_data[format_date(week_ago):format_date(week_later)])
    if len(weather_and_forecast) < 10 :
        past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
        weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10]
    return weather_and_forecast.drops(columns="power"),past_power_output["power"]

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# COMMAND ----------

def train_keras_model(X,y):
    model = Sequential()
    model.add(Dense(100, input_shape = (X.shape[-1],), activation = "relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    
    model.fit(X,y,epochs=30,batch_size=64, validation_split=0.2,verbose=1)
    return model

# COMMAND ----------

X_test, y_test = get_validation_data(wind_farm_data)
sample_x_test = X_test.iloc[:1]
sample_x_test.shape

# COMMAND ----------

import mlflow
import mlflow.tensorflow
import mlflow.keras
from mlflow.models.signature import infer_signature,set_signature


X_train, y_train = get_training_data(wind_farm_data)

with mlflow.start_run() as run:
    mlflow.tensorflow.autolog()

    model = train_keras_model(X_train,y_train)
    run_id = mlflow.active_run().info.run_id
    signature =  infer_signature(X_test, model.predict(X_test))
    model_uri = f"runs:/{run.info.run_id}/model"
    set_signature(model_uri, signature)

    


# COMMAND ----------

model_name = "power-forecasting-model"
print(run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test model

# COMMAND ----------

loaded_model = mlflow.tensorflow.load_model(f"runs:/{run_id}/model")
loaded_model.summary()


# COMMAND ----------

import mlflow
logged_model = f'runs:/{run_id}/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
y_pred = loaded_model.predict(pd.DataFrame(X_test))

# COMMAND ----------

y_pred

# COMMAND ----------



# COMMAND ----------

import mlflow

artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
print("model_uri : ",model_uri)


model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

model_version

# COMMAND ----------

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_detials = client.get_model_version(model_name, model_version)
        status = ModelVersionStatus.from_string(model_version_detials.status)
        
        print("Model status : %s" %ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)


model_name = "power-forecasting-model"
model_version = 1
wait_until_ready(model_name, model_version)

# COMMAND ----------

# Update registered model
client = MlflowClient()
client.update_registered_model(
    name=model_name,
    description="Power forecasting model by ths",
)

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=model_version,
    description="This tf model was built using tensorflow."
)

# COMMAND ----------

## set this model version to production
client.set_registered_model_alias(name=model_name,
                                       version=model_version, alias="Production")

# COMMAND ----------


