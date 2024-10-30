# Databricks notebook source
# MAGIC %md
# MAGIC ## Tuning Distributed Training Algorithm with Hyperopt and Apache Spark MLlib

# COMMAND ----------

# Load data
#import spark


train_path = "/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt"
full_training_data =  spark.read.format("libsvm").load(train_path)

test_path = "/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt"
test_data = spark.read.format("libsvm").load(test_path)

full_training_data.cache()
test_data.cache()

print(f"There are {full_training_data.count()} training images and {test_data.count()} test images.")

# COMMAND ----------

training_data, validation_data = full_training_data.randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

display(training_data)

# COMMAND ----------

type(validation_data)
first_row = validation_data.first()

print(type(first_row))

first_row_features = first_row["features"].toArray()

print((first_row_features))


# COMMAND ----------

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

try:
    import mlflow.pyspark.ml
    mlflow.pyspark.ml.autolog()
except:
    print("MLflow client version need or use Databricks Runtime for ML 8.3 above.")

# COMMAND ----------

def train_tree(minInstancesPerNode, MaxBins):
    # Use MLflow to track the model
    # Specify "nested=True" since this single model will be logged as a child run of Hyperopt's run.
    with mlflow.start_run(nested=True) as run:
        
        indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
        
        # DecisionTreClassifier : learn to predict column "indexedLabel" using the "feature" column.
        #minInstancesPerNode_doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. 
        dtc = DecisionTreeClassifier(labelCol="indexedLabel", minInstancesPerNode=minInstancesPerNode, maxBins=MaxBins)
        

        # Chain indexer and dtc together into a single ML pipeline
        pipeline = Pipeline(stages=[indexer, dtc])
        model = pipeline.fit(training_data)
        
        # Define an eveluation metric and evaluate the model on the validation dataset.
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")
        predictions = model.transform(validation_data)
        validation_metric = evaluator.evaluate(predictions)
        mlflow.log_metric("val_f1_score", validation_metric)

    return model, validation_metric

# COMMAND ----------

initial_model, val_metric = train_tree(minInstancesPerNode = 200, MaxBins =2)
print(f" The trained decision tree achieved a F1 score of {val_metric} on the validation dataset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Hyperopt to tune the Parameter

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def train_with_hyperopt(params):
    minInstancesPerNode = int(params["minInstancesPerNode"])
    maxBins = int(params["maxBins"])
    
    model, f1_score = train_tree(minInstancesPerNode, maxBins)

    # loss take the negative of the f1_score
    loss =  - f1_score
    return {"loss": loss, "status": STATUS_OK, "model": model, "val_f1_score": f1_score}

# COMMAND ----------

import numpy as np
# Search Space
space = {
    "minInstancesPerNode" : hp.uniform("minInstancesPerNode",10,200),
    "maxBins" : hp.uniform("maxBins", 2, 32)
}

# COMMAND ----------

algo = tpe.suggest

# COMMAND ----------

with mlflow.start_run():
    best_params = fmin(
        fn=train_with_hyperopt,
        space=space,
        algo=algo, 
        max_evals = 8)
    

# COMMAND ----------

print(best_params)

# COMMAND ----------

# Retrain the model on training dataset
best_minInstancesPerNode = int(best_params['minInstancesPerNode'])
best_maxBins = int(best_params['maxBins'])

print("best_minInstancesPerNode : ",best_minInstancesPerNode)
print("best_maxBins : ",best_maxBins)


final_model, val_f1_score = train_tree(best_minInstancesPerNode, best_maxBins)
print(f" The trained decision tree achieved a F1 score of {val_f1_score} on the validation dataset.")

# COMMAND ----------

evaluator  = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")

initial_model_test_metric = evaluator.evaluate(initial_model.transform(validation_data))
final_model_test_metric = evaluator.evaluate(final_model.transform(validation_data))

print("initial_model_test_metric : ",initial_model_test_metric)
print("final_model_test_metric : ",final_model_test_metric)

# COMMAND ----------


