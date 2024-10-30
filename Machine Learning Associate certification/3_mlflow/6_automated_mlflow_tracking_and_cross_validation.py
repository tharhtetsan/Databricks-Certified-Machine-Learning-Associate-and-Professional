# Databricks notebook source
# Load data

train_path = "/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt"
full_training_data =  spark.read.format("libsvm").option("numFeatures", "784").load(train_path)

test_path = "/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt"
test_data = spark.read.format("libsvm").option("numFeatures", "784").load(test_path)

full_training_data.cache()
test_data.cache()

print(f"There are {full_training_data.count()} training images and {test_data.count()} test images.")

# COMMAND ----------

training_data, validation_data = full_training_data.randomSplit([0.8, 0.2], seed=12345)
display(training_data)

# COMMAND ----------


from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
import mlflow
import mlflow.spark

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define the ML pipeline

# COMMAND ----------

# StringIndexer : Convert the input column "label" (digits) to categorical values
indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
dtc = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features",maxBins=8, maxDepth = 4)
pipeline = Pipeline(stages=[indexer, dtc])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Train the model and make predictions

# COMMAND ----------

model = pipeline.fit(training_data)

predictions = model.transform(validation_data)

# COMMAND ----------

# Define an eveluation metric and evaluate the model on the validation dataset.
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", metricName="f1")
validation_metric = evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the metric and parameters

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

with mlflow.start_run(nested=True) as run:
    mlflow.spark.log_model(spark_model=model, artifact_path="best-model")
    mlflow.log_metric(evaluator.getMetricName(), validation_metric)
        
    # log all the parameters used in the model
    params = dtc.extractParamMap()
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
        print(f"{param_name.name} : {param_value}")

print(" Test Weighted Precision : ",validation_metric)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model with cross validation

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# COMMAND ----------

# Define the parameter grid to examine
grid = ParamGridBuilder()\
.addGrid(dtc.maxDepth, [2,3,4,5,6,7,8,9,10])\
.addGrid(dtc.maxBins, [2,4,8]).build()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a cross validator

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3)

# COMMAND ----------

with mlflow.start_run(nested=True) as run:
    # train with cross-validation on training set and it will retrun the best model
    cvModel = cv.fit(training_data)

    # Best model params
    bestParams = cvModel.bestModel.stages[-1].extractParamMap()

    test_metric = evaluator.evaluate(cvModel.transform(test_data))
    
    mlflow.log_metric("test_"+evaluator.getMetricName(), test_metric)

    mlflow.spark.log_model(spark_model = cvModel.bestModel,artifact_path="best-model")

    # log all the parameters used in the model
    params = dtc.extractParamMap()
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
        print(f"{param_name.name} : {param_value}")

# COMMAND ----------

print("Test Weighted Precision : ", test_metric)

# COMMAND ----------


