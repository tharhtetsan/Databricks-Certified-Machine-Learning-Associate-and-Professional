# Databricks notebook source
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt import SparkTrials
import mlflow

# COMMAND ----------

iris = load_iris()
X = iris.data
y = iris.target

print(y)

# COMMAND ----------

def optimize(C):
    clf = SVC(C = C)

    # Use the cross validation object to compare the model's performance
    accuracy = cross_val_score(clf, X, y).mean()

    return {'loss': -accuracy, 'status': STATUS_OK}


# COMMAND ----------

search_space = hp.lognormal('C', 0 , 1.0)
algo = tpe.suggest

# COMMAND ----------

from hyperopt import SparkTrials
spark_trials = SparkTrials()

# COMMAND ----------

with mlflow.start_run(run_name='hyperopt') as run:
    argmin = fmin(
    fn= optimize,
    space=search_space,
    algo=algo,
    max_evals = 16,
    trials=spark_trials)

# COMMAND ----------

print("Best value found : ", argmin)

# COMMAND ----------


