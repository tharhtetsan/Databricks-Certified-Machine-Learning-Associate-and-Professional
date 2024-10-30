# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_path = "dbfs:/FileStore/housing.csv"


# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField

schema = StructType([
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("housing_median_age", DoubleType(), True),
    StructField("total_rooms", DoubleType(), True),
    StructField("total_bedrooms", DoubleType(), True),
    StructField("population", DoubleType(), True),
    
    StructField("households", DoubleType(), True),
    StructField("median_income", DoubleType(), True),
    StructField("median_house_value", DoubleType(), True),
    StructField("ocean_proximity", StringType(), True),
])


housing_df = spark.read.format("csv").schema(schema).option("header", "true").load(df_path)
display(housing_df)

# COMMAND ----------

df = housing_df.toPandas()
print(df.shape)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing value imputation

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.fillna(df.mean(numeric_only=True), inplace=True)

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

df.select_dtypes(include=['object']).columns

# COMMAND ----------

df["ocean_proximity"] = df['ocean_proximity'].fillna(df['ocean_proximity'].mode()[0])
df["ocean_proximity"].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Outlier removal
# MAGIC 1. Z-Score Method
# MAGIC 2. Interquartile Range(IQR) Method
# MAGIC 3. Tukey's Fences Method
# MAGIC 4. Standard Deviation Method
# MAGIC 5. precentile Method

# COMMAND ----------

df.dtypes

# COMMAND ----------

numerical_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms','population','households', 'median_income', 'median_house_value']

# Calcuate the first quartile (Q1) and third quartile (Q3) for each numerical column
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)

# Calcuate the interquartile range (IQR)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1) ]
print(df_no_outliers.shape)

# Calcuate the initial row count
initial_row_count = len(df)

final_row_count = len(df_no_outliers)
removed_rows = initial_row_count - final_row_count

print("Number of removed rows : ",removed_rows)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Z-Score Method
# MAGIC

# COMMAND ----------

from scipy import stats
numerical_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms','population','households', 'median_income', 'median_house_value']

z_scores = stats.zscore(df[numerical_columns])

threshold = 3
df_no_outliers = df[(z_scores < threshold).all(axis=1)]
print(df_no_outliers.shape)

# Calcuate the initial row count
initial_row_count = len(df)

final_row_count = len(df_no_outliers)
removed_rows = initial_row_count - final_row_count

print("Number of removed rows : ",removed_rows)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Feature Creation

# COMMAND ----------

df["housing_median_age_days"] = df["housing_median_age"] * 365
df.head()

# COMMAND ----------

df = df.drop(columns="housing_median_age_days")

# COMMAND ----------

# PolynomialFeatures
"""
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
polynomial_features = poly.fit_transform(df[["feature1", "feature2"]])

# Interaction features
df["interaction_feature"] = df["feature1"] * df["feature2"]
"""

#Bining / Discretization
df['age_group'] = pd.cut(df['age'],bins=[0,18,35,50,100],labels=["child","young","middle-aged","elderly"])
df.head()





# COMMAND ----------

#Encoding categorical variables
encoded_df = pd.get_dummies(df, columns=["category"],prefix="category",drop_firest=True)
encoded_df.head()

# COMMAND ----------

# Textual feature extraction (using CountVectorizer for bag of words)
from sklearn.feature_extraction.text import CountVectorizer

text_data = ["This is the first document.","This document is the second document.",]
vectorizer = CountVectorizer()
bag_of_words = vecotizer.fit_transform(text_data)

# COMMAND ----------

# Time-based features
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
df["month"] = pd.to_datetime(df["date"]).dt.month

df.head()

# COMMAND ----------

# Domain-specific feature creation
df["income_ratio"] = df["income"] / df["expenses"]
df["aggregate_feature"] = df["feature1"]+df["feature2"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Feature Scaling
# MAGIC <b> Feature scaling is also known as data normalization : </b> The process of transoforming numerical features in a dataset to a common scale. It is a crucial step in data preprocessing and feature engineering, as it helps to bring the features to a similar range and magnitude. The goal of feature scaling is to ensure that no single feature dominates the learning process or intorduces bias due to its larger values.
# MAGIC
# MAGIC <b> There are two common methods for feature scaling : </b>
# MAGIC 1. <b> Standardization </b>
# MAGIC In this method, each feature is transfromed to have zero mean and unit variance. The formula for standardization is :
# MAGIC x_scaled = (x-mean)/standard_deviation
# MAGIC
# MAGIC
# MAGIC 2. <b> Min-Max scaling </b>
# MAGIC In this method, each feature is sacaled to specific range, typically between 0 and 1. The formula for min-max scaling is:
# MAGIC x_scaled = (X - min)/(max - min)

# COMMAND ----------

# MAGIC %md
# MAGIC <b> Feature scaling is important for several reasons : </b>
# MAGIC 1. Gradient-based optimization algorithms, such as gradient descent, converge faster when features are on a similar scale. Thie helps in achieving faster convergence and more efficient training of machine learning models.
# MAGIC
# MAGIC 2. Features with larger scales can dominate the learning process, leading to biased results. Scaling the features ensures that no singel feature has undue influence on the model.
# MAGIC
# MAGIC 3. Many machine leanring algorithms, such as K-nearest neighbors (KNN) and support vector machines (SVM), rely on calculating distances between data points. If features are not on a similar scale, features with larger values can dominate the distance calculations, leading to suboptimal results.
# MAGIC
# MAGIC 4. Some algorithms, such as principal component analysis (PCA), assume that the data is centered and on a similar scale. Feature scaling is necessary to meet these assumptions and obtain meeaningful results.

# COMMAND ----------

df.columns

# COMMAND ----------

print(numerical_columns)
print("Before")
df.head()

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. One-Hot Encoding (Feature Encoding)

# COMMAND ----------

df.select_dtypes(include=['object']).columns

# COMMAND ----------

df['ocean_proximity'].unique()

# COMMAND ----------

df['ocean_proximity'].nunique()

# COMMAND ----------

print(df.shape)
df.head()

# COMMAND ----------

df = pd.get_dummies(data=df, drop_first=True)
print(df.shape)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Feature Selection
# MAGIC Feature selection is a crucial step in feature engineering, where the goal is to ideantify and select a subset of relevant features from the availabel set of features in a dataset. 
# MAGIC
# MAGIC <b> Benefits of features selection include </b>
# MAGIC 1. Improved model performance.
# MAGIC 2. Faster model training.
# MAGIC 3. Enhanced interpretability.
# MAGIC 4. Reduced dimensionality.

# COMMAND ----------

df.head()

# COMMAND ----------

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


# COMMAND ----------

X.head()

# COMMAND ----------

y.head()

# COMMAND ----------

type(y)

# COMMAND ----------

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=5)

rfe.fit(X,y)

#selected features
selected_features = X.columns[rfe.support_]
print(selected_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7 Feature Transformation
# MAGIC - The process of applying mathematical or statistical tranformations to the existing features in a dataset to make them more suitable for a machine learning algorithm or to reveal underlying patterns in the data.
# MAGIC
# MAGIC - Feature transformation techniques aim to improve the qualiy and representativeness of the features, which can lead to better model performance and more meaningful insights.

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import boxcox

# COMMAND ----------

data = pd.DataFrame({
    'feature1': [10,20,30,40,50],
    'feature2' : [0.1,1,10,100,1000],
    'feature3' : [100,200,300,400,500]
})
print(data)

# COMMAND ----------

# Normalization
scaler = MinMaxScaler()
normalizaed_data = scaler.fit_transform(data)
print(normalizaed_data)

# COMMAND ----------

# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
print(standardized_data)

# COMMAND ----------

# Logarithmic Transformation
log_transformed_features = np.log(data)
print(log_transformed_features)

# COMMAND ----------

# Power transformation
power_tranformed_feature = np.sqrt(data)
print(power_tranformed_feature)


# COMMAND ----------

# Box-Cox Transformation
boxcox_transformed_feature = boxcox(data['feature1'])
boxcox_transformed_feature

# COMMAND ----------

# Binning
bin_edges = [0,20,40,60]
print(data['feature1'])

binned_feature = pd.cut(data['feature1'],bins=bin_edges,labels=False)
print(binned_feature)

# COMMAND ----------

# Polynomial Transformation
poly_features =pd.DataFrame(
    {
        "feature1_squared" : data['feature1'] ** 2,
        "feature1_cubed" : data['feature1'] ** 3
    }
)
print(poly_features)

# COMMAND ----------

# Interaction Terms
interaction_terms = data['feature1'] * data['feature2']
print(interaction_terms)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Dimensionality Reduction
# MAGIC 1. Feature Selection
# MAGIC 2. Feature Extraction (PCA, Lienar Discriminant Analysis)
# MAGIC
# MAGIC <b> Benefits of dimensionality reduction: </b>
# MAGIC 1. Computational Efficiency.
# MAGIC 2. Overfitting Prevention
# MAGIC 3. Improved Visaulization
# MAGIC 4. Enhanced Model Preformance

# COMMAND ----------

print(X.shape)
X.head()

# COMMAND ----------

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)

X_reduced = pca.transform(X)
print(X_reduced.shape)

# COMMAND ----------

# number of principal components
print(pca.n_components_)
print(X_reduced)

# COMMAND ----------


