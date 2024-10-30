# DataBricks_Certification


https://1346229948711899.9.gcp.databricks.com/



## Cross-Validation vs Hyperparameter Optimization vs Data Augmentation
| **Concept**               | **Library Example**                           | **Purpose**                               | **How It Works**                                   | **Goal**                                          |
|---------------------------|-----------------------------------------------|-------------------------------------------|----------------------------------------------------|--------------------------------------------------|
| **Cross-Validation (CV)**  | [Spark MLlib](https://spark.apache.org/mllib/), [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) | Evaluate model performance and generalization | Split data into folds, train/test on different folds | Ensure model generalizes well, avoid overfitting  |
| **Hyperparameter Optimization (HPO)** | [Hyperopt](http://hyperopt.github.io/hyperopt/), [Optuna](https://optuna.org/), [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html) | Tune hyperparameters to improve performance | Search through hyperparameter space (Grid, Random, Bayesian) | Maximize model performance through best hyperparameters |
| **Data Augmentation**      | [Albumentations](https://albumentations.ai/), [Keras ImageDataGenerator](https://keras.io/api/preprocessing/image/), [AugLy](https://github.com/facebookresearch/AugLy) | Increase dataset size and diversity        | Apply transformations to the data (flip, rotate, add noise, etc.) | Prevent overfitting, make the model more robust  |



How They Work Together
- Data Augmentation is often applied first to artificially expand the training data and make the model more robust.
- Cross-Validation is used during training to ensure the model’s performance is evaluated correctly and generalizes well across different subsets of data.
- Hyperparameter Optimization is used to find the best hyperparameter configuration for the model, maximizing its performance.
By combining these techniques, you can build more reliable, robust, and high-performing machine learning models.


##### References
- [databricks-machine-learning-associate-certification study guide](https://medium.com/@chandadipendu/databricks-machine-learning-associate-certification-a-comprehensive-study-guide-af766b69b832)
