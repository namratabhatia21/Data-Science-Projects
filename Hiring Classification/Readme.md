# Hiring Classification

In this project, the objective is to predict whether a candidate was hired or not (i.e., predict the column `embauche`) based on various features such as age, diploma, salary, availability, gender, experience, hair color, and exercise grade.

## Project Steps

### Data Splitting:
Split the data into train (70%), validation (10%), and test (20%) sets.

### Data Exploration:
Explored the data: check data types, look for missing values, incorrect values, and duplicates.

### Visualize the data.
Created a list of changes needed for each column (one-hot encoding, binning, elimination, etc.).
Checked for class balance or imbalance.

### Feature Engineering:
Applied the identified feature engineering steps (e.g., one-hot encoding).

### Model Selection and Hyperparameter Tuning:
- Tried several machine learning models (starting from basics logistic regression to bagging (random forest) and boosting (xgboost, lightgbm) algorithms).
- Performed hyperparameter tuning using the validation set to find the best performing model.
- Utilized classification reports and ROC AUC curves to ensure imbalance does not affect the metrics.
- Considered using the weight parameter in the models to address class imbalance.

