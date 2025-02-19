# Multi-Class Classification Model with XGBoost and Random Forest

This project demonstrates a machine learning pipeline for multi-class classification using **XGBoost** and **Random Forest** models. The pipeline includes data preprocessing, feature selection, data balancing, model training, hyperparameter tuning, and model evaluation.

## Overview

This project focuses on building a classification model for predicting multiple categories based on input features. It follows a structured approach to:

- **Data Preprocessing**: Handle missing values, scale numerical features, and encode categorical features.
- **Feature Selection**: Use XGBoost to determine important features and reduce dimensionality.
- **Data Balancing**: Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance in the training set.
- **Model Training**: Train **Random Forest** and **XGBoost** models.
- **Hyperparameter Tuning**: Use **RandomizedSearchCV** to tune hyperparameters for **XGBoost**.
- **Model Evaluation**: Evaluate models based on accuracy, confusion matrix, and classification reports.
- **LIME Interpretation**: Use **LIME** (Local Interpretable Model-agnostic Explanations) to explain model predictions.
- **Final Predictions**: Generate predictions on test data.

## 1. Requirements

The following Python packages are required to run the code:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- lime

You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lime
```
## 2. Data Preprocessing

- The data is loaded from CSV files: `train_features.csv`, `train_labels.csv`, and `test_features.csv`.
- Columns are renamed for clarity.
- Duplicate entries are removed from the training dataset.
- Categorical columns are encoded, and numerical features are scaled using standardization.

## 3. Feature Selection

- **XGBoost** is used to determine feature importance.
- The top 85% of features are selected based on their importance scores.

## 4. Balancing Data Using SMOTE

- **SMOTE** is applied to address class imbalance in the training set by generating synthetic examples.

## 5. Hyperparameter Tuning with XGBoost

- **RandomizedSearchCV** is used to tune hyperparameters of the **XGBoost** model to improve performance.

## 6. Model Training and Evaluation

- A **Random Forest** model and an **XGBoost** model are trained on the balanced dataset.
- The models are evaluated on the validation set using accuracy and confusion matrices.

## 7. LIME Model Interpretation

- **LIME** is used to provide explanations for model predictions, making the model interpretable by showing feature importance for individual predictions.

## 8. Final Predictions

- The trained **XGBoost** model is used to predict the labels for the test dataset, and the results are saved in a `test_predictions.csv` file.

## 9. Model Evaluation

### Random Forest

- **Accuracy**: `{random_forest_accuracy}`
- **Confusion Matrix**:
  ```plaintext
  {random_forest_conf_matrix}

### XGBoost

- **Accuracy**: `{xgboost_accuracy}`
- **Confusion Matrix**:
  ```plaintext
  {xgboost_conf_matrix}

## 10. Accuracy Comparison

- The accuracy of both **Random Forest** and **XGBoost** models is compared using a bar chart.

## 11. Final Test Predictions

- Predictions for the test set are saved as `test_predictions.csv`. You can download the results from the output file.

## 12. Conclusion

- This pipeline demonstrates how to apply multiple techniques to build an effective classification model for multi-class problems. The use of **SMOTE** for balancing data and **LIME** for model interpretability enhances the overall performance and transparency of the model.
