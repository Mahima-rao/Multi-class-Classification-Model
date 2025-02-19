import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
import lime
import lime.lime_tabular

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")

# --------------------------------------------------------------------
# 1. Load Data and Rename Columns
# --------------------------------------------------------------------
col_names = [f"Feature_{i}" for i in range(903)]  # Rename columns for clarity
train_features = pd.read_csv("train_features.csv", header=None, names=col_names)
train_labels = pd.read_csv("train_labels.csv", header=None)
test_features = pd.read_csv("test_features.csv", header=None, names=col_names)

# Remove duplicates from train_features and train_labels
train_features = train_features.drop_duplicates()
train_labels = train_labels.loc[train_features.index]

# Ensure test_features has the same columns as train_features
test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

# --------------------------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------------------------
cat_cols = train_features.select_dtypes(include=['object']).columns.tolist()
num_cols = train_features.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

# --------------------------------------------------------------------
# 3. Split into Training and Validation Sets
# --------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42
)

# Preprocess
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

print(f"Shape of X_train after preprocessing: {X_train_preprocessed.shape}")

# Get feature names from the preprocessor for interpretability
all_feature_names = preprocessor.get_feature_names_out()

# --------------------------------------------------------------------
# 4. Feature Selection Using XGBoost Importances
# --------------------------------------------------------------------
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X_train_preprocessed, y_train.values.ravel())

# Create a DataFrame of feature importances
feature_importance_df = pd.DataFrame({
    'Feature': range(len(all_feature_names)),
    'Importance': xgb_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Select top 85% of features
top_n = int(len(feature_importance_df) * 0.85)
selected_feature_indices = feature_importance_df.iloc[:top_n]['Feature'].values

# Apply feature selection
X_train_selected = X_train_preprocessed[:, selected_feature_indices]
X_val_selected = X_val_preprocessed[:, selected_feature_indices]

# Map indices back to the actual feature names
selected_feature_names = all_feature_names[selected_feature_indices]

print(f"Shape of X_train_selected: {X_train_selected.shape}")

# --------------------------------------------------------------------
# 5. Balance Data Using SMOTE
# --------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train.values.ravel())

print(f"Shape of X_train_balanced: {X_train_balanced.shape}")

# --------------------------------------------------------------------
# 6. Hyperparameter Tuning on XGBoost
# --------------------------------------------------------------------
param_dist = {
    'n_estimators': [100, 200, 500],
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'gamma': np.linspace(0, 0.3, 10),
    'subsample': np.linspace(0.7, 1.0, 5),
    'colsample_bytree': np.linspace(0.7, 1.0, 5)
}

xgb_tuner = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42
)
xgb_tuner.fit(X_train_balanced, y_train_balanced)
best_xgb = xgb_tuner.best_estimator_

# --------------------------------------------------------------------
# 7. Train RandomForest on Balanced Data
# --------------------------------------------------------------------

# Initialize RandomForest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cv_scores = cross_val_score(rf_clf, X_train_balanced, y_train_balanced, cv=3, scoring='accuracy')
rf_clf.fit(X_train_balanced, y_train_balanced)

# --------------------------------------------------------------------
# 8. Model Evaluation (Confusion Matrices)
# --------------------------------------------------------------------
rf_predictions = rf_clf.predict(X_val_selected)
xgb_predictions = best_xgb.predict(X_val_selected)

rf_accuracy = accuracy_score(y_val, rf_predictions)
xgb_accuracy = accuracy_score(y_val, xgb_predictions)

# # Compute Precision, Recall, F1-score
# rf_classification_report = classification_report(y_val, rf_predictions)
# xgb_classification_report = classification_report(y_val, xgb_predictions)

rf_conf_matrix = confusion_matrix(y_val, rf_predictions)
xgb_conf_matrix = confusion_matrix(y_val, xgb_predictions)

print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
print("RandomForest Confusion Matrix:")
print(rf_conf_matrix)

print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)

# # Print Classification Reports
# print("\nRandomForest Classification Report:\n", rf_classification_report)
# print("\nXGBoost Classification Report:\n", xgb_classification_report)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("RandomForest Confusion Matrix")
sns.heatmap(xgb_conf_matrix, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# 10. Accuracy Comparison Chart
# --------------------------------------------------------------------
plt.figure(figsize=(6, 4))
models = ['RandomForest', 'XGBoost']
accuracies = [rf_accuracy, xgb_accuracy]

sns.barplot(x=models, y=accuracies, palette=['blue', 'green'])
plt.ylim(0, 1)  # Accuracy range from 0 to 1
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.text(0, rf_accuracy + 0.02, f"{rf_accuracy:.4f}", ha='center', fontsize=12)
plt.text(1, xgb_accuracy + 0.02, f"{xgb_accuracy:.4f}", ha='center', fontsize=12)
plt.show()

# --------------------------------------------------------------------
# 11. LIME Analysis on XGBoost Model
# --------------------------------------------------------------------
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_balanced,
    feature_names=selected_feature_names,
    class_names=[str(c) for c in np.unique(y_train_balanced)],
    mode='classification'
)

# Explain the first instance in the validation set
i = 0
exp = lime_explainer.explain_instance(
    data_row=X_val_selected[i],
    predict_fn=best_xgb.predict_proba,
    num_features=10
)
# Display the explanation
exp.show_in_notebook(show_table=True, show_all=False)
print("\nLIME Explanation for validation instance 0:")
print(exp.as_list())

# --------------------------------------------------------------------
# 12. Final Predictions on Test Data
# --------------------------------------------------------------------
# Preprocess the test set, then select the same top features
test_preprocessed = preprocessor.transform(test_features)
test_selected = test_preprocessed[:, selected_feature_indices]

# Predict using the best XGBoost model
final_test_predictions = best_xgb.predict(test_selected)

# Save predictions to CSV (no headers, one prediction per line)
pd.DataFrame(final_test_predictions).to_csv('test_predictions.csv', index=False, header=False)
print(f"\nFinal predictions saved to test_predictions.csv ({len(final_test_predictions)} rows).")
