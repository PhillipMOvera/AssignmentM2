# diabetes_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('data/diabetes_binary_classification_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Data types and basic statistics
print(data.info())
print(data.describe())

# Define features and target variable
X = data.drop(columns=['Diabetes_binary'])  # Use the correct target column
y = data['Diabetes_binary']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize machine learning models with class weights
logistic_model = LogisticRegression(class_weight='balanced')
random_forest_model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
gradient_boosting_model = GradientBoostingClassifier()  # No built-in class weight, but Gradient Boosting models handle imbalance better
xgboost_model = XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), use_label_encoder=False, eval_metric='logloss')  # For XGBoost

# Train each model
logistic_model.fit(X_train_scaled, y_train)
random_forest_model.fit(X_train_scaled, y_train)
gradient_boosting_model.fit(X_train_scaled, y_train)
xgboost_model.fit(X_train_scaled, y_train)

# Test each model
logistic_predictions = logistic_model.predict(X_test_scaled)
rf_predictions = random_forest_model.predict(X_test_scaled)
gb_predictions = gradient_boosting_model.predict(X_test_scaled)
xgboost_predictions = xgboost_model.predict(X_test_scaled)

# Evaluate the models
print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, logistic_predictions))
print(classification_report(y_test, logistic_predictions))

print("\nRandom Forest Results:")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

print("\nGradient Boosting Results:")
print(confusion_matrix(y_test, gb_predictions))
print(classification_report(y_test, gb_predictions))

print("\nXGBoost Results:")
print(confusion_matrix(y_test, xgboost_predictions))
print(classification_report(y_test, xgboost_predictions))

# Visualize the confusion matrix for each model
def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(y_test, logistic_predictions, "Logistic Regression Confusion Matrix")
plot_confusion_matrix(y_test, rf_predictions, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, gb_predictions, "Gradient Boosting Confusion Matrix")
plot_confusion_matrix(y_test, xgboost_predictions, "XGBoost Confusion Matrix")

# Function to plot feature importances
def plot_feature_importance(importances, feature_names, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# Plot feature importance for Random Forest
plot_feature_importance(random_forest_model.feature_importances_, X.columns, "Random Forest Feature Importance")

# Plot feature importance for Gradient Boosting
plot_feature_importance(gradient_boosting_model.feature_importances_, X.columns, "Gradient Boosting Feature Importance")

# Plot feature importance for XGBoost
plot_feature_importance(xgboost_model.feature_importances_, X.columns, "XGBoost Feature Importance")

# For Logistic Regression, we use the absolute values of the coefficients as feature importance
logistic_importance = np.abs(logistic_model.coef_[0])
plot_feature_importance(logistic_importance, X.columns, "Logistic Regression Feature Importance")


        