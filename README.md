# Diabetes Prediction Using Machine Learning Models

## Overview

This project focuses on predicting whether an individual has diabetes using survey data provided by the CDC. Several machine learning models were built and evaluated to identify the most effective classifier for this prediction task. The dataset used contains various health and lifestyle-related features, and the goal was to determine which features are most predictive of diabetes, and to select the best performing model.

### Key Research Questions:

1. **Can survey questions asked from the CDC provide accurate predictions of whether an individual has diabetes?**
2. **What risk factors are most predictive of diabetes risk?**
3. **Can we use a subset of the risk factors to accurately predict whether an individual has diabetes?**
4. **Which machine learning models are best for classifying the disease?**

## Dataset

The dataset used in this project is based on CDC survey data and includes 22 features. Key variables include:
- `Diabetes_binary`: Target variable (0 = No Diabetes, 1 = Diabetes)
- Features such as `BMI`, `HighBP`, `HighChol`, `Smoker`, `PhysActivity`, etc.

### Features and Target:
- **Features**: Various health metrics (e.g., BMI, HighBP, Smoker) and lifestyle variables (e.g., physical activity, alcohol consumption).
- **Target**: Binary classification (0 = No diabetes, 1 = Diabetes).

## Models

The following machine learning models were implemented:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**
4. **XGBoost**

### Class Imbalance Handling

The dataset is imbalanced (fewer cases of diabetes compared to non-diabetes). To account for this, class weights and balancing techniques were used:
- **Logistic Regression** and **Random Forest**: `class_weight='balanced'`.
- **XGBoost**: `scale_pos_weight` was adjusted to address class imbalance.

## Performance Evaluation

The models were evaluated using the following metrics:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: How many of the predicted positives were actually positive.
- **Recall**: How many actual positives were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.

### Results Summary:

- **Logistic Regression**:
  - Accuracy: 73%
  - High recall for detecting diabetes but lower precision.
- **Random Forest**:
  - Accuracy: 86%
  - Balanced performance, with slightly more false negatives than other models.
- **Gradient Boosting**:
  - Accuracy: 87%
  - A balanced model across precision and recall.
- **XGBoost**:
  - Accuracy: 71%
  - Higher recall but more false positives compared to other models.

## Feature Importance

Feature importance analysis was performed for all models:

- **Logistic Regression**: Feature importance based on the absolute values of model coefficients.
- **Random Forest, Gradient Boosting, XGBoost**: Feature importance was extracted directly from the models.

Key features across models include:
- `BMI`
- `HighBP`
- `PhysHlth`
- `Age`

## Conclusion

This project demonstrated that survey data from the CDC can be used to 
predict diabetes with reasonable accuracy using machine learning models. 
Gradient Boosting performed best overall, balancing both precision and 
recall. XGBoost was particularly useful for minimizing false negatives,
critical for detecting diabetes cases. Feature importance analysis 
showed that `BMI` and `HighBP` were consistently strong predictors of
diabetes.

## How to Run the Project

### 1. Clone the Repository:

```bash
cd <repository-directory>
git clone <repository-url>
```

### 2. Navigate to the project directory:
```bash
cd diabetes_analysis
```
### 3. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 4. Install the required packages:

```bash
pip install -r requirements.txt
```

### 5. Run the analysis script
```bash
python diabetes_analysis.p
```

### 6. The script will:

   - Load the dataset
   - Preprocess the data (including feature scaling)
   - Split the data into training and testing sets
   - Train models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
   - Evaluate each model's performance using accuracy, confusion matrix, and classification report
   - Visualize confusion matrices and feature importance



