# HeartAttackPrediction-ML

[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.07%25-brightgreen)](https://github.com/yourusername/HeartAttackPrediction-ML)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-blue)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)](https://xgboost.readthedocs.io)

**99% Accurate Heart Attack Risk Predictor using Scikit-learn & XGBoost**

High-performance ML classifier achieving **99.07% test accuracy** (Random Forest) on clinical biomarkers (BP, Troponin, CK-MB, etc.). Complete pipeline: EDA, outlier removal, hyperparameter tuning (RandomizedSearchCV), benchmarking. Heidelberg University Systems Biology project.

## 🎯 What You'll Find
- **Dataset**: 1190 patient records (Age, Gender, Heart Rate, Systolic/Diastolic BP, Blood Sugar, CK-MB, Troponin)
- **Preprocessing**: Outlier removal (IQR), feature engineering (abnormal flags)
- **Models Benchmarked**:
  | Model          | Test Accuracy | Precision | ROC-AUC |
  |----------------|---------------|-----------|---------|
  | **Random Forest** | **99.07%**   | **99.49%**| **99.09%** |
  | **XGBoost**    | 98.75%       | 98.99%   | 98.68%  |
  | Gradient Boost | 98.75%       | 98.99%   | 98.68%  |
  | AdaBoost       | 98.44%       | 98.50%   | 98.27%  |
  | Decision Tree  | 98.44%       | 98.50%   | 98.43%  |

- **Deployable Model**: `rf_heart_model.pkl` (load & predict instantly)

## 🛠️ Key Libraries Explained
- **Scikit-learn** (`sklearn`): Free Python ML library for classification/regression/clustering. Features 100+ algorithms (Random Forest, SVM, etc.), preprocessing (scaling), metrics, and tools like GridSearchCV. Consistent API: `model.fit(X, y)` → `model.predict(new_data)`. Built on NumPy/SciPy—gold standard for classical ML

  ## 📋 Pipeline Summary

**Complete workflow from your notebook** (copy-paste to understand the full analysis):

```python
# ===== 1. IMPORTS & LIBRARIES =====
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # XGBoost used here
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ===== 2. LOAD DATA & EDA =====
df = pd.read_csv('data.csv')  # 1190 rows: Age, BP, HR, Sugar, CK-MB, Troponin
sns.boxplot(data=df[['Heart_rate', 'Systolic_bp', 'Blood_sugar', 'CK-MB']])  # Outlier detection
df = df[(df['Heart_rate'] <= 200) & (df['Blood_sugar'] <= 600)]  # Remove outliers
df['flag_abnormal_CKMB'] = (df['CK-MB'] > 50).astype(int)
df['Troponin_flag'] = (df['Troponin'] > 0.05).astype(int)

# ===== 3. TRAIN/TEST SPLIT =====
X = df.drop('Result', axis=1); y = (df['Result'] == 'positive').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

# ===== 4. BENCHMARK 5 MODELS (XGBoost included) =====
models = {'DT': DecisionTreeClassifier(), 'RF': RandomForestClassifier(), 
          'GB': GradientBoostingClassifier(), 'AB': AdaBoostClassifier(), 
          'XGB': XGBClassifier()}  # XGBoost benchmarked

params = {'RF': {'n_estimators': , 'max_depth': [10,20,None]}}  # Tuned via RandomizedSearchCV
best_rf = RandomizedSearchCV(RandomForestClassifier(), params['RF'], cv=3, n_iter=50).fit(X_train, y_train).best_estimator_

# ===== 5. RESULTS (99.07% TOP SCORE) =====
y_pred = best_rf.predict(X_test)
print(f"✅ RF Accuracy: {accuracy_score(y_test, y_pred):.2%}")  # 99.07%
print(f"Precision: {precision_score(y_test, y_pred):.2%}, Recall: {recall_score(y_test, y_pred):.2%}")

# ===== 6. SAVE MODEL =====
import joblib; joblib.dump(best_rf, 'rf_heart_model.pkl'); joblib.dump(scaler, 'scaler.pkl')
print("✅ Model saved! Ready for predictions.")

- **XGBoost** (eXtreme Gradient Boosting): Optimized gradient-boosted decision trees library. Builds trees sequentially (each correcting prior errors) with parallelism, L1/L2 regularization (anti-overfitting), missing-value handling, and GPU support. Faster/more accurate than scikit-learn's GradientBoosting.
## 🚀 Quick Start
```bash
pip install scikit-learn xgboost pandas seaborn matplotlib joblib
jupyter notebook heart_attack_classification_Project.ipynb
