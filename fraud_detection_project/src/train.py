import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score

# Replacing TensorFlow/Keras with Scikit-Learn Classifiers
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("🚀 Starting Credit Card Fraud Detection Pipeline (Scikit-Learn Ensemble)...")

# Make sure directories exists
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('notebooks/plots', exist_ok=True)

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
print("\n[INFO] Step 2: Data Preprocessing...")

# Try to find the dataset in multiple locations
possible_paths = [
    '../data/creditcard.csv',
    'data/creditcard.csv',
    '../../synthetic_credit_card_data.csv',
    '../synthetic_credit_card_data.csv',
    'synthetic_credit_card_data.csv'
]

df = None
for path in possible_paths:
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if df is None:
    raise FileNotFoundError("❌ Critical Error: Could not find 'creditcard.csv' or 'synthetic_credit_card_data.csv' in any expected location.")
    
# Preprocess synthetic to simulate kaggle format if testing this locally
if 'Class' not in df.columns and 'fraud_label' in df.columns:
    df.rename(columns={'fraud_label': 'Class'}, inplace=True)
    
df = df.select_dtypes(include=[np.number]) # Keep only numerical for ML

# Checking for missing values
if df.isnull().sum().max() > 0:
    df.dropna(inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Scaling
# Scaling Time and Amount (or all features as many are PCA-based but still benefit from scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training Data Shape: {X_train.shape}, Test Data Shape: {X_test.shape}")

# ==========================================
# STEP 3: HANDLING IMBALANCE (SMOTE)
# ==========================================
print("\n[INFO] Step 3: Applying SMOTE to handle Imbalance...")
# Notice: SMOTE only applied to Training data to avoid leakage.
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE - Fraud: {sum(y_train==1)}, Normal: {sum(y_train==0)}")
print(f"After SMOTE  - Fraud: {sum(y_train_smote==1)}, Normal: {sum(y_train_smote==0)}")

# ==========================================
# STEP 4 & 5: MODEL BUILDING & TRAINING
# ==========================================
print("\n[INFO] Step 4 & 5: Building and Training Models...")

# --- 1. XGBoost Classifier ---
print("--> Training XGBoost Classifier...")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_smote, y_train_smote)
joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.joblib'))

# --- 2. LightGBM Classifier ---
print("--> Training LightGBM Classifier...")
# we enforce single thread warnings if any by passing n_jobs=-1
lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
lgb_model.fit(X_train_smote, y_train_smote)
joblib.dump(lgb_model, os.path.join(MODEL_DIR, 'lgb_model.joblib'))

# --- 3. Random Forest Classifier ---
print("--> Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_smote, y_train_smote)
joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_model.joblib'))

# ==========================================
# STEP 6: ENSEMBLE LEARNING (Averaging Probabilities)
# ==========================================
print("\n[INFO] Step 6: Ensemble Learning (Averaging)...")
# Get predicted probabilities for the 'Fraud' class (index 1)
pred_xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
pred_lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
pred_rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Averaging Ensemble (Soft Voting)
pred_ensemble_prob = (pred_xgb_prob + pred_lgb_prob + pred_rf_prob) / 3.0
pred_ensemble_binary = (pred_ensemble_prob > 0.5).astype(int)

# ==========================================
# STEP 7: MODEL EVALUATION
# ==========================================
print("\n[INFO] Step 7: Final Model Evaluation...")
print("\n--- Classification Report (Ensemble) ---")
print(classification_report(y_test, pred_ensemble_binary))

cm = confusion_matrix(y_test, pred_ensemble_binary)
print("Confusion Matrix:\n", cm)

recall = recall_score(y_test, pred_ensemble_binary)
roc_auc = roc_auc_score(y_test, pred_ensemble_prob)
print(f"\nFocus -> RECALL (Fraud cases caught): {recall:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Visualization (Save Confusion Matrix)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Ensemble (XGB/LGB/RF) Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('notebooks/plots/confusion_matrix.png')

print("\n✅ Training Pipeline Completed! Models saved as .joblib files in the 'models/' folder.")
