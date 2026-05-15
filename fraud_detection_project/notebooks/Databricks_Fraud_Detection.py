# Databricks notebook source
# MAGIC %md
# MAGIC # Deep Learning Ensemble with Data Resampling for Credit Card Fraud Detection
# MAGIC 
# MAGIC **Final Year Project** focus: Utilizing SMOTE alongside an ensemble of Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), and Long Short-Term Memory (LSTM) to correctly identify imbalanced fraud cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 1: Install Required Libraries
# MAGIC We need to install the `imbalanced-learn` library. Databricks ML runtimes generally include TensorFlow, Pandas, and Scikit-Learn natively.

# COMMAND ----------

# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 2: Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping

print(f"TensorFlow Version: {tf.__version__}")
print("All libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 3: Load the Dataset
# MAGIC **Important:** You must upload the `creditcard.csv` file into the Databricks DBFS.
# MAGIC Typically, uploaded files go to `/FileStore/tables/`.

# COMMAND ----------

try:
    # DBFS API path mapping for Pandas
    dataset_path = '/dbfs/FileStore/tables/creditcard.csv'
    df = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")
    display(df.head())
except FileNotFoundError:
    print("❌ Error: creditcard.csv not found.")
    print("Please upload the Kaggle creditcard.csv file to Databricks (Data -> Create Table -> Upload File)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 4: Data Preprocessing (Scaling & Splitting)

# COMMAND ----------

# Drop missing values if any
df.dropna(inplace=True)

# Separate Features (X) and Target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the 'Time' and 'Amount' features (PCA features V1-V28 are usually already scaled)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 5: Handling Imbalance with SMOTE
# MAGIC SMOTE is applied **ONLY** to the training data to prevent Data Leakage.

# COMMAND ----------

print("Before SMOTE:")
print(f"Normal: {sum(y_train==0)}, Fraud: {sum(y_train==1)}")

smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(f"Normal: {sum(y_train_smote==0)}, Fraud: {sum(y_train_smote==1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 6: Model Building & Training

# COMMAND ----------

input_dim = X_train_smote.shape[1]
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ----- 1. ANN MODEL -----
print("==== Training ANN ====")
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_smote, y_train_smote, epochs=10, batch_size=256, validation_split=0.2, callbacks=[early_stop], verbose=1)

# ----- 2. CNN MODEL -----
print("\n==== Training CNN ====")
# Reshape for CNN: (samples, timesteps, features)
X_train_cnn = X_train_smote.reshape(X_train_smote.shape[0], X_train_smote.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_smote, epochs=10, batch_size=256, validation_split=0.2, callbacks=[early_stop], verbose=1)

# ----- 3. LSTM MODEL -----
print("\n==== Training LSTM ====")
lstm_model = Sequential([
    LSTM(64, activation='tanh', input_shape=(input_dim, 1)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_cnn, y_train_smote, epochs=10, batch_size=256, validation_split=0.2, callbacks=[early_stop], verbose=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 7: Ensemble Averaging

# COMMAND ----------

print("Generating predictions...")
pred_ann = ann_model.predict(X_test, verbose=0)
pred_cnn = cnn_model.predict(X_test_cnn, verbose=0)
pred_lstm = lstm_model.predict(X_test_cnn, verbose=0)

# Soft Voting / Averaging
pred_ensemble = (pred_ann + pred_cnn + pred_lstm) / 3.0
pred_ensemble_binary = (pred_ensemble > 0.5).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 Step 8: Final Evaluation & Results

# COMMAND ----------

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, pred_ensemble_binary))

cm = confusion_matrix(y_test, pred_ensemble_binary)
recall = recall_score(y_test, pred_ensemble_binary)
roc_auc = roc_auc_score(y_test, pred_ensemble)

print(f"\nFOCUS METRIC - Recall: {recall:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Ensemble Model Confusion Matrix')
plt.xlabel('Predicted Label (0: Normal, 1: Fraud)')
plt.ylabel('True Label')
display(plt.show())
