import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

# 1. Import Dataset 
print(" Tahap 1: Import Dataset ")
df = pd.read_csv("diabetes.csv")
print("Jumlah baris dan kolom:", df.shape)
print("\n5 baris pertama:")
print(df.head())
print("\nJumlah nilai nol pada tiap kolom:")
print((df == 0).sum())

# 2. Data Cleaning 
print("\nTahap 2: Data Cleaning")
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)
print("Nilai nol setelah cleaning:")
print((df == 0).sum())

print("\nStatistik deskriptif setelah cleaning:")
print(df.describe())

# 3. Deteksi Outlier dengan IQR
print("\nTahap 3: Deteksi Outlier (IQR) ")
for col in df.columns[:-1]:  
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outlier terdeteksi")

# 4. Pisahkan Fitur dan Target 
print("\nTahap 4: Pemisahan Fitur dan Target")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print("Jumlah fitur:", X.shape[1])
print("Jumlah target (Outcome=1):", y.sum(), "| (Outcome=0):", len(y)-y.sum())

# 5. Standardisasi (Z-Score)
print("\nTahap 5: Standardisasi (Z-Score)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Rata-rata fitur setelah standarisasi (mendekati 0):", np.round(X_scaled.mean(), 3))
print("Standar deviasi fitur setelah standarisasi (mendekati 1):", np.round(X_scaled.std(), 3))

# 6. Split Data: 60% Train, 20% Validation, 20% Test
print("\n=== Tahap 6: Pembagian Data ===")
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"Training set: {len(X_train)} data")
print(f"Validation set: {len(X_val)} data")
print(f"Testing set: {len(X_test)} data")

# 7. Model 1: Random Forest
print("\nTraining Model - Random Forest")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
print("Model Random Forest selesai dilatih.")

# Evaluasi pada validation set
y_val_pred_rf = rf.predict(X_val)
print("\n[Evaluasi Random Forest - Validation Set]")
print("Accuracy:", round(accuracy_score(y_val, y_val_pred_rf), 4))
print("Precision:", round(precision_score(y_val, y_val_pred_rf), 4))
print("Recall:", round(recall_score(y_val, y_val_pred_rf), 4))
print("F1-Score:", round(f1_score(y_val, y_val_pred_rf), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, y_val_pred_rf), 4))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_rf))

# 7. Model 2: Logistic Regression
print("\nTraining Model - Logistic Regression")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
print("Model Logistic Regression selesai dilatih.")

# Evaluasi pada validation set
y_val_pred_log = logreg.predict(X_val)
print("\n[Evaluasi Logistic Regression - Validation Set]")
print("Accuracy:", round(accuracy_score(y_val, y_val_pred_log), 4))
print("Precision:", round(precision_score(y_val, y_val_pred_log), 4))
print("Recall:", round(recall_score(y_val, y_val_pred_log), 4))
print("F1-Score:", round(f1_score(y_val, y_val_pred_log), 4))
print("ROC-AUC:", round(roc_auc_score(y_val, y_val_pred_log), 4))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_log))

# 8. Evaluasi Akhir di Test Set (Random Forest & Logistic Regression)
print("\nTahap 8: Evaluasi Akhir (Test Set)")

# Random Forest
y_test_pred_rf = rf.predict(X_test)
print("\n[Random Forest - Test Set]")
print("Accuracy:", round(accuracy_score(y_test, y_test_pred_rf), 4))
print("Precision:", round(precision_score(y_test, y_test_pred_rf), 4))
print("Recall:", round(recall_score(y_test, y_test_pred_rf), 4))
print("F1-Score:", round(f1_score(y_test, y_test_pred_rf), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_pred_rf), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_rf))

# Logistic Regression
y_test_pred_log = logreg.predict(X_test)
print("\n[Logistic Regression - Test Set]")
print("Accuracy:", round(accuracy_score(y_test, y_test_pred_log), 4))
print("Precision:", round(precision_score(y_test, y_test_pred_log), 4))
print("Recall:", round(recall_score(y_test, y_test_pred_log), 4))
print("F1-Score:", round(f1_score(y_test, y_test_pred_log), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_pred_log), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_log))
