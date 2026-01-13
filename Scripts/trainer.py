import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# File paths
DATA_PATH = '../CSVs/preprocessed.csv'
RF_MODEL_PATH = './rf_model.pkl'
SVM_MODEL_PATH = './svm_model.pkl'
SCALER_PATH = './scaler.pkl'

def train_and_compare():
    print("Loading preprocessed csv from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Feature and label separation
    X = df.drop('label', axis=1)
    y = df['label']

    # 80/20 train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale landmark coordinates
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Random Forest Implementation
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)

    # 2. SVM Implementation
    print("Training SVM...")
    svm = SVC(C=1.5, kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)

    # Results Comparison
    print("\n" + "RESULTS")
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"SVM Accuracy:           {svm_acc*100:.2f}%")

    # Save components
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(svm, SVM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

if __name__ == "__main__":
    train_and_compare()
