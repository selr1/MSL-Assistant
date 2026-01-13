import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = '../CSVs/preprocessed.csv'
MODEL_PATH = './svm_modelc1.5.pkl'
SCALER_PATH = './scaler.pkl'

def generate_visuals():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Scale the test set
    X_test_scaled = scaler.transform(X_test)

    # Prediction
    y_pred = model.predict(X_test_scaled)

    # 1. Classification Report
    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    # 2. Confusion Matrix Visual
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(22, 16)) # Large canvas for MSL labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    
    plt.title('MSL Recognition Performance - Confusion Matrix (SVM C=1.5)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n" + "RESULTS")
    print("\nsaved: confusion_matrix.png and classification_report.txt")

if __name__ == "__main__":
    generate_visuals()
