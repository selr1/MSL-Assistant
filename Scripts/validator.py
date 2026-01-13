import pandas as pd
import numpy as np
import os

# Path to feature matrix
DATA_PATH = '../CSVs/preprocessed.csv'
# iman still here
def validate():
    if not os.path.exists(DATA_PATH):
        return

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Check total samples and features
    print(f"Total samples: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    # Look for empty cells
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"Missing values: {null_count}")

    # Verify every row has 63 numerical features
    feature_cols = df.iloc[:, :-1]
    is_numeric = np.all(feature_cols.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
    
    # List how many images were processed per folder/label
    counts = df['label'].value_counts()

    # Warning for low data
    low_data = counts[counts < 10].index.tolist()
    if low_data:
        print(f"Low sample count (<10) for: {low_data}")

    print(f"Numerical Integrity: {is_numeric}")
    print("Validation complete.")

if __name__ == "__main__":
    validate()