"""
Step 1.1: Data Preprocessing

Handle missing values, remove ECG columns, scale features, 
and create patient-level train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_dataset(csv_path='data/heart_iot_synthetic.csv'):
    """Load the synthetic dataset."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def remove_ecg_columns(df):
    """Remove ECG-related columns since we don't have ECG sensors."""
    print("\nRemoving ECG columns (not available in IoT setup)...")
    
    ecg_columns = [
        'ecg_hr_bpm', 
        'ecg_rr_interval_ms', 
        'ecg_rmssd_ms', 
        'ecg_sdnn_ms', 
        'ecg_signal_quality'
    ]
    
    existing_ecg_cols = [col for col in ecg_columns if col in df.columns]
    
    if existing_ecg_cols:
        df = df.drop(columns=existing_ecg_cols)
        print(f"✓ Removed {len(existing_ecg_cols)} ECG columns: {existing_ecg_cols}")
    else:
        print("  No ECG columns found")
    
    return df


def handle_missing_values(df):
    """Handle missing values using median imputation."""
    print("\nHandling missing values...")
    
    missing_before = df.isnull().sum().sum()
    print(f"  Missing values before: {missing_before}")
    
    # Get columns with missing values
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    if cols_with_missing:
        print(f"  Columns with missing values: {cols_with_missing}")
        
        # Use median imputation for ALL columns with missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64']:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"    - {col}: filled {df[col].isnull().sum()} NaNs with median = {median_val:.2f}")
    
    missing_after = df.isnull().sum().sum()
    print(f"✓ Missing values after: {missing_after}")
    
    # Double check - remove any remaining NaN rows
    if df.isnull().sum().sum() > 0:
        print(f"  WARNING: Still have NaN values, dropping rows...")
        df = df.dropna()
        print(f"  After dropping NaN rows: {len(df)} rows remaining")
    
    return df


def prepare_features_and_target(df):
    """Separate features and target, identify feature types."""
    print("\nPreparing features and target...")
    
    # Define features (exclude patient_id, timestamp, label)
    exclude_cols = ['patient_id', 'timestamp', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  Features: {feature_cols}")
    print(f"  Target: label")
    
    X = df[feature_cols]
    y = df['label']
    patient_ids = df['patient_id']
    
    print(f"✓ X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, patient_ids, feature_cols


def scale_features(X_train, X_test, feature_cols):
    """Scale numerical features using StandardScaler."""
    print("\nScaling features...")
    
    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
    
    print(f"  Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Initialize and fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    
    # Transform both train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"✓ Features scaled successfully")
    print(f"  Sample scaled values (first row):")
    print(f"    {X_train_scaled.iloc[0].to_dict()}")
    
    return X_train_scaled, X_test_scaled, scaler, numerical_cols, categorical_cols


def patient_level_train_test_split(X, y, patient_ids, test_size=0.3, random_state=42):
    """
    Split data by patient (not random rows) to prevent data leakage.
    
    CRITICAL: Each patient must be entirely in either train or test.
    Never split a single patient across train and test.
    """
    print("\nPerforming patient-level train/test split...")
    print(f"  Strategy: Split by patient_id (NOT random rows)")
    print(f"  Reason: Prevent data leakage - same patient shouldn't be in both train & test")
    
    # Get unique patients
    unique_patients = patient_ids.unique()
    n_patients = len(unique_patients)
    
    print(f"  Total unique patients: {n_patients}")
    
    # Randomly shuffle patients and split
    np.random.seed(random_state)
    shuffled_patients = np.random.permutation(unique_patients)
    
    split_idx = int(n_patients * (1 - test_size))
    train_patients = shuffled_patients[:split_idx]
    test_patients = shuffled_patients[split_idx:]
    
    print(f"  Train patients: {len(train_patients)} ({len(train_patients)/n_patients*100:.1f}%)")
    print(f"  Test patients: {len(test_patients)} ({len(test_patients)/n_patients*100:.1f}%)")
    
    # Create masks
    train_mask = patient_ids.isin(train_patients)
    test_mask = patient_ids.isin(test_patients)
    
    # Split data
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"✓ Train set: {len(X_train):,} rows")
    print(f"✓ Test set: {len(X_test):,} rows")
    print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, train_patients, test_patients


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, feature_cols, 
                           numerical_cols, categorical_cols):
    """Save preprocessed data and preprocessing objects."""
    print("\nSaving preprocessed data...")
    
    # Create preprocessed directory
    preprocess_dir = Path('data/preprocessed')
    preprocess_dir.mkdir(exist_ok=True)
    
    # Save train data
    X_train.to_csv(preprocess_dir / 'X_train.csv', index=False)
    y_train.to_csv(preprocess_dir / 'y_train.csv', index=False)
    
    # Save test data
    X_test.to_csv(preprocess_dir / 'X_test.csv', index=False)
    y_test.to_csv(preprocess_dir / 'y_test.csv', index=False)
    
    # Save scaler
    joblib.dump(scaler, preprocess_dir / 'scaler.pkl')
    
    # Save feature metadata
    metadata = {
        'feature_cols': feature_cols,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    import json
    with open(preprocess_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved preprocessed training data: {len(X_train):,} rows")
    print(f"✓ Saved preprocessed test data: {len(X_test):,} rows")
    print(f"✓ Saved scaler and metadata")
    print(f"  Location: {preprocess_dir}")
    
    return preprocess_dir


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 80)
    print("PHASE 1, STEP 1.1: DATA PREPROCESSING")
    print("=" * 80)
    
    # Step 1: Load data
    df = load_dataset()
    
    print(f"\nDataset columns before processing:")
    print(f"  {list(df.columns)}")
    
    # Step 2: Remove ECG columns
    df = remove_ecg_columns(df)
    
    print(f"\nDataset columns after removing ECG:")
    print(f"  {list(df.columns)}")
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Prepare features and target
    X, y, patient_ids, feature_cols = prepare_features_and_target(df)
    
    # Step 5: Patient-level train/test split (BEFORE scaling)
    X_train, X_test, y_train, y_test, train_patients, test_patients = \
        patient_level_train_test_split(X, y, patient_ids, test_size=0.3, random_state=42)
    
    # Step 6: Scale features
    X_train_scaled, X_test_scaled, scaler, numerical_cols, categorical_cols = \
        scale_features(X_train, X_test, feature_cols)
    
    # Step 7: Save preprocessed data
    preprocess_dir = save_preprocessed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        scaler, feature_cols, numerical_cols, categorical_cols
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {preprocess_dir}")
    print(f"\nNext step: Train machine learning models")
    print(f"Run: python scripts/train_models.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
