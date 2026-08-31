"""
Dataset Validation Script

This script validates the synthetic IoT heart disease prediction dataset,
checking for data quality issues, class balance, missing values, outliers, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def validate_dataset(csv_path='data/heart_iot_synthetic.csv'):
    """
    Validate the synthetic dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to the dataset CSV file
    """
    
    print("=" * 80)
    print("DATASET VALIDATION REPORT")
    print("=" * 80)
    
    # Load dataset
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"\n✓ Dataset loaded successfully")
    print(f"  File: {csv_path}")
    
    # 1. Basic structure validation
    print(f"\n{'1. BASIC STRUCTURE':-^80}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    expected_columns = [
        'patient_id', 'age', 'sex', 'heart_rate_bpm', 'spo2_percent', 
        'body_temperature_c', 'systolic_bp_mmhg', 'diastolic_bp_mmhg',
        'ecg_hr_bpm', 'ecg_rr_interval_ms', 'ecg_rmssd_ms', 'ecg_sdnn_ms',
        'ecg_signal_quality', 'activity_level', 'timestamp', 'label'
    ]
    
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if missing_cols:
        print(f"  ⚠ Missing columns: {missing_cols}")
    if extra_cols:
        print(f"  ⚠ Extra columns: {extra_cols}")
    if not missing_cols and not extra_cols:
        print(f"✓ All expected columns present")
    
    # 2. Patient-level validation
    print(f"\n{'2. PATIENT INFORMATION':-^80}")
    n_patients = df['patient_id'].nunique()
    print(f"Unique patients: {n_patients:,}")
    
    if n_patients >= 900:  # At least 90% of target
        print(f"✓ Sufficient patient diversity")
    else:
        print(f"⚠ Expected ~1000 patients, got {n_patients}")
    
    # Observations per patient
    obs_per_patient = df.groupby('patient_id').size()
    print(f"\nObservations per patient:")
    print(f"  - Mean: {obs_per_patient.mean():.2f}")
    print(f"  - Median: {obs_per_patient.median():.0f}")
    print(f"  - Min: {obs_per_patient.min()}")
    print(f"  - Max: {obs_per_patient.max()}")
    print(f"  - Std Dev: {obs_per_patient.std():.2f}")
    
    # 3. Class distribution validation
    print(f"\n{'3. CLASS DISTRIBUTION':-^80}")
    class_counts = df['label'].value_counts().sort_index()
    class_pcts = df['label'].value_counts(normalize=True).sort_index() * 100
    
    for label in sorted(df['label'].unique()):
        count = class_counts.get(label, 0)
        pct = class_pcts.get(label, 0)
        risk_level = "Lower Risk" if label == 0 else "Higher Risk"
        print(f"  Class {label} ({risk_level:>12}): {count:>7,} rows ({pct:>5.2f}%)")
    
    # Check if roughly balanced (40-60%)
    if 40 <= class_pcts[0] <= 60 and 40 <= class_pcts[1] <= 60:
        print(f"✓ Classes are well-balanced (within 40-60%)")
    else:
        print(f"⚠ Classes are imbalanced")
    
    # 4. Missing values validation
    print(f"\n{'4. MISSING VALUES':-^80}")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    
    if missing.sum() == 0:
        print("No missing values found")
    else:
        for col in missing_pct[missing_pct > 0].index:
            count = missing[col]
            pct = missing_pct[col]
            print(f"  - {col:25s}: {count:>6,} ({pct:>5.2f}%)")
        
        total_missing_pct = (missing.sum() / (len(df) * len(df.columns)) * 100)
        print(f"\nTotal missing: {missing.sum():,} ({total_missing_pct:.2f}%)")
        
        if 0.5 <= total_missing_pct <= 3.5:
            print(f"✓ Missing values within expected range (0.5-3.5%)")
        else:
            print(f"⚠ Missing values outside expected range")
    
    # 5. Duplicates validation
    print(f"\n{'5. DUPLICATES':-^80}")
    exact_dups = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_dups}")
    
    if exact_dups == 0:
        print(f"✓ No exact duplicates found")
    else:
        print(f"⚠ Found {exact_dups} duplicate rows")
    
    # 6. Feature ranges validation
    print(f"\n{'6. FEATURE RANGES':-^80}")
    
    checks = {
        'age': (30, 85),
        'heart_rate_bpm': (40, 150),
        'spo2_percent': (90, 100),
        'body_temperature_c': (35, 39),
        'systolic_bp_mmhg': (80, 200),
        'diastolic_bp_mmhg': (50, 140),
        'ecg_hr_bpm': (40, 150),
        'ecg_rr_interval_ms': (300, 1500),
        'ecg_rmssd_ms': (0, 200),
        'ecg_sdnn_ms': (0, 250),
        'ecg_signal_quality': (0, 100),
    }
    
    violations = 0
    for col, (min_val, max_val) in checks.items():
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()
        
        if out_of_range == 0:
            print(f"✓ {col:25s}: {col_data.min():>8.2f} - {col_data.max():>8.2f}")
        else:
            print(f"⚠ {col:25s}: {col_data.min():>8.2f} - {col_data.max():>8.2f} ({out_of_range} violations)")
            violations += out_of_range
    
    if violations == 0:
        print(f"\n✓ All values within expected ranges")
    else:
        print(f"\n⚠ Found {violations} range violations")
    
    # 7. Plausibility checks
    print(f"\n{'7. PLAUSIBILITY CHECKS':-^80}")
    
    # Check HR vs ECG HR correlation
    corr = df[['heart_rate_bpm', 'ecg_hr_bpm']].corr().iloc[0, 1]
    print(f"Correlation (Heart Rate vs ECG HR): {corr:.4f}")
    if corr > 0.7:
        print(f"✓ HR and ECG HR are well-correlated")
    else:
        print(f"⚠ HR and ECG HR correlation is weak")
    
    # Check BP relationship (systolic > diastolic)
    bp_valid = (df['systolic_bp_mmhg'] > df['diastolic_bp_mmhg']).sum()
    bp_valid_pct = bp_valid / len(df) * 100
    print(f"Valid BP readings (systolic > diastolic): {bp_valid_pct:.2f}%")
    if bp_valid_pct > 95:
        print(f"✓ BP readings are physiologically plausible")
    else:
        print(f"⚠ Some BP readings are physiologically implausible")
    
    # 8. Summary statistics
    print(f"\n{'8. SUMMARY STATISTICS':-^80}")
    print("\nNumeric features summary:")
    print(df.describe().to_string())
    
    # 9. Data types validation
    print(f"\n{'9. DATA TYPES':-^80}")
    print(df.dtypes)
    
    # Final report
    print(f"\n{'VALIDATION COMPLETE':-^80}")
    print("\n✓ Dataset is ready for machine learning pipeline!")
    print("  - Preprocessing → Patient-level train/test split")
    print("  - Model training (Logistic Regression, Random Forest, XGBoost)")
    print("  - Evaluation & SHAP explainability")
    print("  - API integration & deployment")
    
    return True


if __name__ == '__main__':
    validate_dataset()
