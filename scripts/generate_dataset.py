"""
IoT-Based Real-Time Heart Disease Prediction - Synthetic Dataset Generator

This script generates a SYNTHETIC development dataset for software development and 
pipeline testing only. DO NOT present this as real patient data or clinical evidence.

Dataset: ~20,000 rows with ~1,000 synthetic patients
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_synthetic_dataset(n_rows=20000, n_patients=1000):
    """
    Generate synthetic IoT heart disease prediction dataset.
    
    Parameters:
    -----------
    n_rows : int
        Total number of observations
    n_patients : int
        Number of unique patients
    
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with all features
    """
    
    print("=" * 80)
    print("SYNTHETIC DATASET GENERATION REPORT")
    print("=" * 80)
    print(f"\n[1/5] Initializing parameters...")
    print(f"  - Target rows: {n_rows:,}")
    print(f"  - Target patients: {n_patients:,}")
    
    # Generate patient IDs and latent risk factors
    patient_ids = np.random.choice(n_patients, size=n_rows, replace=True)
    
    # Create patient-level latent risk factor (used for label generation)
    patient_risk = np.random.beta(a=2, b=5, size=n_patients)  # Most low-risk
    
    # Initialize lists to store data
    data = {
        'patient_id': [],
        'age': [],
        'sex': [],
        'heart_rate_bpm': [],
        'spo2_percent': [],
        'body_temperature_c': [],
        'systolic_bp_mmhg': [],
        'diastolic_bp_mmhg': [],
        'ecg_hr_bpm': [],
        'ecg_rr_interval_ms': [],
        'ecg_rmssd_ms': [],
        'ecg_sdnn_ms': [],
        'ecg_signal_quality': [],
        'activity_level': [],
        'timestamp': [],
        'label': []
    }
    
    print(f"\n[2/5] Generating patient demographics and measurements...")
    
    # Timestamp generation
    start_time = datetime(2024, 1, 1)
    
    for idx in range(n_rows):
        if (idx + 1) % 5000 == 0:
            print(f"  - Generated {idx + 1:,}/{n_rows:,} rows...")
        
        # Patient ID
        pid = patient_ids[idx]
        data['patient_id'].append(pid)
        
        # Age: sampled from realistic distribution (30-80 years)
        age = np.random.normal(loc=60, scale=15)
        age = np.clip(age, 30, 85)
        data['age'].append(int(age))
        
        # Sex: binary (0=Female, 1=Male)
        sex = np.random.binomial(1, 0.55)  # 55% male, 45% female
        data['sex'].append(sex)
        
        # Activity level: 0=rest, 1=light, 2=moderate, 3=vigorous
        activity = np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.2, 0.05])
        data['activity_level'].append(activity)
        
        # Heart rate (correlated with activity and age)
        base_hr = 70 + age * 0.1 + patient_risk[pid] * 20
        activity_hr = activity * 15
        noise_hr = np.random.normal(0, 5)
        hr = base_hr + activity_hr + noise_hr
        hr = np.clip(hr, 40, 150)
        data['heart_rate_bpm'].append(round(hr, 1))
        
        # ECG heart rate (related to wearable HR with noise)
        ecg_hr = hr + np.random.normal(0, 3)
        ecg_hr = np.clip(ecg_hr, 40, 150)
        data['ecg_hr_bpm'].append(round(ecg_hr, 1))
        
        # RR Interval: inversely related to HR (60000 / HR)
        rr_interval = 60000 / ecg_hr + np.random.normal(0, 5)
        rr_interval = np.clip(rr_interval, 300, 1500)
        data['ecg_rr_interval_ms'].append(round(rr_interval, 1))
        
        # HRV features (RMSSD, SDNN) - vary with physiological state
        if activity <= 1:  # Rest or light activity - higher HRV
            rmssd = np.random.normal(50, 15)
            sdnn = np.random.normal(80, 20)
        else:  # Moderate/vigorous - lower HRV
            rmssd = np.random.normal(30, 10)
            sdnn = np.random.normal(50, 15)
        
        rmssd = np.clip(rmssd, 5, 150)
        sdnn = np.clip(sdnn, 10, 200)
        data['ecg_rmssd_ms'].append(round(rmssd, 1))
        data['ecg_sdnn_ms'].append(round(sdnn, 1))
        
        # SpO2: usually high, with small number of lower values
        if patient_risk[pid] > 0.7:  # Higher risk patients
            spo2_mean = 96
        else:
            spo2_mean = 97.5
        
        spo2 = np.random.normal(spo2_mean, 1.5)
        spo2 = np.clip(spo2, 90, 100)
        data['spo2_percent'].append(round(spo2, 1))
        
        # Body temperature (plausible human range: 36.1-37.2°C)
        temp = np.random.normal(36.7, 0.3)
        temp = np.clip(temp, 35.5, 38.5)
        data['body_temperature_c'].append(round(temp, 2))
        
        # Blood pressure (varies with age and latent risk)
        systolic = 120 + age * 0.3 + patient_risk[pid] * 25
        systolic += np.random.normal(0, 8)
        systolic = np.clip(systolic, 90, 180)
        
        diastolic = 80 + age * 0.15 + patient_risk[pid] * 15
        diastolic += np.random.normal(0, 5)
        diastolic = np.clip(diastolic, 60, 120)
        
        data['systolic_bp_mmhg'].append(round(systolic, 1))
        data['diastolic_bp_mmhg'].append(round(diastolic, 1))
        
        # ECG signal quality (0-100%, affected by movement and sensor quality)
        base_quality = 90 - activity * 10
        quality = base_quality + np.random.normal(0, 5)
        quality = np.clip(quality, 30, 100)
        data['ecg_signal_quality'].append(round(quality, 1))
        
        # Timestamp (sequential, one reading per minute approximately)
        timestamp = start_time + timedelta(minutes=idx)
        data['timestamp'].append(timestamp.isoformat())
        
        # Label: derived from multiple interacting factors (multi-variate)
        # NOT a simple rule-based threshold
        risk_score = (
            0.3 * patient_risk[pid] +  # Patient inherent risk
            0.2 * (age / 85) +  # Age factor
            0.15 * ((hr - 60) / 90) +  # HR deviation
            0.15 * ((systolic - 120) / 60) +  # BP elevation
            0.1 * ((spo2 - 98) / 8) +  # SpO2 deviation
            0.1 * np.random.normal(0, 0.1)  # Random variation
        )
        
        risk_score = np.clip(risk_score, 0, 1)
        label = 1 if risk_score > 0.5 else 0
        data['label'].append(label)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n[3/5] Adding missing values...")
    # Add ~1-3% missing values to selected fields
    missing_rate = 0.02
    missing_fields = ['ecg_rmssd_ms', 'ecg_sdnn_ms', 'spo2_percent', 'ecg_signal_quality']
    
    for field in missing_fields:
        missing_idx = np.random.choice(
            len(df), 
            size=int(len(df) * missing_rate), 
            replace=False
        )
        df.loc[missing_idx, field] = np.nan
    
    print(f"  - Added missing values to: {', '.join(missing_fields)}")
    
    print(f"\n[4/5] Adding plausible outliers and noise...")
    # Add ~0.5% plausible outliers
    outlier_idx = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
    for idx in outlier_idx:
        # Randomly perturb one field
        field = np.random.choice(['heart_rate_bpm', 'systolic_bp_mmhg', 'spo2_percent'])
        current_val = df.loc[idx, field]
        if pd.notna(current_val):
            df.loc[idx, field] = current_val * np.random.uniform(1.15, 1.35)
    
    print(f"  - Added {len(outlier_idx)} outlier observations")
    
    print(f"\n[5/5] Calculating dataset statistics...")
    
    # Calculate statistics for summary
    stats = {
        'generation_date': datetime.now().isoformat(),
        'total_rows': len(df),
        'unique_patients': df['patient_id'].nunique(),
        'age_range': [int(df['age'].min()), int(df['age'].max())],
        'class_0_count': int((df['label'] == 0).sum()),
        'class_1_count': int((df['label'] == 1).sum()),
        'class_0_percent': float(round((df['label'] == 0).sum() / len(df) * 100, 2)),
        'class_1_percent': float(round((df['label'] == 1).sum() / len(df) * 100, 2)),
        'missing_values_per_column': df.isnull().sum().to_dict(),
        'observations_per_patient_mean': float(round(df.groupby('patient_id').size().mean(), 2)),
        'observations_per_patient_min': int(df.groupby('patient_id').size().min()),
        'observations_per_patient_max': int(df.groupby('patient_id').size().max())
    }
    
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Unique patients: {stats['unique_patients']:,}")
    print(f"Age range: {stats['age_range'][0]} - {stats['age_range'][1]} years")
    print(f"\nClass Distribution:")
    print(f"  - Class 0 (Lower Risk): {stats['class_0_count']:,} ({stats['class_0_percent']:.1f}%)")
    print(f"  - Class 1 (Higher Risk): {stats['class_1_count']:,} ({stats['class_1_percent']:.1f}%)")
    print(f"\nObservations per patient:")
    print(f"  - Mean: {stats['observations_per_patient_mean']:.1f}")
    print(f"  - Min: {stats['observations_per_patient_min']}")
    print(f"  - Max: {stats['observations_per_patient_max']}")
    print("\nMissing values:")
    for col, count in stats['missing_values_per_column'].items():
        if count > 0:
            pct = count / len(df) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")
    print("\n" + "=" * 80)
    
    return df, stats


def create_data_dictionary():
    """Create a data dictionary describing all columns."""
    
    data_dict = {
        'patient_id': 'Unique identifier for each synthetic patient (0 to 999)',
        'age': 'Patient age in years (30-85)',
        'sex': 'Biological sex: 0=Female, 1=Male',
        'heart_rate_bpm': 'Heart rate measured by wearable sensor (beats per minute)',
        'spo2_percent': 'Blood oxygen saturation from wearable sensor (%)',
        'body_temperature_c': 'Body temperature from wearable sensor (Celsius)',
        'systolic_bp_mmhg': 'Systolic blood pressure measurement (mm Hg)',
        'diastolic_bp_mmhg': 'Diastolic blood pressure measurement (mm Hg)',
        'ecg_hr_bpm': 'Heart rate derived from ECG signal (beats per minute)',
        'ecg_rr_interval_ms': 'ECG RR interval (milliseconds) - time between heartbeats',
        'ecg_rmssd_ms': 'Root Mean Square of Successive Differences (HRV measure, ms)',
        'ecg_sdnn_ms': 'Standard Deviation of NN intervals (HRV measure, ms)',
        'ecg_signal_quality': 'ECG signal quality (0-100%)',
        'activity_level': 'Physical activity level: 0=Rest, 1=Light, 2=Moderate, 3=Vigorous',
        'timestamp': 'ISO format timestamp of the measurement',
        'label': 'Target variable: 0=Lower Risk, 1=Higher Risk (SYNTHETIC - FOR DEVELOPMENT ONLY)'
    }
    
    df_dict = pd.DataFrame([
        {'Column': col, 'Description': desc} 
        for col, desc in data_dict.items()
    ])
    
    return df_dict


def main():
    """Main execution function."""
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    df, stats = generate_synthetic_dataset(n_rows=20000, n_patients=1000)
    
    print(f"\n[Saving] Exporting dataset to multiple formats...")
    
    # Save as CSV
    csv_path = data_dir / 'heart_iot_synthetic.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path}")
    
    # Save as Parquet
    parquet_path = data_dir / 'heart_iot_synthetic.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Saved Parquet: {parquet_path}")
    
    # Save data dictionary
    data_dict_df = create_data_dictionary()
    dict_path = data_dir / 'data_dictionary.csv'
    data_dict_df.to_csv(dict_path, index=False)
    print(f"  ✓ Saved Data Dictionary: {dict_path}")
    
    # Save summary statistics as JSON
    summary_path = data_dir / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  ✓ Saved Dataset Summary: {summary_path}")
    
    print("\n✓ Dataset generation completed successfully!")
    print(f"\nIMPORTANT: This is a SYNTHETIC development dataset.")
    print("Do NOT use this for clinical validation or medical research claims.")
    print("See README.md for more information.")


if __name__ == '__main__':
    main()
