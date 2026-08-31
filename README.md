# IoT-Based Heart Disease Prediction - Synthetic Development Dataset

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A SYNTHETIC DEVELOPMENT DATASET FOR SOFTWARE TESTING ONLY.**

- ❌ **DO NOT** use this for clinical validation, medical research, or real-world diagnosis
- ❌ **DO NOT** present this as real patient data or clinical evidence
- ✓ **DO USE** this for software development, pipeline testing, and architecture demonstration

The synthetic labels (0/1) are NOT real diagnoses and cannot be used to validate medical claims.

---

## Dataset Overview

A synthetic, longitudinal IoT-based heart disease prediction dataset generated for development and testing of:
- IoT sensor data pipeline
- Wearable sensor integration (MAX30102, AD8232)
- Machine learning model training
- SHAP explainability analysis
- FastAPI server integration
- Real-time dashboard prototyping

### Dataset Statistics

- **Total Rows**: 20,000+
- **Unique Patients**: ~1,000
- **Time Period**: Continuous readings (simulated)
- **Class Distribution**: ~45-55% (Low-Risk vs Higher-Risk)
- **Observations per Patient**: ~20 average
- **Missing Values**: ~1-3% (simulated real-world data quality issues)

---

## File Structure

```
project/
├── data/
│   ├── heart_iot_synthetic.csv          # Main dataset (CSV format)
│   ├── heart_iot_synthetic.parquet      # Main dataset (Parquet format)
│   ├── data_dictionary.csv              # Column descriptions
│   └── dataset_summary.json             # Statistics summary
│
├── scripts/
│   ├── generate_dataset.py              # Dataset generation script
│   └── validate_dataset.py              # Dataset validation script
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb     # Interactive exploration
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Columns Description

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `patient_id` | int | 0-999 | Unique patient identifier |
| `age` | int | 30-85 | Patient age in years |
| `sex` | int | {0, 1} | 0=Female, 1=Male |
| `heart_rate_bpm` | float | 40-150 | Wearable heart rate (BPM) |
| `spo2_percent` | float | 90-100 | Blood oxygen saturation (%) |
| `body_temperature_c` | float | 35-39 | Body temperature (°C) |
| `systolic_bp_mmhg` | float | 80-200 | Systolic blood pressure (mm Hg) |
| `diastolic_bp_mmhg` | float | 50-140 | Diastolic blood pressure (mm Hg) |
| `ecg_hr_bpm` | float | 40-150 | ECG-derived heart rate (BPM) |
| `ecg_rr_interval_ms` | float | 300-1500 | RR interval from ECG (ms) |
| `ecg_rmssd_ms` | float | 0-200 | HRV: RMSSD (Root Mean Square of Successive Differences) |
| `ecg_sdnn_ms` | float | 0-250 | HRV: SDNN (Standard Deviation of NN intervals) |
| `ecg_signal_quality` | float | 0-100 | ECG signal quality percentage |
| `activity_level` | int | {0,1,2,3} | 0=Rest, 1=Light, 2=Moderate, 3=Vigorous |
| `timestamp` | string | ISO format | UTC timestamp of measurement |
| `label` | int | {0, 1} | **SYNTHETIC** target: 0=Lower-Risk, 1=Higher-Risk |

---

## How the Dataset Was Generated

### 1. **Patient Demographics**
- ~1,000 unique synthetic patients
- Age distribution: Normal(μ=60, σ=15), clipped to 30-85 years
- Sex distribution: 55% male, 45% female

### 2. **Measurement Generation**
All measurements are **correlated and physiologically plausible**:

- **Heart Rate**: Varies by patient age, activity level, and latent risk factor
- **ECG Heart Rate**: Related to wearable HR with realistic measurement noise
- **RR Interval**: Inverse relationship to HR (60,000 / ECG_HR)
- **HRV (RMSSD, SDNN)**: Higher during rest/light activity, lower during intense activity
- **SpO2**: Typically 96-98%, lower values for higher-risk patients
- **Temperature**: Normal distribution around 36.7°C
- **Blood Pressure**: Increases with age and patient risk factor

### 3. **Label Generation**
The binary label (0/1) is generated from **multiple interacting latent factors**:
- Patient inherent risk (30%)
- Age (20%)
- Heart rate deviation (15%)
- Blood pressure elevation (15%)
- SpO2 deviation (10%)
- Random noise (10%)

**NOT a simple rule** (e.g., "HR > 100 = disease")

### 4. **Data Quality Simulation**
- ~1-3% missing values in selected fields
- ~0.5% plausible outliers
- Longitudinal measurements per patient (~20 readings each)

---

## How to Use This Dataset

### Step 1: Generate the Dataset
```bash
python scripts/generate_dataset.py
```

This will:
- Create synthetic data
- Save to `data/heart_iot_synthetic.csv` and `.parquet`
- Generate data dictionary and summary statistics
- Print a generation report

### Step 2: Validate the Dataset
```bash
python scripts/validate_dataset.py
```

This will:
- Check row/column counts
- Verify patient diversity
- Confirm class balance
- Report missing values and duplicates
- Validate feature ranges
- Check physiological plausibility

### Step 3: Explore the Dataset
Open `notebooks/01_dataset_exploration.ipynb` to:
- Visualize class distributions
- Plot feature distributions
- Examine correlation matrices
- Check missing value patterns
- Analyze outliers
- Profile observations per patient

### Step 4: Use in ML Pipeline
```python
import pandas as pd

# Load data
df = pd.read_csv('data/heart_iot_synthetic.csv')

# Preprocess features
# Split by patient (not random rows!)
# Train model (Logistic Regression, Random Forest, XGBoost)
# Evaluate & explain with SHAP
# Deploy to FastAPI
```

---

## Data Regeneration

To regenerate the dataset with different parameters:

```python
from scripts.generate_dataset import generate_synthetic_dataset

# Generate 50,000 rows with 2,000 patients
df, stats = generate_synthetic_dataset(n_rows=50000, n_patients=2000)
```

The fixed random seed (`RANDOM_SEED = 42`) ensures reproducibility.

---

## Limitations & Caveats

✓ **Good for:**
- Testing data pipelines
- ML algorithm development
- Feature engineering exploration
- SHAP explainability demo
- API & dashboard prototyping
- Teaching ML workflows

❌ **NOT suitable for:**
- Clinical validation
- Medical research publications
- Real-world diagnosis
- Regulatory compliance
- Model deployment to patients

---

## Technical Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pyarrow>=12.0.0
jupyter>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Project Architecture

```
Wearable Sensors → ESP32 → MQTT/HTTP → FastAPI 
                                          ↓
                          ECG/Feature Processing 
                                          ↓
                              ML Model Inference 
                                          ↓
                           SHAP Explanation 
                                          ↓
                    Database Storage ← → Dashboard
```

This dataset is designed to support the complete pipeline from IoT ingestion to real-time predictions with explanations.

---

## References & Further Reading

For real-world heart disease prediction:
1. UCI Machine Learning Repository - Heart Disease Dataset
2. Kaggle Heart Disease Datasets
3. PhysioNet - Cardiovascular datasets
4. Official medical literature on ECG feature extraction
5. HRV analysis standards (Task Force, 1996)

---

## License & Attribution

This synthetic dataset was generated for educational purposes.
For the complete project documentation, see the associated implementation guide.

**Generated**: 2024
**Random Seed**: 42 (reproducible)
**Status**: SYNTHETIC DEVELOPMENT DATA ONLY
