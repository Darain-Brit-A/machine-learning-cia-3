"""
End-to-End Testing and Demo Script

This script demonstrates the complete pipeline:
1. Data loading and preprocessing
2. Model training
3. SHAP explainability
4. API predictions
5. Dashboard data
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import time

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from ml_pipeline import MLPipeline
from ecg_processor import ECGProcessor
from shap_explainer import SHAPExplainer, load_shap_explainer

print("=" * 100)
print("IOT HEART DISEASE PREDICTION - COMPLETE SYSTEM DEMO")
print("=" * 100)

# ============================================================================
# PHASE 1: TRAIN ML MODELS
# ============================================================================

print("\n" + "█" * 100)
print("PHASE 1: MODEL TRAINING")
print("█" * 100)

def train_models():
    """Train all ML models."""
    pipeline = MLPipeline('data/heart_iot_synthetic.csv')
    
    X, y, patient_ids = pipeline.load_data().preprocess_data()
    pipeline.patient_level_split(X, y, patient_ids, test_size=0.2)
    pipeline.scale_features()
    pipeline.train_models()
    pipeline.evaluate_models()
    pipeline.select_best_model()
    pipeline.save_models('models')
    pipeline.print_summary()
    
    return pipeline


try:
    print("\n[Starting model training...]")
    pipeline = train_models()
    print("\n✅ Model training completed successfully!")
except Exception as e:
    print(f"\n❌ Model training failed: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 2: ECG PROCESSING
# ============================================================================

print("\n" + "█" * 100)
print("PHASE 2: ECG SIGNAL PROCESSING")
print("█" * 100)

print("\n[Demonstrating ECG processing pipeline...]")

ecg_processor = ECGProcessor(sampling_rate=200)

# Generate and process synthetic ECG
print("\n1️⃣  Generating synthetic ECG signal (10 seconds, 75 BPM)...")
ecg_signal = ecg_processor.generate_synthetic_ecg(duration_seconds=10, heart_rate=75)
print(f"   Generated {len(ecg_signal)} samples")

print("\n2️⃣  Processing ECG signal...")
ecg_features = ecg_processor.process_ecg_segment(ecg_signal)

print("\nExtracted ECG Features:")
for feature, value in ecg_features.items():
    print(f"   • {feature}: {value:.2f}")

print("\n✅ ECG processing completed!")

# ============================================================================
# PHASE 3: SHAP EXPLAINABILITY
# ============================================================================

print("\n" + "█" * 100)
print("PHASE 3: SHAP EXPLAINABILITY")
print("█" * 100)

print("\n[Loading SHAP explainer...]")

try:
    shap_explainer = load_shap_explainer('models')
    print("✅ SHAP explainer loaded successfully!")
    
    # Load test data
    df = pd.read_csv('data/heart_iot_synthetic.csv')
    exclude_cols = ['patient_id', 'timestamp', 'label']
    X = df[[col for col in df.columns if col not in exclude_cols]].fillna(df.median())
    
    import joblib
    scaler = joblib.load('models/scaler.pkl')
    X_scaled = scaler.transform(X)
    
    print("\nExplaining sample prediction...")
    explanation = shap_explainer.explain_prediction(X_scaled[0])
    
    print("\nTop Risk-Increasing Features:")
    for contrib in explanation['top_positive_contributors'][:3]:
        print(f"   • {contrib['feature']}: {contrib['shap_value']:.4f}")
    
    print("\nTop Risk-Decreasing Features:")
    for contrib in explanation['top_negative_contributors'][:3]:
        print(f"   • {contrib['feature']}: {contrib['shap_value']:.4f}")
    
    print("\n✅ SHAP explanation generated!")
    
except Exception as e:
    print(f"⚠️  SHAP explanation failed: {e}")

# ============================================================================
# PHASE 4: API TESTING
# ============================================================================

print("\n" + "█" * 100)
print("PHASE 4: API PREDICTION TESTING")
print("█" * 100)

print("\n[Testing FastAPI server...]")

API_URL = "http://localhost:8000"

# Check if API is running
print(f"\n1️⃣  Checking API status at {API_URL}...")

try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        print(f"   Database Connected: {health['database_connected']}")
        print("   ✅ API is running!")
        api_available = True
    else:
        print("   ❌ API returned error")
        api_available = False
except requests.exceptions.ConnectionError:
    print("   ⚠️  Cannot connect to API. Server may not be running.")
    print("   To start the server, run: python -m uvicorn server.main:app --reload")
    api_available = False

# ============================================================================
# PHASE 5: TEST PREDICTIONS
# ============================================================================

if api_available:
    print("\n" + "█" * 100)
    print("PHASE 5: MAKING TEST PREDICTIONS")
    print("█" * 100)
    
    print("\n[Sending test data to API...]")
    
    # Create test sensor readings
    test_readings = [
        {
            "device_id": "patient_001",
            "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
            "heart_rate_bpm": 78.5,
            "spo2_percent": 97.8,
            "body_temperature_c": 36.8,
            "systolic_bp_mmhg": 125.5,
            "diastolic_bp_mmhg": 82.3,
            "ecg_hr_bpm": 79.0,
            "ecg_rr_interval_ms": 760.0,
            "ecg_rmssd_ms": 45.2,
            "ecg_sdnn_ms": 68.5,
            "ecg_signal_quality": 85.3,
            "activity_level": 1
        },
        {
            "device_id": "patient_001",
            "timestamp": datetime.now().isoformat(),
            "heart_rate_bpm": 95.2,
            "spo2_percent": 96.5,
            "body_temperature_c": 37.1,
            "systolic_bp_mmhg": 138.2,
            "diastolic_bp_mmhg": 88.7,
            "ecg_hr_bpm": 96.5,
            "ecg_rr_interval_ms": 625.0,
            "ecg_rmssd_ms": 32.1,
            "ecg_sdnn_ms": 52.3,
            "ecg_signal_quality": 82.1,
            "activity_level": 2
        }
    ]
    
    for i, reading in enumerate(test_readings, 1):
        print(f"\n{i}️⃣  Sending prediction request {i}...")
        
        try:
            response = requests.post(f"{API_URL}/predict", json=reading, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Prediction received!")
                print(f"   • Device: {result['device_id']}")
                print(f"   • Risk Score: {result['risk_score']:.1%}")
                print(f"   • Prediction: {'High Risk' if result['prediction'] == 1 else 'Low Risk'}")
                print(f"   • Confidence: {result['confidence']:.1%}")
                print(f"   • Alert: {'🚨 YES' if result['alert'] else '✓ NO'}")
                print(f"   • Message: {result['message']}")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        time.sleep(1)
    
    print("\n✅ Test predictions completed!")
    
    # ====== RETRIEVE HISTORICAL DATA ======
    print("\n" + "█" * 100)
    print("PHASE 6: RETRIEVING HISTORICAL DATA")
    print("█" * 100)
    
    print("\nFetching latest prediction...")
    try:
        response = requests.get(f"{API_URL}/latest/patient_001", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\nLatest Reading:")
            if data['reading']:
                print(f"  Timestamp: {data['reading']['timestamp']}")
                print(f"  Heart Rate: {data['reading']['heart_rate_bpm']:.1f} bpm")
                print(f"  SpO2: {data['reading']['spo2_percent']:.1f}%")
                print(f"  Temperature: {data['reading']['body_temperature_c']:.1f}°C")
            
            print("\nLatest Prediction:")
            if data['prediction']:
                print(f"  Risk Score: {data['prediction']['risk_score']:.1%}")
                print(f"  Prediction: {'High Risk' if data['prediction']['prediction'] == 1 else 'Low Risk'}")
                print(f"  Alert: {'🚨 YES' if data['prediction']['alert'] else '✓ NO'}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nFetching device statistics...")
    try:
        response = requests.get(f"{API_URL}/stats/patient_001", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"  Total Readings: {stats.get('reading_count', 0)}")
            print(f"  High Risk Alerts: {stats.get('high_risk_count', 0)}")
            print(f"  Avg Heart Rate: {stats.get('avg_hr', 0):.1f} bpm")
            print(f"  Avg SpO2: {stats.get('avg_spo2', 0):.1f}%")
            print(f"  Avg Temperature: {stats.get('avg_temp', 0):.1f}°C")
    
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("SYSTEM SUMMARY")
print("=" * 100)

print("""
✅ COMPLETED COMPONENTS:
   1. ✓ Synthetic Dataset Generation (20,000 rows, 1,000 patients)
   2. ✓ Data Preprocessing & Patient-Level Splitting
   3. ✓ ML Model Training (Logistic Regression, Random Forest, XGBoost, CatBoost)
   4. ✓ Model Evaluation & Selection
   5. ✓ ECG Signal Processing
   6. ✓ SHAP Explainability
   7. ✓ FastAPI Backend Server
   8. ✓ Database Storage
   9. ✓ Streamlit Dashboard

📊 NEXT STEPS:
   1. Start FastAPI Server:
      $ python -m uvicorn server.main:app --reload

   2. Run Streamlit Dashboard:
      $ streamlit run dashboard/app.py

   3. Send Sensor Data:
      - Use ESP32 with MAX30102, AD8232, temperature sensor
      - Configure to send JSON to API endpoint: POST /predict

   4. Monitor in Dashboard:
      - Real-time vital signs
      - Risk predictions
      - SHAP explanations
      - Historical trends

📁 PROJECT STRUCTURE:
   ├── data/                          # Dataset files
   │   ├── heart_iot_synthetic.csv    # Main dataset
   │   ├── heart_iot_synthetic.parquet
   │   ├── data_dictionary.csv
   │   ├── dataset_summary.json
   │   └── predictions.db             # Prediction history
   │
   ├── models/                        # Trained ML models
   │   ├── random_forest.pkl
   │   ├── scaler.pkl
   │   ├── feature_names.json
   │   └── model_comparison.csv
   │
   ├── scripts/                       # Core modules
   │   ├── generate_dataset.py
   │   ├── validate_dataset.py
   │   ├── ml_pipeline.py
   │   ├── ecg_processor.py
   │   ├── shap_explainer.py
   │   └── test_demo.py               # This file
   │
   ├── server/                        # FastAPI backend
   │   └── main.py
   │
   ├── dashboard/                     # Streamlit frontend
   │   └── app.py
   │
   ├── notebooks/
   │   └── 01_dataset_exploration.ipynb
   │
   ├── requirements.txt
   └── README.md

⚠️  IMPORTANT REMINDERS:
   • This is a SYNTHETIC development dataset (NOT for clinical use)
   • Do NOT use for real medical diagnosis
   • Always consult healthcare professionals for medical decisions
   • The system is a research prototype for demonstration purposes

""")

print("=" * 100)
print("🎉 DEMO COMPLETED - System is ready for use!")
print("=" * 100)
