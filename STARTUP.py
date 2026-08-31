"""
Complete System Startup and Usage Guide

This script provides step-by-step instructions to run the entire system.
"""

import subprocess
import sys
from pathlib import Path
import time
import os

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*100}")
    print(f"{description}")
    print(f"{'='*100}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main startup guide."""
    
    print("\n" + "█" * 100)
    print("IOT HEART DISEASE PREDICTION SYSTEM - COMPLETE STARTUP GUIDE")
    print("█" * 100)
    
    project_root = Path(__file__).parent
    
    # ========================================================================
    # STEP 1: DEPENDENCIES
    # ========================================================================
    
    print("\n" + "▓" * 100)
    print("STEP 1: INSTALL DEPENDENCIES")
    print("▓" * 100)
    
    print("\n📦 Installing required Python packages...")
    success = run_command(
        "pip install -r requirements.txt",
        "Installing dependencies from requirements.txt"
    )
    
    if not success:
        print("⚠️  Some packages may not have installed correctly")
    else:
        print("✅ Dependencies installed successfully")
    
    # ========================================================================
    # STEP 2: GENERATE DATASET (if not exists)
    # ========================================================================
    
    print("\n" + "▓" * 100)
    print("STEP 2: GENERATE/VALIDATE DATASET")
    print("▓" * 100)
    
    dataset_path = project_root / 'data' / 'heart_iot_synthetic.csv'
    
    if dataset_path.exists():
        print(f"\n✅ Dataset already exists at {dataset_path}")
        print("   Skipping generation (delete data/ folder to regenerate)")
    else:
        print("\n📊 Dataset not found. Generating...")
        success = run_command(
            "python scripts/generate_dataset.py",
            "Generating synthetic dataset"
        )
        if success:
            print("✅ Dataset generated successfully")
        else:
            print("❌ Dataset generation failed")
            return False
    
    # Validate dataset
    print("\n🔍 Validating dataset...")
    success = run_command(
        "python scripts/validate_dataset.py",
        "Validating dataset integrity"
    )
    
    if success:
        print("✅ Dataset validation passed")
    else:
        print("⚠️  Dataset validation reported issues")
    
    # ========================================================================
    # STEP 3: TRAIN ML MODELS
    # ========================================================================
    
    print("\n" + "▓" * 100)
    print("STEP 3: TRAIN MACHINE LEARNING MODELS")
    print("▓" * 100)
    
    models_path = project_root / 'models' / 'random_forest.pkl'
    
    if models_path.exists():
        print(f"\n✅ Models already trained at {models_path}")
        print("   Skipping training (delete models/ folder to retrain)")
    else:
        print("\n🤖 Training ML models...")
        print("   This may take 2-5 minutes...")
        
        success = run_command(
            "python scripts/ml_pipeline.py",
            "Training ML models"
        )
        
        if success:
            print("\n✅ Model training completed successfully")
        else:
            print("\n❌ Model training failed")
            return False
    
    # ========================================================================
    # STEP 4: READY TO RUN
    # ========================================================================
    
    print("\n" + "▓" * 100)
    print("STEP 4: SYSTEM READY - NEXT STEPS")
    print("▓" * 100)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RUNNING THE COMPLETE SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────────┘

🖥️  TERMINAL 1 - Start FastAPI Backend Server:
────────────────────────────────────────────────
   $ python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
   
   ✓ API will be available at: http://localhost:8000
   ✓ Documentation at: http://localhost:8000/docs

🌐 TERMINAL 2 - Start Streamlit Dashboard:
────────────────────────────────────────────────
   $ streamlit run dashboard/app.py
   
   ✓ Dashboard will open at: http://localhost:8501
   ✓ Enter device ID when prompted

📊 TERMINAL 3 - Run Demo/Testing (Optional):
────────────────────────────────────────────────
   $ python scripts/test_demo.py
   
   ✓ Runs complete end-to-end system test
   ✓ Makes test predictions via API

📓 NOTEBOOK - Interactive Exploration:
────────────────────────────────────────────────
   $ jupyter notebook notebooks/01_dataset_exploration.ipynb
   
   ✓ Explore dataset visually
   ✓ Analyze features and distributions

┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXAMPLE WORKFLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

1. Open 3 terminal windows

2. Terminal 1 - Start API Server:
   > python -m uvicorn server.main:app --reload

3. Terminal 2 - Start Dashboard:
   > streamlit run dashboard/app.py
   
4. Terminal 3 - Send Test Data:
   > python -c "
   import requests
   from datetime import datetime
   
   data = {
       'device_id': 'patient_001',
       'timestamp': datetime.now().isoformat(),
       'heart_rate_bpm': 85.0,
       'spo2_percent': 97.5,
       'body_temperature_c': 36.8,
       'systolic_bp_mmhg': 130.0,
       'diastolic_bp_mmhg': 85.0,
       'ecg_hr_bpm': 86.0,
       'ecg_rr_interval_ms': 700.0,
       'ecg_rmssd_ms': 45.0,
       'ecg_sdnn_ms': 60.0,
       'ecg_signal_quality': 85.0,
       'activity_level': 1
   }
   response = requests.post('http://localhost:8000/predict', json=data)
   print(response.json())
   "

5. View Dashboard at http://localhost:8501
   - Enter device ID: patient_001
   - See live predictions and explanations

┌─────────────────────────────────────────────────────────────────────────────┐
│                         API ENDPOINTS REFERENCE                             │
└─────────────────────────────────────────────────────────────────────────────┘

GET /
   → Root endpoint / API info

GET /health
   → System health check

POST /predict
   → Make prediction from sensor data
   Body: {device_id, timestamp, heart_rate_bpm, spo2_percent, ...}
   Response: {prediction, risk_score, confidence, explanation, alert, message}

GET /latest/{device_id}
   → Get latest reading and prediction

GET /readings/{device_id}
   → Get recent readings

GET /predictions/{device_id}
   → Get recent predictions

GET /stats/{device_id}
   → Get device statistics

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FILES & DIRECTORIES                                │
└─────────────────────────────────────────────────────────────────────────────┘

data/
  ├── heart_iot_synthetic.csv       ← Main dataset (20,000 rows)
  ├── heart_iot_synthetic.parquet   ← Compressed format
  ├── data_dictionary.csv           ← Feature descriptions
  ├── dataset_summary.json          ← Statistics
  └── predictions.db                ← API predictions storage

models/
  ├── random_forest.pkl             ← Trained model
  ├── scaler.pkl                    ← Feature scaler
  ├── feature_names.json            ← Feature list
  └── model_comparison.csv          ← Model metrics

scripts/
  ├── generate_dataset.py           ← Dataset generation
  ├── validate_dataset.py           ← Dataset validation
  ├── ml_pipeline.py                ← Model training
  ├── ecg_processor.py              ← ECG processing
  ├── shap_explainer.py             ← SHAP explanations
  ├── test_demo.py                  ← End-to-end demo

server/
  └── main.py                       ← FastAPI server

dashboard/
  └── app.py                        ← Streamlit dashboard

notebooks/
  └── 01_dataset_exploration.ipynb  ← Interactive exploration

┌─────────────────────────────────────────────────────────────────────────────┐
│                       TROUBLESHOOTING                                       │
└─────────────────────────────────────────────────────────────────────────────┘

❌ "Model not found" error in dashboard:
   → Run: python scripts/ml_pipeline.py
   → This trains the models needed by the API

❌ "Cannot connect to API" in dashboard:
   → Make sure API server is running
   → Run: python -m uvicorn server.main:app --reload
   → Check that port 8000 is not in use

❌ Dependencies not installing:
   → Try: pip install --upgrade pip
   → Then: pip install -r requirements.txt

❌ Database error:
   → Delete: data/predictions.db
   → API will auto-create on startup

⚠️  Streamlit not found:
   → Run: pip install streamlit
   → Then: streamlit run dashboard/app.py

┌─────────────────────────────────────────────────────────────────────────────┐
│                       HARDWARE INTEGRATION                                  │
└─────────────────────────────────────────────────────────────────────────────┘

When using with real ESP32 hardware:

1. Program ESP32 to:
   - Read sensors (MAX30102, AD8232, temperature)
   - Connect to Wi-Fi
   - Send JSON to: POST http://[server_ip]:8000/predict
   - Send every 1-10 seconds

2. Example JSON payload:
   {
     "device_id": "esp32_001",
     "timestamp": "2024-01-15T10:30:45.123456",
     "heart_rate_bpm": 75.5,
     "spo2_percent": 98.2,
     "body_temperature_c": 36.8,
     "systolic_bp_mmhg": 125.0,
     "diastolic_bp_mmhg": 82.0,
     "ecg_hr_bpm": 76.0,
     "ecg_rr_interval_ms": 789.0,
     "ecg_rmssd_ms": 48.5,
     "ecg_sdnn_ms": 65.3,
     "ecg_signal_quality": 87.2,
     "activity_level": 1
   }

3. Dashboard will automatically update with:
   - Real-time vital signs
   - Risk predictions
   - SHAP explanations
   - Alert notifications

┌─────────────────────────────────────────────────────────────────────────────┐
│                       IMPORTANT DISCLAIMERS                                 │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️  SYNTHETIC DATA:
   • This dataset is for DEVELOPMENT and TESTING only
   • NOT real patient data
   • Do NOT use for clinical diagnosis
   
⚠️  RESEARCH PROTOTYPE:
   • This is NOT a medical device
   • Do NOT use for autonomous diagnosis
   • Always consult healthcare professionals
   
⚠️  EXPERIMENTAL SYSTEM:
   • Predictions are for demonstration purposes
   • Accuracy not validated against real patients
   • Should not inform medical decisions

""")
    
    print("\n" + "█" * 100)
    print("✅ SYSTEM SETUP COMPLETE!")
    print("█" * 100)
    
    print("\n🚀 Ready to start? Follow the instructions above!")
    print("\nStart with Terminal 1: python -m uvicorn server.main:app --reload")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
