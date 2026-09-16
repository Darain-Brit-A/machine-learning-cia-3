# IoT Heart Disease Prediction System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424.svg?style=flat)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-FFA000.svg?style=flat)](https://catboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen.svg?style=flat)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com)

An end-to-end, IoT-enabled cardiovascular risk prediction and real-time vital signs monitoring prototype. The system ingests streaming health telemetry from wearable edge devices (ESP32 microcontrollers with pulse and temperature sensors), processes features through an asynchronous **FastAPI** backend, evaluates patient risk using trained **Machine Learning** models with **SHAP explainability**, stores longitudinal telemetry in **SQLite**, and presents real-time analytics in an interactive **Streamlit** clinician/patient dashboard.

```
┌─────────────────┐       Wi-Fi / HTTP        ┌────────────────────────┐
│  Wearable IoT   │ ────────────────────────> │    FastAPI Backend     │
│  (ESP32 Node)   │   POST /predict (JSON)    │    (Port 8000)         │
└─────────────────┘                           └───────────┬────────────┘
                                                          │
                    ┌─────────────────────────────────────┴───────────────────────────────────┐
                    ▼                                     ▼                                   ▼
        ┌───────────────────────┐             ┌───────────────────────┐           ┌───────────────────────┐
        │   ML Inference &      │             │    SQLite Database    │           │  Streamlit Dashboard  │
        │   SHAP Explanations   │             │   (data/predictions.db)│           │      (Port 8501)      │
        └───────────────────────┘             └───────────────────────┘           └───────────────────────┘
```

---

## ⚠️ Important Medical & Educational Disclaimer

> **IMPORTANT**: This project is developed strictly for **educational, academic, research, and prototype demonstration purposes**. 
> - The training data is **synthetic** longitudinal telemetry.
> - The software and models have **not been clinically validated or approved by regulatory bodies (FDA, CE, CDSCO, etc.)**.
> - It **must not** be used as a medical device, for clinical diagnosis, real patient treatment, emergency response, or medical decision-making.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Dataset & Feature Dictionary](#-dataset--feature-dictionary)
- [Machine Learning Pipeline & Model Evaluation](#-machine-learning-pipeline--model-evaluation)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Backend API Specification](#-backend-api-specification)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [IoT Hardware & Firmware (ESP32)](#-iot-hardware--firmware-esp32)
- [Installation & Quickstart](#-installation--quickstart)
- [Docker Deployment](#-docker-deployment)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Configuration Reference](#-configuration-reference)
- [Roadmap & Future Enhancements](#-roadmap--future-enhancements)

---

## 🏛 System Architecture

The project is designed with a decoupled 4-tier microservices/edge architecture:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. EDGE LAYER (IoT Hardware & Firmware)                                                │
│    • ESP32 NodeMCU / DevKit V1                                                         │
│    • MAX30102 Pulse Oximeter & Heart-Rate Sensor (I2C)                                  │
│    • DHT22 Ambient / Skin Temperature Sensor (GPIO 4)                                  │
│    • Edge feature packing into JSON payloads dispatched over 2.4GHz Wi-Fi               │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼ HTTP POST /predict
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. INGESTION & INFERENCE LAYER (FastAPI Backend)                                       │
│    • Request validation via strict Pydantic v2 schemas                                 │
│    • SQLite persistence for raw readings (`readings` table)                            │
│    • Feature alignment & scaling using pre-fitted `StandardScaler`                     │
│    • Real-time classification: Logistic Regression / Random Forest / XGBoost / CatBoost│
│    • Dynamic vital sign threshold alerts + high cardiovascular risk triggers          │
│    • Feature importance & local SHAP value computation                                 │
│    • Prediction archiving in SQLite (`predictions` table)                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
┌──────────────────────────────────────┐      ┌──────────────────────────────────────────┐
│ 3. PERSISTENCE LAYER (SQLite Engine) │      │ 4. VISUALIZATION LAYER (Streamlit App)   │
│    • `readings` (Raw IoT telemetry)  │      │    • Real-time KPI vital gauges          │
│    • `predictions` (Scores & SHAP)   │      │    • Dual Blood Pressure display         │
│    • ACID transactions with WAL mode │      │    • Longitudinal risk trajectory charts │
└──────────────────────────────────────┘      │    • Global & local SHAP bar charts      │
                                              │    • Active multi-device telemetry viewer│
                                              └──────────────────────────────────────────┘
```

---

## ✨ Key Features

- **End-to-End IoT Pipeline**: Automated telemetry flow from embedded firmware to cloud API to UI.
- **Robust Machine Learning Suite**: Four trained classification algorithms (Logistic Regression, Random Forest, XGBoost, CatBoost) with patient-stratified data splits to prevent information leakage.
- **Model Explainability with SHAP**: Interpret predictions transparently with Tree and Linear SHAP explainers (Waterfall & Summary charts).
- **Automated Vital Sign Alerts**: Real-time anomaly detection for Tachycardia/Bradycardia ($HR < 50$ or $> 120$ BPM), Hypoxemia ($SpO_2 < 94\%$), and Hypertensive Crisis ($SBP > 180$ or $< 90$ mmHg).
- **Asynchronous REST API**: Fast, non-blocking ingestion built with FastAPI and auto-generated Swagger/OpenAPI documentation.
- **Modern Interactive Dashboard**: Premium Streamlit interface featuring Plotly gauges, historical charts, dynamic device switching, and health metrics.
- **Comprehensive Automation**: Complete one-command pipeline launcher (`STARTUP.py`), synthetic dataset generator, ECG DSP signal processor, and mock IoT client simulator.
- **Containerized Ready**: Full multi-container Docker & Docker Compose setup with volume persistence.

---

## 📂 Repository Structure

```
.
├── config.py                         # Centralized configuration, thresholds, feature sets & paths
├── requirements.txt                  # Python runtime dependencies
├── STARTUP.py                        # Automated guided bootstrap & setup wizard
├── Dockerfile                        # Multi-stage production container image
├── docker-compose.yml                # Multi-service orchestration (API + Dashboard)
├── IMPLEMENTATION_GUIDE.md           # Step-by-step architectural design & implementation guide
│
├── data/                             # Data storage & preprocessing directory
│   ├── heart_iot_synthetic.csv       # 20,000-sample longitudinal synthetic dataset
│   ├── heart_iot_synthetic.parquet   # High-efficiency columnar storage format
│   ├── data_dictionary.csv           # Detailed metadata & feature definitions
│   ├── dataset_summary.json          # Summary statistics & cohort distributions
│   ├── predictions.db                # Runtime SQLite database (readings & predictions)
│   └── preprocessed/                 # Scaled train/test feature matrices and artifacts
│
├── models/                           # Trained ML weights, metadata & visualizations
│   ├── best_model_info.json          # Production model metadata & feature manifest
│   ├── logistic_regression.pkl       # Serialized Logistic Regression model
│   ├── random_forest.pkl             # Serialized Random Forest classifier
│   ├── xgboost.pkl                   # Serialized XGBoost model
│   ├── catboost.pkl                  # Serialized CatBoost classifier
│   ├── scaler.pkl                    # Production StandardScaler
│   ├── feature_names.json            # Strict feature ordering definition
│   ├── model_comparison.csv          # Cross-model evaluation metrics table
│   ├── validation_report.json        # 5-fold cross-validation & threshold report
│   ├── confusion_matrices.png        # Multi-model confusion matrix plots
│   ├── roc_curves.png                # Combined ROC curves plot
│   ├── precision_recall_curve.png    # Precision-Recall curves plot
│   ├── threshold_analysis.png        # Decision threshold optimization plot
│   └── shap_summary_bar_*.png        # Global SHAP importance visualizations
│
├── scripts/                          # Machine learning, DSP, and simulation utilities
│   ├── generate_dataset.py           # Synthetic longitudinal cohort generator
│   ├── validate_dataset.py           # Dataset integrity, distribution & range checker
│   ├── preprocess_data.py            # Feature cleaning, median imputation & scaling
│   ├── train_models.py               # Model training, hyperparameter setup & evaluation
│   ├── ml_pipeline.py                # End-to-end master ML orchestration pipeline
│   ├── validate_model.py             # 5-fold patient-level cross-validation & calibration
│   ├── shap_explainer.py             # Core SHAP explanation helper classes
│   ├── explain_shap.py               # SHAP artifact generation & visualization exporter
│   ├── ecg_processor.py              # Butterworth bandpass filter & Pan-Tompkins QRS DSP
│   ├── mock_iot_device.py            # Simulated streaming IoT hardware client
│   └── test_demo.py                  # End-to-end integration & scenario testing script
│
├── server/                           # FastAPI Backend application
│   ├── main.py                       # Application endpoints, inference engine & database operations
│   ├── database/                     # Modular database connection & models
│   ├── routes/                       # Modular route controllers
│   ├── schemas/                      # Pydantic validation schemas
│   └── utils/                        # Server-side utilities & loggers
│
├── dashboard/                        # Frontend UI application
│   └── app.py                        # Streamlit web dashboard with Plotly charts
│
└── tests/                            # Automated test suite
    └── test_api.py                   # API unit and integration test suite
```

---

## 📊 Dataset & Feature Dictionary

The pipeline uses a synthetic longitudinal cohort of **1,000 distinct patients** comprising **20,000 observations** (average 20 observations per patient, range 7–37).

### Patient-Level Data Splitting
To prevent **data leakage** across time-series observations of the same individual, dataset splitting is strictly stratified by `patient_id`:
- **Training Set (70%)**: 700 patients (~14,049 samples)
- **Test Set (30%)**: 300 patients (~5,951 samples)

### Feature Manifest

| Feature Name | Data Type | Valid Range | Optimal Range | Description |
|---|---|---|---|---|
| `patient_id` | Integer | $0 - 999$ | N/A | Unique synthetic patient identifier |
| `age` | Integer | $18 - 120$ | $30 - 85$ | Patient age in years |
| `sex` | Binary | $0, 1$ | $0, 1$ | Biological sex ($0 = \text{Female}, 1 = \text{Male}$) |
| `heart_rate_bpm` | Float | $20 - 200$ | $60 - 100$ | Heart rate from wearable pulse sensor (BPM) |
| `spo2_percent` | Float | $70 - 100$ | $95 - 100$ | Blood oxygen saturation ($\%$) |
| `body_temperature_c` | Float | $32.0 - 42.0$ | $36.5 - 37.5$ | Body/skin temperature ($^\circ\text{C}$) |
| `systolic_bp_mmhg` | Float | $60 - 250$ | $90 - 120$ | Systolic blood pressure ($\text{mm Hg}$) |
| `diastolic_bp_mmhg` | Float | $30 - 150$ | $60 - 80$ | Diastolic blood pressure ($\text{mm Hg}$) |
| `activity_level` | Categorical | $0 - 3$ | N/A | Physical activity level ($0=\text{Rest}, 1=\text{Light}, 2=\text{Mod}, 3=\text{Vig}$) |
| `ecg_hr_bpm` *(opt)* | Float | $20 - 200$ | $60 - 100$ | Heart rate derived from ECG waveform |
| `ecg_rr_interval_ms` *(opt)* | Float | $200 - 2000$ | $600 - 1000$ | Mean RR interval between heartbeats ($\text{ms}$) |
| `ecg_rmssd_ms` *(opt)* | Float | $0 - 300$ | $20 - 100$ | Root Mean Square of Successive Differences ($\text{ms}$) |
| `ecg_sdnn_ms` *(opt)* | Float | $0 - 400$ | $30 - 150$ | Standard deviation of NN intervals ($\text{ms}$) |
| `ecg_signal_quality` *(opt)* | Float | $0 - 100$ | $80 - 100$ | ECG signal-to-noise quality metric ($\%$) |
| **`label` (Target)** | Binary | $0, 1$ | N/A | **Target Risk**: $0 = \text{Lower Risk}, 1 = \text{Higher Risk}$ |

---

## 🤖 Machine Learning Pipeline & Model Evaluation

Four classification models were benchmarked on the test set. Due to high discriminatory power on standardized physiological vectors, **Logistic Regression** was chosen as the default lightweight, highly interpretable production model, with **Random Forest** as the ensemble backup.

### Benchmark Results (Held-Out Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **95.60%** | **79.07%** | **40.48%** | **0.535** | **0.968** | 🏆 **Active Default** |
| **Random Forest** | 95.00% | 78.65% | 27.78% | 0.411 | 0.955 | 🛡️ Secondary Fallback |
| **XGBoost** | 95.62% | 77.14% | 42.86% | 0.551 | 0.958 | ⚡ Supported |
| **CatBoost** | 95.40% | 74.81% | 40.08% | 0.522 | 0.963 | ⚡ Supported |

### 5-Fold Cross-Validation Performance (Logistic Regression)
- **Mean Cross-Validated ROC-AUC**: `0.9779 ± 0.0029`
- **Mean Cross-Validated Recall**: `95.19% ± 1.22%`
- **Mean Cross-Validated Accuracy**: `91.66% ± 0.30%`

---

## 🔍 Explainable AI (SHAP)

Trust and interpretability are paramount in biomedical computing. The system integrates **SHAP (SHapley Additive exPlanations)** to provide both macro-level and individual decision insights.

### Global Feature Importance
Across the cohort, feature attribution follows clear clinical patterns:
1. **Systolic Blood Pressure (`systolic_bp_mmhg`)**: Highest overall positive contributor to cardiovascular risk score.
2. **Heart Rate (`heart_rate_bpm`)**: Elevated resting heart rate strongly shifts risk score upwards.
3. **Age (`age`)**: Progressive demographic baseline factor.
4. **Diastolic Blood Pressure (`diastolic_bp_mmhg`)**: Significant secondary hemodynamic marker.
5. **Blood Oxygen (`spo2_percent`)**: Protective negative contribution when $>97\%$; risk driver when $<94\%$.

### Local Prediction Transparency
Every response returned by `POST /predict` contains local explanation data showing which vital signs pushed the patient towards higher or lower risk categories.

---

## 🔌 Backend API Specification

The backend is built with **FastAPI** running on **Uvicorn** (`http://localhost:8000`).

### Interactive Documentation
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoint Summary

| Method | Endpoint | Description | Request / Query |
|---|---|---|---|
| `GET` | `/` | Service root and navigation links | None |
| `GET` | `/health` | System status, DB connectivity, model loading check | None |
| `POST` | `/predict` | Ingests sensor reading, stores in DB, runs ML inference & alerts | JSON `SensorReading` body |
| `GET` | `/latest/{device_id}` | Retrieves most recent reading & prediction pair for a device | Path: `device_id` |
| `GET` | `/readings/{device_id}` | Retrieves historical raw telemetry points | Path: `device_id`, Query: `limit` |
| `GET` | `/predictions/{device_id}` | Retrieves historical predictions with explanations | Path: `device_id`, Query: `limit` |
| `GET` | `/stats/{device_id}` | Aggregate statistics (reading counts, alert count, vital averages)| Path: `device_id` |

### Sample `POST /predict` Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "device_id": "ESP32_001",
       "timestamp": "2026-09-16T12:00:00Z",
       "heart_rate_bpm": 88.5,
       "spo2_percent": 96.8,
       "body_temperature_c": 36.7,
       "systolic_bp_mmhg": 145.0,
       "diastolic_bp_mmhg": 92.0,
       "activity_level": 1
     }'
```

### Sample Response Payload

```json
{
  "device_id": "ESP32_001",
  "timestamp": "2026-09-16T12:00:00Z",
  "input_features": {
    "age": 60.0,
    "sex": 0.0,
    "heart_rate_bpm": 88.5,
    "spo2_percent": 96.8,
    "body_temperature_c": 36.7,
    "systolic_bp_mmhg": 145.0,
    "diastolic_bp_mmhg": 92.0,
    "activity_level": 1.0
  },
  "prediction": 1,
  "risk_score": 0.742,
  "confidence": 0.742,
  "explanation": {
    "model_type": "LogisticRegression",
    "prediction_class": "Higher Risk",
    "risk_score": 0.742,
    "confidence": 0.742
  },
  "alert": true,
  "message": "⚠️ Higher Risk (Score: 74.20%)"
}
```

---

## 💻 Streamlit Dashboard

The frontend application (`dashboard/app.py`) runs at `http://localhost:8501`.

### Key Dashboard Views & Modules
- **Vital Signs Overview Cards**: Real-time cards displaying Heart Rate (BPM), $SpO_2$ ($\%$), Body Temperature ($^\circ\text{C}$), and Blood Pressure ($\text{mm Hg}$) with physiological status indicators.
- **Risk Evaluation Gauges**: Plotly radial gauge displaying current cardiovascular risk percentage, color-coded into *Lower Risk* (Green), *Moderate Risk* (Amber), and *High Risk* (Red).
- **Time-Series Telemetry Trends**: Historical charts showing vital signs over time alongside changing risk score trajectories.
- **Explainability Explorer**: Interactive view displaying top feature contributions for the latest telemetry reading.
- **Device Management & Historical Logs**: Device selector dropdown, historical telemetry table, raw database query explorer, and alert counters.

---

## 📡 IoT Hardware & Firmware (ESP32)

### Hardware Components
1. **ESP32 DevKit V1 Microcontroller** (Wi-Fi 802.11 b/g/n, 2.4 GHz)
2. **MAX30102 / MAX30105 Pulse Oximeter Module** (I2C interface)
3. **DHT22 Temperature & Humidity Sensor** (Single-bus digital interface)
4. *(Optional / Roadmap)* **AD8232 Single-Lead ECG Monitor** (Analog ADC input)

### Circuit Wiring Table

| Sensor / Module | Sensor Pin | ESP32 GPIO Pin | Description |
|---|---|---|---|
| **MAX30102** | `VCC` | `3V3` / `VIN` (3.3V) | Power supply |
| | `GND` | `GND` | Ground |
| | `SDA` | `GPIO 21` | I2C Data line |
| | `SCL` | `GPIO 22` | I2C Clock line |
| **DHT22** | `VCC` | `3V3` | Power supply |
| | `GND` | `GND` | Ground |
| | `DATA` | `GPIO 4` | Digital data pin (with 10k pullup) |

### Firmware Configuration
In your Arduino IDE or PlatformIO project, configure your local network credentials and server IP:
```cpp
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* API_URL       = "http://<YOUR_COMPUTER_LAN_IP>:8000/predict";
const char* DEVICE_ID     = "ESP32_001";
```

> **Note**: For local development without physical hardware, use the built-in simulator:
> ```powershell
> python scripts/mock_iot_device.py
> ```

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- (Optional) Docker and Docker Compose

### 1. Clone the Repository
```bash
git clone https://github.com/Darain-Brit-A/machine-learning-cia-3.git
cd machine-learning-cia-3
```

### 2. Setup Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Automated Bootstrap (One Command)
Run the guided startup utility to verify data, train all models, and validate artifacts:
```bash
python STARTUP.py
```

### 4. Or Run Individual Pipeline Steps Manually
```bash
# Step 1: Generate synthetic dataset
python scripts/generate_dataset.py

# Step 2: Validate dataset integrity
python scripts/validate_dataset.py

# Step 3: Run the end-to-end ML pipeline (training + SHAP)
python scripts/ml_pipeline.py
```

### 5. Launch Application Services

**Terminal 1 — Start the FastAPI Backend:**
```powershell
python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start the Streamlit Dashboard:**
```powershell
streamlit run dashboard/app.py
```

**Terminal 3 (Optional) — Run Mock IoT Sensor Stream:**
```powershell
python scripts/mock_iot_device.py
```

Access the dashboard at **`http://localhost:8501`** and API documentation at **`http://localhost:8000/docs`**.

---

## 🐳 Docker Deployment

The repository includes a ready-to-run multi-container setup via Docker Compose.

```powershell
# Build and run API and Dashboard containers
docker compose up --build
```

- **FastAPI Backend**: `http://localhost:8000`
- **Streamlit Dashboard**: `http://localhost:8501`
- Host volumes `data/` and `models/` are mounted automatically for data persistence.

To stop the containers:
```powershell
docker compose down
```

---

## 🧪 Testing & Quality Assurance

### Automated Unit & API Tests
Run the test suite using Python's built-in `unittest`:
```powershell
python -m unittest tests/test_api.py -v
```

### End-to-End System Demo Test
Execute the comprehensive end-to-end scenario testing script to verify all live endpoints, database persistence, and threshold alerts:
```powershell
python scripts/test_demo.py
```

---

## ⚙️ Configuration Reference

All constants, model hyperparameters, and alerting thresholds are maintained centrally in [`config.py`](config.py):

| Parameter | Default Value | Description |
|---|---|---|
| `ALERT_THRESHOLDS['high_risk_score']` | `0.70` | Model risk score alert cutoff ($> 70\%$) |
| `ALERT_THRESHOLDS['heart_rate_min']` | `50 BPM` | Bradycardia warning threshold |
| `ALERT_THRESHOLDS['heart_rate_max']` | `120 BPM` | Tachycardia warning threshold |
| `ALERT_THRESHOLDS['spo2_min']` | `94.0%` | Hypoxemia warning threshold |
| `ALERT_THRESHOLDS['systolic_bp_max']` | `180 mm Hg` | Hypertensive crisis warning cutoff |
| `ALERT_THRESHOLDS['systolic_bp_min']` | `90 mm Hg` | Hypotension warning cutoff |
| `TEST_SIZE` | `0.2` | Evaluation split proportion |
| `RANDOM_SEED` | `42` | Global reproducibility seed |

---

## 🗺️ Roadmap & Future Enhancements

- [ ] **Hardware SpO2 DSP**: Integrate real-time Red/IR AC/DC ratio extraction algorithms for raw MAX30102 photodiode values.
- [ ] **Hardware Blood Pressure**: Connect cuff-based or optical PTT blood-pressure devices over UART / Bluetooth Low Energy (BLE).
- [ ] **Edge ECG Integration**: Deploy real-time Pan-Tompkins QRS wave detection directly on ESP32 firmware using AD8232 analog inputs.
- [ ] **Patient Profile Registry**: Implement user login and dynamic patient profile management (individual baseline age, sex, medical history).
- [ ] **Production Cloud Deployment**: Transition SQLite to PostgreSQL / TimescaleDB with OAuth2 JWT authentication and TLS/HTTPS encryption.

---

## 📄 License & Attribution

This project is created for academic coursework and educational demonstration purposes. All synthetic datasets and code artifacts are provided for learning, research, and non-clinical development.
