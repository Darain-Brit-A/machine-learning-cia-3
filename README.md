# IoT Heart Disease Prediction System

An educational end-to-end prototype that collects wearable-style health readings, sends them from an ESP32 to a FastAPI service, runs a trained machine-learning model, stores the results in SQLite, and displays them in a Streamlit dashboard.

```text
Sensors -> ESP32 -> Wi-Fi -> FastAPI -> preprocessing -> ML prediction
                                      -> SQLite storage
                                      -> Streamlit dashboard
```

## Important Disclaimer

This project uses a synthetic dataset and is intended for software development, demonstrations, and education only. It is not a medical device, has not been clinically validated, and must not be used for diagnosis, treatment, or emergency decisions.

## Features

- Synthetic longitudinal heart-health dataset generation
- Dataset validation and exploration
- Patient-level machine-learning train/test splitting
- Missing-value handling, outlier clipping, and feature scaling
- Logistic Regression, Random Forest, XGBoost, and CatBoost training
- Accuracy, precision, recall, F1, ROC-AUC, and validation reports
- FastAPI REST API for sensor ingestion and prediction
- SQLite storage for readings and predictions
- Risk score, confidence, alert status, and feature explanations
- Streamlit dashboard for live readings, trends, alerts, and model information
- ESP32 firmware for Wi-Fi transmission of sensor readings
- Optional Docker Compose setup for the API and dashboard

## Current Hardware Scope

The ESP32 firmware in [esp32.ino](esp32.ino) currently supports:

- MAX30102/MAX30105 over I2C for heart-rate detection
- DHT22 on GPIO 4 for temperature readings
- Wi-Fi and HTTP communication with the API

SpO2 and blood-pressure values are currently development values. ECG is not currently sampled by the firmware; the dashboard ECG plot is synthetic. Complete physical wiring and upload instructions are in [esp32.md](esp32.md).

## Repository Structure

```text
project/
├── config.py                         Shared project configuration
├── requirements.txt                  Python dependencies
├── STARTUP.py                        Guided setup and startup helper
├── esp32.ino                         ESP32 firmware
├── esp32.md                          ESP32 wiring and integration guide
├── Dockerfile                        Container image definition
├── docker-compose.yml                API and dashboard services
├── data/
│   ├── heart_iot_synthetic.csv       Synthetic source dataset
│   ├── data_dictionary.csv           Feature descriptions
│   ├── dataset_summary.json          Dataset statistics
│   ├── preprocessed/                 Train/test data and metadata
│   └── predictions.db                Runtime SQLite database
├── models/                           Trained models and evaluation output
├── notebooks/
│   └── 01_dataset_exploration.ipynb  Interactive data exploration
├── scripts/
│   ├── generate_dataset.py           Generate synthetic data
│   ├── validate_dataset.py           Validate source data
│   ├── preprocess_data.py            Prepare features
│   ├── train_models.py               Train individual models
│   ├── ml_pipeline.py                Run the complete ML pipeline
│   ├── validate_model.py             Validate trained models
│   ├── explain_shap.py               Generate SHAP outputs
│   ├── shap_explainer.py             SHAP helper logic
│   ├── ecg_processor.py              ECG processing utilities
│   ├── mock_iot_device.py            Send development readings
│   └── test_demo.py                  End-to-end API demo
├── server/
│   └── main.py                       FastAPI app, inference, and persistence
├── dashboard/
│   └── app.py                        Streamlit user interface
└── tests/
    └── test_api.py                   API tests
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer recommended
- 2 GB or more available memory for model training
- Arduino IDE and an ESP32 board for hardware testing
- A 2.4 GHz Wi-Fi network for many ESP32 boards

Install the Python dependencies from [requirements.txt](requirements.txt). Arduino dependencies are listed in [esp32.md](esp32.md).

## Installation

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Build the Dataset and Models

The repository may already contain generated data and trained model artifacts. Run these commands when setting up from scratch or when regenerating the pipeline:

```powershell
python scripts/generate_dataset.py
python scripts/validate_dataset.py
python scripts/ml_pipeline.py
```

The pipeline prepares the data, trains several classifiers, evaluates them, and saves model artifacts in `models/`. The API loads the preferred Logistic Regression model when available and otherwise falls back to the Random Forest model.

Generated artifacts include model files, the scaler, feature metadata, comparison tables, validation reports, and SHAP outputs.

## Run the Complete System Locally

Start the API in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

Start the dashboard in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. Use the sidebar to select the device ID that is sending readings, for example `ESP32_001`.

Check the API before sending data:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

The health response should show `model_loaded` as `true`.

### Guided startup

`STARTUP.py` installs dependencies, creates or validates the dataset, trains missing models, and prints the commands for starting the API and dashboard:

```powershell
python STARTUP.py
```

It does not keep the API or dashboard running for you; start those services in separate terminals.

## API Workflow

### Sensor input

Send a JSON reading to `POST /predict`:

```json
{
  "device_id": "ESP32_001",
  "timestamp": "2026-09-13T12:00:00Z",
  "heart_rate_bpm": 85.5,
  "spo2_percent": 97.2,
  "body_temperature_c": 36.8,
  "systolic_bp_mmhg": 125.0,
  "diastolic_bp_mmhg": 80.0,
  "ecg_hr_bpm": 86.0,
  "ecg_rr_interval_ms": 700.0,
  "ecg_rmssd_ms": 45.0,
  "ecg_sdnn_ms": 60.0,
  "ecg_signal_quality": 80.0,
  "activity_level": 1
}
```

Required fields are `device_id`, `timestamp`, `heart_rate_bpm`, `spo2_percent`, `body_temperature_c`, `systolic_bp_mmhg`, and `diastolic_bp_mmhg`. ECG fields and `activity_level` are optional.

Example request:

```powershell
$payload = @{
  device_id = 'demo_sensor_001'
  timestamp = '2026-09-13T12:00:00Z'
  heart_rate_bpm = 85.5
  spo2_percent = 97.2
  body_temperature_c = 36.8
  systolic_bp_mmhg = 140.2
  diastolic_bp_mmhg = 90.1
  activity_level = 1
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $payload
```

The response contains:

- `prediction`: `0` for lower risk or `1` for higher risk
- `risk_score`: probability of the higher-risk class
- `confidence`: probability of the predicted class
- `alert`: threshold or risk alert status
- `message`: human-readable status
- `explanation`: model details and available feature importance information

### Endpoint reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information and links |
| GET | `/health` | API, model, and database health |
| POST | `/predict` | Store a reading and generate an ML prediction |
| GET | `/readings/{device_id}` | Recent raw readings |
| GET | `/predictions/{device_id}` | Recent predictions and explanations |
| GET | `/latest/{device_id}` | Latest reading and prediction pair |
| GET | `/stats/{device_id}` | Reading counts, alerts, and vital averages |

### Prediction processing

For each `/predict` request, the server:

1. Validates the JSON using the Pydantic schema.
2. Stores the sensor reading in `data/predictions.db`.
3. Builds the feature vector in the model's saved feature order.
4. Uses default `age = 60` and `sex = 0` because the current sensor payload does not include them.
5. Applies the saved scaler.
6. Runs the selected model and calculates a risk score.
7. Checks risk and vital-sign alert thresholds.
8. Stores and returns the prediction.

## Dashboard

The Streamlit dashboard provides:

- Live vital-sign metrics
- Risk score, confidence, and alert status
- Vital-sign gauges
- Blood-pressure display
- Historical risk trends
- Prediction history
- Feature-importance and SHAP sections when explanation data is available
- API, model, and database status

The dashboard reads from the API using `/latest/{device_id}`, `/predictions/{device_id}`, and `/stats/{device_id}`. It currently provides a manual refresh button. To enable automatic refreshing, install `streamlit-autorefresh` and follow the instructions in [esp32.md](esp32.md).

## ESP32 Integration

The hardware data flow is:

```text
MAX30102 + DHT22 -> ESP32 -> HTTP POST /predict -> API -> dashboard
```

Open [esp32.md](esp32.md) for:

- MAX30102, DHT22, AD8232, and blood-pressure wiring
- Arduino library installation
- Wi-Fi and computer IP configuration
- Firmware upload instructions
- Serial Monitor verification
- LAN and firewall troubleshooting

Before uploading, change these values in `esp32.ino`:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* API_URL = "http://COMPUTER_LAN_IP:8000/predict";
const char* DEVICE_ID = "ESP32_001";
```

Use the computer's LAN IP, not `localhost` or `127.0.0.1`. The ESP32 and computer must be connected to the same network.

The firmware sends readings every 10 seconds. In the current development configuration, heart rate and temperature are read from hardware while SpO2 and blood pressure remain simulated:

```cpp
const bool USE_DEVELOPMENT_SENSOR_VALUES = true;
```

## Testing

Run the API tests with:

```powershell
python -m unittest tests/test_api.py -v
```

Run the demonstration client after starting the API:

```powershell
python scripts/test_demo.py
```

The demo sends sample readings to `/predict` and prints the response. It does not represent real patient data.

## Docker Compose

Build and start the services:

```powershell
docker compose up --build
```

The API is exposed on port 8000 and the dashboard on port 8501:

- `http://localhost:8000`
- `http://localhost:8501`

The `data/` and `models/` directories are mounted into the containers so the SQLite database and model artifacts persist on the host.

For ESP32 hardware, use the host computer's LAN IP and port 8000 in `esp32.ino`. The dashboard currently uses a localhost API URL in its source, so verify API connectivity when using the dashboard inside containers and adjust its API configuration if necessary.

## Data and Model Artifacts

Important files created or consumed by the pipeline include:

| Location | Description |
|---|---|
| `data/heart_iot_synthetic.csv` | Main synthetic dataset |
| `data/data_dictionary.csv` | Feature meanings and metadata |
| `data/preprocessed/` | Train/test matrices and preprocessing metadata |
| `data/predictions.db` | Runtime readings and predictions |
| `models/feature_names.json` | Model feature order |
| `models/scaler.pkl` | Production preprocessing scaler |
| `models/logistic_regression.pkl` | Preferred API model when present |
| `models/random_forest.pkl` | API fallback model |
| `models/model_comparison.csv` | Training comparison results |
| `models/validation_report.json` | Model validation output |

## Configuration

[config.py](config.py) contains paths, API defaults, model settings, feature names, valid ranges, alert thresholds, and display ranges. The server currently resolves its paths relative to the project root and uses SQLite at `data/predictions.db`.

Important default alert thresholds include:

- Risk score above `0.7`
- Heart rate below `50` or above `120` BPM
- SpO2 below `94%`
- Systolic pressure below `90` or above `180` mmHg
- ECG signal quality below `50` when supplied

These are software thresholds for the prototype, not clinical recommendations.

## Known Limitations

- The dataset and labels are synthetic.
- The API defaults age and sex rather than receiving them from the ESP32.
- SpO2 and blood pressure are simulated in the current firmware.
- DHT22 temperature is ambient temperature, not validated body temperature.
- ECG firmware acquisition is not implemented and the dashboard waveform is synthetic.
- The dashboard's refresh slider requires the optional `streamlit-autorefresh` package for automatic polling.
- SQLite and the unauthenticated HTTP API are suitable for local demonstrations, not production deployment.
- No authentication, encryption, patient identity management, or clinical validation is implemented.

## Suggested Development Roadmap

1. Add a real SpO2 algorithm and validate it against the selected MAX30102 hardware.
2. Integrate a documented blood-pressure device through UART or BLE.
3. Add ECG sampling and feature extraction with signal-quality checks.
4. Include configured patient metadata instead of hard-coded age and sex defaults.
5. Add authentication, HTTPS, input rate limits, and a production database.
6. Replace synthetic training data with appropriately governed, clinically validated data before making any medical claim.
