# IoT Heart Disease Prediction - Implementation Guide
## Step-by-Step Implementation Without ECG

**Project Overview:** IoT-based real-time heart disease prediction using wearable sensors (heart rate, SpO2, temperature, blood pressure) with machine learning and explainability.

---

## Table of Contents
1. [Phase 1: ML Pipeline Development](#phase-1-ml-pipeline-development)
2. [Phase 2: FastAPI Server Setup](#phase-2-fastapi-server-setup)
3. [Phase 3: Dashboard Development](#phase-3-dashboard-development)
4. [Phase 4: IoT Integration](#phase-4-iot-integration)
5. [Phase 5: Testing & Validation](#phase-5-testing--validation)
6. [Phase 6: Deployment](#phase-6-deployment)

---

# PHASE 1: ML PIPELINE DEVELOPMENT

## Step 1.1: Data Preprocessing

### Objective
Prepare the synthetic dataset for machine learning by handling missing values, scaling features, and creating a patient-level train/test split.

### Implementation Details

**1.1.1 Load & Explore Data**
```
Input: data/heart_iot_synthetic.csv
Output: Loaded DataFrame with basic statistics
```

**1.1.2 Handle Missing Values**
- Identify missing values in: spo2_percent, ecg_signal_quality, ecg_rmssd_ms, ecg_sdnn_ms
- Strategy: Use median imputation for sensor readings (physiological stability)
- Percentage: ~2% missing per column

**1.1.3 Remove ECG Columns** (Since not available)
- Delete: ecg_hr_bpm, ecg_rr_interval_ms, ecg_rmssd_ms, ecg_sdnn_ms, ecg_signal_quality
- Keep only wearable sensor features

**1.1.4 Feature Scaling**
- Apply StandardScaler to numerical features
- Age, heart_rate_bpm, spo2_percent, body_temperature_c, systolic_bp_mmhg, diastolic_bp_mmhg
- Keep categorical features as-is: sex, activity_level

**1.1.5 Patient-Level Train/Test Split**
- **CRITICAL:** Split by patient_id, NOT random rows
- Why? Prevent data leakage (same patient in train and test)
- Ratio: 70% patients (train), 30% patients (test)
- Result: Training set with ~700 patients, Test set with ~300 patients

### Code Output
- Preprocessed training data
- Preprocessed test data
- Scaler object (for production)
- Feature names list

---

## Step 1.2: Model Training

### Objective
Train multiple classification models and compare performance metrics.

### Implementation Details

**1.2.1 Model Selection**
Train 4 baseline models:
1. Logistic Regression (interpretable baseline)
2. Random Forest (ensemble, good baseline)
3. XGBoost (gradient boosting)
4. CatBoost (handles mixed data types well)

**1.2.2 Training Process**

For each model:
```
1. Initialize model with hyperparameters
2. Train on preprocessed training data
3. Make predictions on test set
4. Calculate evaluation metrics
5. Save trained model
```

**1.2.3 Evaluation Metrics**
For each model, calculate:
- **Accuracy:** (TP + TN) / Total
- **Precision:** TP / (TP + FP) - How many predicted positive are correct?
- **Recall:** TP / (TP + FN) - How many actual positives are found?
- **F1-Score:** Harmonic mean of Precision & Recall
- **ROC-AUC:** Area under receiver operating characteristic curve
- **Confusion Matrix:** TP, TN, FP, FN visualization
- **Calibration:** Probability calibration plot

**1.2.4 Model Selection Criteria**
- Best ROC-AUC score (primary)
- Good recall (find positives even if false alarms)
- Interpretability (for healthcare)
- Production deployment readiness

### Code Output
- Trained model objects (4 models)
- Evaluation report (CSV/JSON)
- Performance comparison plots
- Selected best model

---

## Step 1.3: SHAP Explainability

### Objective
Explain why the model makes predictions for transparency and trust.

### Implementation Details

**1.3.1 SHAP Analysis**
- Use SHAP (SHapley Additive exPlanations) for model interpretation
- Works with tree-based models (Random Forest, XGBoost, CatBoost)

**1.3.2 Global Interpretability**
**Feature Importance Analysis:**
- Identify which features matter most for ALL predictions
- Shap values aggregated across entire dataset
- Output: Ranked list of features by importance

**Example Output:**
```
1. systolic_bp_mmhg      - 0.42 (42% of prediction importance)
2. heart_rate_bpm        - 0.28 (28%)
3. age                   - 0.15 (15%)
4. diastolic_bp_mmhg     - 0.10 (10%)
5. spo2_percent          - 0.03 (3%)
6. body_temperature_c    - 0.02 (2%)
```

**1.3.3 Local Interpretability**
**Individual Prediction Explanation:**
- For each patient prediction, show contribution of each feature
- Feature value + impact on prediction = explanation

**Example Output for Patient:**
```
Patient ID: 123
Prediction: HIGH RISK (91% probability)

Feature Contributions:
- systolic_bp_mmhg: 165 → +0.38 (pushes toward HIGH RISK)
- heart_rate_bpm: 120 → +0.22 (pushes toward HIGH RISK)
- age: 72 → +0.15 (pushes toward HIGH RISK)
- spo2_percent: 95 → -0.12 (pushes toward LOW RISK)
- diastolic_bp_mmhg: 95 → +0.08 (pushes toward HIGH RISK)
```

**1.3.4 Visualization**
- SHAP Summary Plots (feature importance)
- SHAP Dependence Plots (feature value vs SHAP value)
- SHAP Force Plot (individual prediction breakdown)
- Waterfall Plot (single prediction explanation)

### Code Output
- SHAP Explainer object
- Feature importance rankings
- Explanation templates for predictions
- Visualization functions

---

## Step 1.4: Model Validation & Performance Report

### Objective
Comprehensive validation of model quality before deployment.

### Implementation Details

**1.4.1 Cross-Validation**
- 5-fold patient-level cross-validation
- Each fold: different 20% of patients held out
- Ensures robust performance estimate

**1.4.2 Class Balance Metrics**
- Check if model performs equally well on both classes
- ROC curve (TPR vs FPR trade-off)
- Precision-Recall curve (for imbalanced data)

**1.4.3 Threshold Analysis**
- By default, classification threshold = 0.5
- But for healthcare, maybe want to err on side of caution
- Analyze: If we lower threshold to 0.4, recall improves but precision drops
- Choose optimal threshold based on clinical requirements

**1.4.4 Failure Analysis**
- Identify predictions where model is most uncertain
- Analyze false positives (healthy flagged as at-risk)
- Analyze false negatives (at-risk flagged as healthy) - MORE CRITICAL!

### Code Output
- Cross-validation scores
- Threshold optimization analysis
- Performance validation report
- Uncertainty estimates

---

# PHASE 2: FASTAPI SERVER SETUP

## Step 2.1: Project Structure

```
backend/
├── main.py                      # FastAPI app entry point
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── models/
│   ├── __init__.py
│   ├── predictor.py             # Model loading & inference
│   └── explainer.py             # SHAP explanation generation
├── schemas/
│   ├── __init__.py
│   ├── input.py                 # Request schemas (Pydantic)
│   └── output.py                # Response schemas
├── routes/
│   ├── __init__.py
│   ├── predict.py               # Prediction endpoints
│   ├── health.py                # Health check endpoints
│   └── history.py               # Patient history endpoints
├── database/
│   ├── __init__.py
│   ├── db.py                    # Database connection
│   └── models.py                # SQLAlchemy models
└── utils/
    ├── __init__.py
    ├── preprocessing.py         # Feature scaling
    └── logger.py                # Logging setup
```

---

## Step 2.2: API Endpoints

### Endpoint 1: Prediction
```
POST /api/v1/predict
Input: Patient IoT sensor readings
Output: Risk prediction + explanation
```

**Input JSON:**
```json
{
  "patient_id": 123,
  "age": 65,
  "sex": 1,
  "heart_rate_bpm": 85.5,
  "spo2_percent": 97.2,
  "body_temperature_c": 36.8,
  "systolic_bp_mmhg": 140.2,
  "diastolic_bp_mmhg": 90.1,
  "activity_level": 1
}
```

**Output JSON:**
```json
{
  "prediction_id": "pred_12345",
  "patient_id": 123,
  "prediction": 1,
  "probability": 0.72,
  "confidence": "HIGH",
  "risk_level": "HIGHER RISK",
  "timestamp": "2024-01-31T14:30:00Z",
  "explanation": {
    "top_features": [
      {
        "feature": "systolic_bp_mmhg",
        "value": 140.2,
        "contribution": 0.38,
        "direction": "increases_risk"
      },
      {
        "feature": "heart_rate_bpm",
        "value": 85.5,
        "contribution": 0.22,
        "direction": "increases_risk"
      }
    ],
    "reasoning": "Elevated blood pressure and normal heart rate suggest cardiovascular stress."
  }
}
```

### Endpoint 2: Get Patient History
```
GET /api/v1/patients/{patient_id}/history
Output: Historical predictions for a patient
```

**Query Parameters:**
- `limit`: Number of recent predictions (default: 10)
- `days`: Last N days of history (default: 30)

**Output JSON:**
```json
{
  "patient_id": 123,
  "total_predictions": 45,
  "risk_trend": "stable",
  "predictions": [
    {
      "timestamp": "2024-01-31T14:30:00Z",
      "prediction": 1,
      "probability": 0.72
    }
  ]
}
```

### Endpoint 3: Health Check
```
GET /api/v1/health
Output: API status and model status
```

**Output JSON:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "timestamp": "2024-01-31T14:30:00Z"
}
```

### Endpoint 4: Model Info
```
GET /api/v1/model/info
Output: Model metadata and performance
```

**Output JSON:**
```json
{
  "model_name": "XGBoost_Heart_Disease_Predictor",
  "model_version": "1.0.0",
  "training_date": "2024-01-31",
  "training_samples": 14000,
  "test_auc": 0.92,
  "test_accuracy": 0.87,
  "feature_count": 8,
  "features": ["age", "sex", "heart_rate_bpm", "..."]
}
```

---

## Step 2.3: Data Models (Pydantic Schemas)

**Input Schema:**
```python
class SensorReading(BaseModel):
    patient_id: int
    age: int
    sex: int  # 0=Female, 1=Male
    heart_rate_bpm: float
    spo2_percent: float
    body_temperature_c: float
    systolic_bp_mmhg: float
    diastolic_bp_mmhg: float
    activity_level: int  # 0=Rest, 1=Light, 2=Moderate, 3=Vigorous
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": 123,
                "age": 65,
                "sex": 1,
                "heart_rate_bpm": 85.5,
                "spo2_percent": 97.2,
                "body_temperature_c": 36.8,
                "systolic_bp_mmhg": 140.2,
                "diastolic_bp_mmhg": 90.1,
                "activity_level": 1
            }
        }
```

**Output Schema:**
```python
class PredictionResponse(BaseModel):
    prediction_id: str
    patient_id: int
    prediction: int  # 0 or 1
    probability: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    risk_level: str  # "LOWER RISK", "HIGHER RISK"
    timestamp: datetime
    explanation: Dict
```

---

## Step 2.4: Database Setup

### Database Schema

**Predictions Table:**
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    prediction_id VARCHAR(50) UNIQUE,
    patient_id INTEGER NOT NULL,
    prediction INTEGER,
    probability FLOAT,
    confidence VARCHAR(20),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    sensor_readings JSON,
    explanation JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patient_id ON predictions(patient_id);
CREATE INDEX idx_timestamp ON predictions(timestamp);
```

**Patients Table:**
```sql
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    patient_id INTEGER UNIQUE,
    name VARCHAR(100),
    age INTEGER,
    sex VARCHAR(10),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME ON UPDATE CURRENT_TIMESTAMP
);
```

### Database Operations
- Store all predictions with timestamps
- Enable historical analysis
- Track patient trends over time
- Query recent predictions for monitoring

---

# PHASE 3: DASHBOARD DEVELOPMENT

## Step 3.1: Dashboard Architecture

**Frontend Stack:**
- React.js (UI framework)
- TailwindCSS (styling)
- Chart.js or Recharts (visualization)
- Axios (API calls)

**Dashboard Structure:**
```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Header.jsx           # Navigation & title
│   │   ├── PatientSearch.jsx    # Patient lookup
│   │   ├── MetricsPanel.jsx     # Current vital signs
│   │   ├── PredictionCard.jsx   # Risk prediction display
│   │   ├── ExplanationPanel.jsx # Feature importance explanation
│   │   ├── HistoryChart.jsx     # Risk trend over time
│   │   └── AlertNotification.jsx # Risk alerts
│   ├── pages/
│   │   ├── Dashboard.jsx        # Main dashboard
│   │   ├── PatientDetail.jsx    # Individual patient view
│   │   └── Analytics.jsx        # System analytics
│   ├── services/
│   │   └── api.js               # API calls
│   ├── App.jsx
│   └── index.js
└── package.json
```

---

## Step 3.2: Dashboard Screens

### Screen 1: Real-Time Monitoring Dashboard
**Components:**
1. **Header**
   - Page title: "IoT Heart Disease Prediction"
   - Current time
   - Login info

2. **Patient Search Bar**
   - Search by patient ID
   - Auto-suggest from recent patients
   - Display current patient info

3. **Vital Signs Panel**
   - Heart Rate: 85 BPM (visual gauge)
   - SpO2: 97% (color-coded: green=good, yellow=ok, red=low)
   - Temperature: 36.8°C (visual indicator)
   - Blood Pressure: 140/90 mm Hg (status: normal/elevated/high)
   - Activity Level: Light

4. **Prediction Panel**
   - Large prediction indicator (HIGH RISK / LOW RISK)
   - Probability: 72%
   - Confidence level: HIGH
   - Time since last reading: 5 mins ago
   - Action button: "View Details"

5. **Explanation Panel**
   - "Why is this patient HIGH RISK?"
   - Top 3 contributing factors:
     - Systolic BP: 140 → HIGH (contributes +0.38)
     - Heart Rate: 85 → MODERATE (contributes +0.22)
     - Age: 65 → MODERATE (contributes +0.15)
   - Visual: Horizontal bar chart

6. **Risk Trend Chart**
   - X-axis: Time (last 24 hours)
   - Y-axis: Risk probability (0-100%)
   - Line chart showing prediction history
   - Color: Red = high risk, Green = low risk

7. **Alerts Section**
   - Recent alerts for this patient
   - Alert type: Risk increase, Abnormal reading, etc.
   - Timestamp and status

### Screen 2: Patient Detail View
- Full sensor reading history (last 30 days)
- Detailed statistics (mean, median, std dev)
- Correlation matrix of features
- Individual prediction explanations
- Download prediction history

### Screen 3: System Analytics
- Total predictions today
- Average prediction confidence
- Model performance metrics
- System uptime
- Active patients

---

## Step 3.3: Dashboard Data Flow

```
1. User opens dashboard
   ↓
2. Frontend sends: GET /api/v1/patients/{patient_id}/history
   ↓
3. Backend queries database for recent predictions
   ↓
4. Backend returns: List of predictions with timestamps
   ↓
5. Frontend displays:
   - Latest prediction prominently
   - Historical trend chart
   - Feature importance breakdown
   ↓
6. User clicks "Get New Prediction"
   ↓
7. Frontend sends: POST /api/v1/predict (with sensor readings)
   ↓
8. Backend:
   - Preprocesses input
   - Runs model inference
   - Generates SHAP explanation
   - Stores in database
   - Returns prediction + explanation
   ↓
9. Frontend updates display in real-time
   ↓
10. Dashboard refreshes automatically every 30 seconds
```

---

## Step 3.4: Real-Time Updates (WebSocket)

**Optional Enhancement:**
- Use WebSocket for real-time updates
- Endpoint: `ws://api/v1/realtime/patient/{patient_id}`
- Live streaming of predictions
- Instant alerts on risk changes

---

# PHASE 4: IOT INTEGRATION

## Step 4.1: IoT Device Setup (ESP32)

**Hardware Components:**
- ESP32 microcontroller
- MAX30102 sensor (heart rate + SpO2)
- Temperature sensor (MLX90614 or DHT22)
- Blood pressure cuff (Bluetooth/UART)
- Power supply

**NOT NEEDED:** ECG sensor (removed from project)

---

## Step 4.2: Data Collection Flow

```
Sensors → ESP32 → JSON Format → MQTT/HTTP → FastAPI Server → Dashboard
```

**Step 1: Read Sensor Data**
```cpp
// Pseudocode - actual C++ implementation needed
void readSensors() {
  heartRate = max30102.getHeartRate();      // 40-150 BPM
  spo2 = max30102.getSpO2();                // 90-100%
  temperature = mlx90614.getTemp();         // 35-39°C
  systolic, diastolic = bp_cuff.getReading(); // mm Hg
  activityLevel = getActivityFromAccel();   // 0-3
}
```

**Step 2: Package Data as JSON**
```json
{
  "device_id": "ESP32_001",
  "patient_id": 123,
  "age": 65,
  "sex": 1,
  "readings": {
    "heart_rate_bpm": 85.5,
    "spo2_percent": 97.2,
    "body_temperature_c": 36.8,
    "systolic_bp_mmhg": 140.2,
    "diastolic_bp_mmhg": 90.1,
    "activity_level": 1
  },
  "timestamp": "2024-01-31T14:30:00Z"
}
```

**Step 3: Send to Server**
- **Option A (MQTT):** 
  - Topic: `iot/patients/{patient_id}/readings`
  - QoS: 1 (at least once delivery)
  
- **Option B (HTTP):**
  - POST to `/api/v1/predict`
  - More direct, easier for development

**Step 4: Receive Prediction**
```json
{
  "prediction": 1,
  "probability": 0.72,
  "recommendation": "Schedule cardiology appointment"
}
```

**Step 5: Display on Dashboard**
- Prediction result shown in real-time
- Alert nurse/doctor if HIGH RISK
- Store in database for history

---

## Step 4.3: Error Handling & Reliability

**Sensor Failures:**
- If sensor disconnected: Use last valid reading
- If all sensors fail: Alert system, don't make prediction
- Retry mechanism: Try reading 3 times before giving up

**Network Issues:**
- If can't reach server: Store locally on ESP32
- Sync when connection restored
- Queue system for offline readings

**Data Validation:**
- Check ranges (HR: 40-150, SpO2: 85-100, Temp: 35-40)
- Discard unrealistic readings
- Log anomalies for investigation

---

# PHASE 5: TESTING & VALIDATION

## Step 5.1: Unit Testing

**Test preprocessing:**
```
Input: Raw sensor data with missing values
Expected: Cleaned, scaled data ready for model
Verify: No data leakage, correct scaling applied
```

**Test prediction:**
```
Input: Known sensor reading from test set
Expected: Prediction matches model output
Verify: Deterministic (same input → same output)
```

**Test SHAP explanation:**
```
Input: Prediction with features
Expected: Explanation sums to base value + prediction
Verify: Mathematically correct SHAP values
```

**Test API endpoints:**
```
Input: Valid JSON request
Expected: 200 response with prediction
Verify: Response schema matches documentation
```

---

## Step 5.2: Integration Testing

**Test end-to-end flow:**
```
1. Send sensor reading via API
2. Verify preprocessing correct
3. Verify model inference
4. Verify database storage
5. Verify dashboard update
6. Verify explanation generated
```

**Test data flow with mock IoT device:**
```
1. Mock ESP32 sends JSON
2. Server receives and processes
3. Prediction stored in database
4. Dashboard shows latest prediction
5. Historical chart updates
```

---

## Step 5.3: Performance Testing

**Load Testing:**
- Can server handle 100 predictions/second?
- Database query time < 100ms?
- API response time < 500ms?

**Model Performance:**
- Cross-validation AUC > 0.85?
- Test set AUC > 0.82?
- Prediction time < 50ms?

**Dashboard Performance:**
- Page load time < 3 seconds?
- Chart rendering < 1 second?
- Real-time updates < 500ms delay?

---

## Step 5.4: Validation Against Real-World Requirements

**Clinical Validation:**
- Does model performance meet healthcare standards?
- Are explanations clinically meaningful?
- Are false negatives (missed at-risk) minimized?

**System Reliability:**
- 99.9% uptime requirement?
- Data accuracy > 99%?
- Recovery from failures < 5 minutes?

**User Acceptance Testing:**
- Can healthcare staff use the system?
- Are explanations understandable?
- Does dashboard show all needed information?

---

# PHASE 6: DEPLOYMENT

## Step 6.1: Pre-Deployment Checklist

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance targets met
- [ ] Security review completed
- [ ] Database backups configured
- [ ] Monitoring alerts setup
- [ ] Documentation complete
- [ ] Team training completed

---

## Step 6.2: Deployment Architecture

```
Production Environment:

IoT Devices (ESP32) → Load Balancer → FastAPI Server (Docker) × 3
                                              ↓
                                      PostgreSQL Database
                                      (with backup)
                                              ↓
                                      Redis Cache
                                      (session management)
                                              ↓
                                      React Frontend (CDN)
                                              ↓
                                      Monitoring/Logging
```

---

## Step 6.3: Containerization (Docker)

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-slim AS build
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:latest
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## Step 6.4: Deployment Steps

**Step 1: Prepare Production Server**
- Set up Ubuntu/Linux server
- Install Docker & Docker Compose
- Configure SSL certificates

**Step 2: Deploy with Docker Compose**
```yaml
version: '3.8'
services:
  backend:
    image: heart-disease-backend:1.0
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
    depends_on:
      - postgres
  
  frontend:
    image: heart-disease-frontend:1.0
    ports:
      - "80:80"
  
  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=secure_password
```

**Step 3: Run Migrations**
- Create database tables
- Load initial data
- Test connections

**Step 4: Start Services**
```bash
docker-compose up -d
```

**Step 5: Verify Deployment**
- Test health endpoint: GET /api/v1/health
- Test prediction endpoint
- Check dashboard loads
- Verify database connectivity

---

## Step 6.5: Post-Deployment Monitoring

**Set up monitoring for:**
- API response times
- Error rates
- Database query times
- CPU/Memory usage
- Disk space
- Prediction confidence trends

**Alerting:**
- Alert on API errors > 1%
- Alert on response time > 1 second
- Alert on database down
- Alert on high-risk prediction spike

**Logging:**
- Log all predictions
- Log all errors with stack traces
- Log API access
- Log model updates

---

## Step 6.6: Maintenance & Updates

**Regular Tasks:**
- Daily: Check system health
- Weekly: Review error logs
- Monthly: Analyze prediction accuracy
- Quarterly: Retrain model with new data
- Annually: Security audit

**Model Updates:**
- Collect new real-world data
- Retrain model with updated data
- Validate new model performance
- Blue-green deployment (old & new side-by-side)
- Gradual rollout to users

---

# IMPLEMENTATION TIMELINE

## Week 1: ML Pipeline
- Day 1-2: Data preprocessing
- Day 3-4: Model training & evaluation
- Day 5: SHAP explainability

## Week 2: FastAPI Server
- Day 1-2: API design & schemas
- Day 3-4: Database setup
- Day 5: Endpoint implementation & testing

## Week 3: Dashboard
- Day 1-2: Frontend setup & layout
- Day 3-4: Chart components & real-time updates
- Day 5: Integration with backend

## Week 4: IoT & Deployment
- Day 1-2: IoT integration
- Day 3-4: Testing & validation
- Day 5: Deployment & monitoring setup

---

# TECHNOLOGY STACK SUMMARY

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Backend | scikit-learn, XGBoost, CatBoost | Model training |
| Explainability | SHAP | Prediction explanation |
| API Server | FastAPI, Uvicorn | REST API endpoints |
| Database | PostgreSQL | Data persistence |
| Cache | Redis | Session management |
| Frontend | React, TailwindCSS, Recharts | Web dashboard |
| IoT Device | ESP32 + Sensors | Data collection |
| Communication | MQTT/HTTP | Device to server |
| Containerization | Docker, Docker Compose | Deployment |
| Monitoring | Prometheus, Grafana | System health |
| Logging | ELK Stack | Log aggregation |

---

# SUCCESS CRITERIA

✅ **Completed when:**
1. Model achieves > 85% ROC-AUC on test set
2. SHAP explanations are clinically meaningful
3. API responds < 500ms for predictions
4. Dashboard displays all required information
5. IoT device sends readings reliably
6. Database stores and retrieves data correctly
7. All unit & integration tests pass
8. System deployed and running 24/7
9. Monitoring & alerts configured
10. Documentation complete and team trained

---

**Status:** Ready to implement
**Next Action:** Begin Phase 1, Step 1.1 (Data Preprocessing)

