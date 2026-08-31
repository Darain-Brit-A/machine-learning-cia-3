"""
FastAPI Backend Server

This module implements:
- REST API endpoints for sensor data ingestion
- Real-time feature preprocessing
- ML model inference
- SHAP explainability
- Database storage
- Historical data retrieval
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IoT Heart Disease Prediction API",
    description="Real-time cardiovascular health monitoring system",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
SCALER = None
SHAP_EXPLAINER = None
FEATURE_NAMES = None
DB_PATH = 'data/predictions.db'


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SensorReading(BaseModel):
    """Sensor data from ESP32."""
    device_id: str = Field(..., description="Device/patient ID")
    timestamp: datetime = Field(..., description="Reading timestamp")
    heart_rate_bpm: float = Field(..., description="Heart rate (BPM)")
    spo2_percent: float = Field(..., description="Blood oxygen saturation (%)")
    body_temperature_c: float = Field(..., description="Body temperature (°C)")
    systolic_bp_mmhg: float = Field(..., description="Systolic BP (mm Hg)")
    diastolic_bp_mmhg: float = Field(..., description="Diastolic BP (mm Hg)")
    ecg_hr_bpm: Optional[float] = None
    ecg_rr_interval_ms: Optional[float] = None
    ecg_rmssd_ms: Optional[float] = None
    ecg_sdnn_ms: Optional[float] = None
    ecg_signal_quality: Optional[float] = None
    activity_level: Optional[int] = 0
    

class PredictionResponse(BaseModel):
    """API response with prediction and explanation."""
    device_id: str
    timestamp: datetime
    input_features: Dict[str, float]
    prediction: int
    risk_score: float
    confidence: float
    explanation: Dict
    alert: bool
    message: str


class HealthStatus(BaseModel):
    """System health status."""
    status: str
    model_loaded: bool
    database_connected: bool
    uptime_seconds: float


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

@contextmanager
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def initialize_database():
    """Initialize database tables."""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                heart_rate_bpm REAL,
                spo2_percent REAL,
                body_temperature_c REAL,
                systolic_bp_mmhg REAL,
                diastolic_bp_mmhg REAL,
                ecg_hr_bpm REAL,
                ecg_rr_interval_ms REAL,
                ecg_rmssd_ms REAL,
                ecg_sdnn_ms REAL,
                ecg_signal_quality REAL,
                activity_level INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                reading_id INTEGER,
                prediction INTEGER,
                risk_score REAL,
                confidence REAL,
                explanation TEXT,
                alert BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (reading_id) REFERENCES readings(id)
            )
        ''')
        
        conn.commit()
    
    logger.info("Database initialized")


def save_reading(reading: SensorReading) -> int:
    """Save sensor reading to database."""
    with get_db_connection() as conn:
        cursor = conn.execute('''
            INSERT INTO readings (
                device_id, timestamp, heart_rate_bpm, spo2_percent,
                body_temperature_c, systolic_bp_mmhg, diastolic_bp_mmhg,
                ecg_hr_bpm, ecg_rr_interval_ms, ecg_rmssd_ms, ecg_sdnn_ms,
                ecg_signal_quality, activity_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reading.device_id,
            reading.timestamp.isoformat(),
            reading.heart_rate_bpm,
            reading.spo2_percent,
            reading.body_temperature_c,
            reading.systolic_bp_mmhg,
            reading.diastolic_bp_mmhg,
            reading.ecg_hr_bpm,
            reading.ecg_rr_interval_ms,
            reading.ecg_rmssd_ms,
            reading.ecg_sdnn_ms,
            reading.ecg_signal_quality,
            reading.activity_level
        ))
        conn.commit()
        return cursor.lastrowid


def save_prediction(
    device_id: str,
    timestamp: datetime,
    reading_id: int,
    prediction: int,
    risk_score: float,
    confidence: float,
    explanation: dict,
    alert: bool
):
    """Save prediction to database."""
    with get_db_connection() as conn:
        conn.execute('''
            INSERT INTO predictions (
                device_id, timestamp, reading_id, prediction,
                risk_score, confidence, explanation, alert
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id,
            timestamp.isoformat(),
            reading_id,
            prediction,
            risk_score,
            confidence,
            json.dumps(explanation),
            alert
        ))
        conn.commit()


# ============================================================================
# ML FUNCTIONS
# ============================================================================

def load_model():
    """Load trained model, scaler, and feature names."""
    global MODEL, SCALER, FEATURE_NAMES, SHAP_EXPLAINER
    
    try:
        model_path = Path('models') / 'random_forest.pkl'
        scaler_path = Path('models') / 'scaler.pkl'
        features_path = Path('models') / 'feature_names.json'
        
        if not model_path.exists():
            raise FileNotFoundError("Model files not found. Please train models first.")
        
        MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            FEATURE_NAMES = json.load(f)
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def prepare_features(reading: SensorReading) -> np.ndarray:
    """
    Convert sensor reading to feature array.
    Handle missing ECG features with defaults.
    """
    features = {
        'age': 60,  # From dataset generation, average
        'sex': 0,
        'heart_rate_bpm': reading.heart_rate_bpm,
        'spo2_percent': reading.spo2_percent,
        'body_temperature_c': reading.body_temperature_c,
        'systolic_bp_mmhg': reading.systolic_bp_mmhg,
        'diastolic_bp_mmhg': reading.diastolic_bp_mmhg,
        'ecg_hr_bpm': reading.ecg_hr_bpm or reading.heart_rate_bpm,
        'ecg_rr_interval_ms': reading.ecg_rr_interval_ms or (60000 / reading.heart_rate_bpm),
        'ecg_rmssd_ms': reading.ecg_rmssd_ms or 50.0,
        'ecg_sdnn_ms': reading.ecg_sdnn_ms or 70.0,
        'ecg_signal_quality': reading.ecg_signal_quality or 80.0,
        'activity_level': reading.activity_level or 0
    }
    
    # Create array in feature order
    feature_array = np.array([[features[f] for f in FEATURE_NAMES]])
    
    return feature_array


def make_prediction(feature_array: np.ndarray) -> tuple:
    """
    Make prediction with model and generate explanation.
    
    Returns:
    --------
    prediction : int (0 or 1)
    risk_score : float (0-1)
    confidence : float (0-1)
    explanation : dict
    """
    # Scale features
    scaled_features = SCALER.transform(feature_array)
    
    # Prediction
    prediction = MODEL.predict(scaled_features)[0]
    risk_proba = MODEL.predict_proba(scaled_features)[0]
    risk_score = float(risk_proba[1])  # Probability of class 1 (higher risk)
    confidence = float(max(risk_proba))
    
    # Basic feature importance (using model's feature_importances if available)
    explanation = {
        'model_type': 'Random Forest',
        'prediction_class': 'Higher Risk' if prediction == 1 else 'Lower Risk',
        'risk_score': risk_score,
        'confidence': confidence
    }
    
    # Add SHAP explanation if available
    if SHAP_EXPLAINER is not None:
        try:
            shap_explanation = SHAP_EXPLAINER.explain_prediction(scaled_features[0], return_dict=True)
            explanation['shap_explanation'] = shap_explanation
        except:
            pass
    
    # Add feature importance from model
    if hasattr(MODEL, 'feature_importances_'):
        importances = MODEL.feature_importances_
        feature_importance = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )
        explanation['top_features'] = [
            {'feature': f, 'importance': float(imp)} 
            for f, imp in feature_importance[:5]
        ]
    
    return prediction, risk_score, confidence, explanation


def check_alert_conditions(reading: SensorReading, risk_score: float, prediction: int) -> bool:
    """
    Check if alert conditions are met.
    
    Alert conditions:
    - High risk prediction (>0.7)
    - Abnormal vital signs
    - Concerning ECG signal quality
    """
    alert = False
    
    if risk_score > 0.7:
        alert = True
    
    if reading.heart_rate_bpm > 120 or reading.heart_rate_bpm < 50:
        alert = True
    
    if reading.spo2_percent < 94:
        alert = True
    
    if reading.systolic_bp_mmhg > 180 or reading.systolic_bp_mmhg < 90:
        alert = True
    
    if reading.ecg_signal_quality and reading.ecg_signal_quality < 50:
        alert = True
    
    return alert


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    initialize_database()
    if not load_model():
        logger.warning("Model not loaded - API will not make predictions")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "IoT Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"], response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy",
        model_loaded=MODEL is not None,
        database_connected=Path(DB_PATH).exists(),
        uptime_seconds=0
    )


@app.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict(reading: SensorReading):
    """
    Make prediction from sensor reading.
    
    Receives sensor data, preprocesses it, runs ML model,
    generates SHAP explanation, and stores results.
    """
    
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train models first."
        )
    
    try:
        # Save reading to database
        reading_id = save_reading(reading)
        
        # Prepare features
        feature_array = prepare_features(reading)
        
        # Make prediction
        prediction, risk_score, confidence, explanation = make_prediction(feature_array)
        
        # Check alert conditions
        alert = check_alert_conditions(reading, risk_score, prediction)
        
        # Generate message
        if prediction == 1:
            message = f"⚠️ Higher Risk (Score: {risk_score:.2%})" if alert else f"Higher Risk (Score: {risk_score:.2%})"
        else:
            message = f"✓ Lower Risk (Score: {risk_score:.2%})"
        
        # Save prediction to database
        save_prediction(
            reading.device_id,
            reading.timestamp,
            reading_id,
            prediction,
            risk_score,
            confidence,
            explanation,
            alert
        )
        
        # Return response
        return PredictionResponse(
            device_id=reading.device_id,
            timestamp=reading.timestamp,
            input_features={f: float(feature_array[0][i]) for i, f in enumerate(FEATURE_NAMES)},
            prediction=prediction,
            risk_score=risk_score,
            confidence=confidence,
            explanation=explanation,
            alert=alert,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/readings/{device_id}", tags=["Data"])
async def get_device_readings(device_id: str, limit: int = 100):
    """Get recent readings for a device."""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM readings
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (device_id, limit))
            
            readings = [dict(row) for row in cursor.fetchall()]
            return readings
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{device_id}", tags=["Data"])
async def get_device_predictions(device_id: str, limit: int = 50):
    """Get recent predictions for a device."""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM predictions
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (device_id, limit))
            
            predictions = []
            for row in cursor.fetchall():
                pred = dict(row)
                pred['explanation'] = json.loads(pred['explanation'])
                predictions.append(pred)
            
            return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest/{device_id}", tags=["Data"])
async def get_latest_prediction(device_id: str):
    """Get latest prediction and readings for device."""
    try:
        with get_db_connection() as conn:
            # Get latest reading
            cursor = conn.execute('''
                SELECT * FROM readings
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (device_id,))
            
            reading = dict(cursor.fetchone()) if cursor.fetchone() else None
            
            # Get latest prediction
            cursor = conn.execute('''
                SELECT * FROM predictions
                WHERE device_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (device_id,))
            
            prediction = dict(cursor.fetchone()) if cursor.fetchone() else None
            if prediction:
                prediction['explanation'] = json.loads(prediction['explanation'])
            
            return {'reading': reading, 'prediction': prediction}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/{device_id}", tags=["Data"])
async def get_device_statistics(device_id: str):
    """Get summary statistics for device."""
    try:
        with get_db_connection() as conn:
            # Count readings
            cursor = conn.execute(
                'SELECT COUNT(*) as count FROM readings WHERE device_id = ?',
                (device_id,)
            )
            reading_count = cursor.fetchone()['count']
            
            # Count high-risk predictions
            cursor = conn.execute('''
                SELECT COUNT(*) as count FROM predictions
                WHERE device_id = ? AND prediction = 1
            ''', (device_id,))
            high_risk_count = cursor.fetchone()['count']
            
            # Average metrics
            cursor = conn.execute('''
                SELECT 
                    AVG(heart_rate_bpm) as avg_hr,
                    AVG(spo2_percent) as avg_spo2,
                    AVG(body_temperature_c) as avg_temp
                FROM readings
                WHERE device_id = ?
            ''', (device_id,))
            
            stats = dict(cursor.fetchone())
            stats['reading_count'] = reading_count
            stats['high_risk_count'] = high_risk_count
            
            return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DEVELOPMENT/TESTING
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
