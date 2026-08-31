"""
Configuration and Setup File
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
SERVER_DIR = PROJECT_ROOT / 'server'
DASHBOARD_DIR = PROJECT_ROOT / 'dashboard'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, SCRIPTS_DIR, NOTEBOOKS_DIR, SERVER_DIR, DASHBOARD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FILE PATHS
# ============================================================================

DATASET_CSV = DATA_DIR / 'heart_iot_synthetic.csv'
DATASET_PARQUET = DATA_DIR / 'heart_iot_synthetic.parquet'
DATA_DICTIONARY = DATA_DIR / 'data_dictionary.csv'
DATASET_SUMMARY = DATA_DIR / 'dataset_summary.json'
DATABASE = DATA_DIR / 'predictions.db'

MODEL_PATH = MODELS_DIR / 'random_forest.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.json'
MODEL_COMPARISON_PATH = MODELS_DIR / 'model_comparison.csv'

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_RELOAD = os.getenv('API_RELOAD', 'True').lower() == 'true'
API_WORKERS = int(os.getenv('API_WORKERS', 1))

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', 'localhost')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8501))
API_URL = os.getenv('API_URL', f'http://{API_HOST}:{API_PORT}')

# ============================================================================
# ML CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TEST_SIZE = 0.2
TRAIN_BATCH_SIZE = 32
PREDICTION_THRESHOLD = 0.5

# Model parameters
MODELS_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
        'random_state': RANDOM_SEED,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    },
    'catboost': {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_SEED,
        'verbose': 0
    }
}

# ============================================================================
# ECG CONFIGURATION
# ============================================================================

ECG_SAMPLING_RATE = 200  # Hz
ECG_LOWCUT = 0.5  # Hz
ECG_HIGHCUT = 50  # Hz
ECG_FILTER_ORDER = 4

# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

ALERT_THRESHOLDS = {
    'high_risk_score': 0.7,
    'heart_rate_min': 50,
    'heart_rate_max': 120,
    'spo2_min': 94,
    'systolic_bp_min': 90,
    'systolic_bp_max': 180,
    'ecg_signal_quality_min': 50
}

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_TYPE = 'sqlite'  # sqlite, postgresql, mysql
DB_ECHO = False  # SQL logging

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_NAMES = [
    'age', 'sex', 'heart_rate_bpm', 'spo2_percent', 'body_temperature_c',
    'systolic_bp_mmhg', 'diastolic_bp_mmhg', 'ecg_hr_bpm', 'ecg_rr_interval_ms',
    'ecg_rmssd_ms', 'ecg_sdnn_ms', 'ecg_signal_quality', 'activity_level'
]

VITAL_SIGNS = [
    'heart_rate_bpm', 'spo2_percent', 'body_temperature_c',
    'systolic_bp_mmhg', 'diastolic_bp_mmhg'
]

ECG_FEATURES = [
    'ecg_hr_bpm', 'ecg_rr_interval_ms', 'ecg_rmssd_ms', 'ecg_sdnn_ms',
    'ecg_signal_quality'
]

# ============================================================================
# VALIDATION RANGES
# ============================================================================

VALID_RANGES = {
    'age': (18, 120),
    'heart_rate_bpm': (20, 200),
    'spo2_percent': (70, 100),
    'body_temperature_c': (32, 42),
    'systolic_bp_mmhg': (60, 250),
    'diastolic_bp_mmhg': (30, 150),
    'ecg_hr_bpm': (20, 200),
    'ecg_rr_interval_ms': (200, 2000),
    'ecg_rmssd_ms': (0, 300),
    'ecg_sdnn_ms': (0, 400),
    'ecg_signal_quality': (0, 100),
    'activity_level': (0, 3)
}

# ============================================================================
# OPTIMAL RANGES (for display)
# ============================================================================

OPTIMAL_RANGES = {
    'heart_rate_bpm': (60, 100),
    'spo2_percent': (95, 100),
    'body_temperature_c': (36.5, 37.5),
    'systolic_bp_mmhg': (90, 120),
    'diastolic_bp_mmhg': (60, 80)
}
