"""
Streamlit Dashboard for Real-Time Heart Disease Prediction

Interactive web interface showing:
- Live vital signs and ECG
- Risk predictions
- SHAP feature explanations
- Historical trends
- Device status and alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sqlite3

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="IoT Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-high { color: #d32f2f; font-weight: bold; }
    .alert-medium { color: #f57c00; font-weight: bold; }
    .alert-low { color: #388e3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG
# ============================================================================

API_URL = "http://localhost:8000"
DB_PATH = "data/predictions.db"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_latest_data(device_id):
    """Fetch latest reading and prediction from API."""
    try:
        response = requests.get(f"{API_URL}/latest/{device_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def fetch_device_predictions(device_id, limit=50):
    """Fetch prediction history."""
    try:
        response = requests.get(f"{API_URL}/predictions/{device_id}?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def fetch_device_stats(device_id):
    """Fetch device statistics."""
    try:
        response = requests.get(f"{API_URL}/stats/{device_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_risk_color(risk_score):
    """Get color based on risk level."""
    if risk_score > 0.7:
        return "#d32f2f"  # Red
    elif risk_score > 0.4:
        return "#f57c00"  # Orange
    else:
        return "#388e3c"  # Green


def get_risk_level(risk_score):
    """Get risk level text."""
    if risk_score > 0.7:
        return "⚠️ HIGH RISK"
    elif risk_score > 0.4:
        return "⚡ MODERATE RISK"
    else:
        return "✓ LOWER RISK"


def create_vital_gauge(value, min_val, max_val, optimal_min, optimal_max, title, unit):
    """Create gauge chart for vital sign."""
    
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        delta={'reference': (optimal_min + optimal_max) / 2},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': get_risk_color(0.5 if optimal_min <= value <= optimal_max else 0.8)},
            'steps': [
                {'range': [min_val, optimal_min], 'color': '#ffebee'},
                {'range': [optimal_min, optimal_max], 'color': '#e8f5e9'},
                {'range': [optimal_max, max_val], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': optimal_max
            }
        },
        suffix=unit,
        number={'suffix': unit}
    )])
    
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def create_ecg_plot():
    """Create synthetic ECG waveform."""
    t = np.linspace(0, 10, 1000)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.normal(0, 0.05, 1000)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=ecg,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#1976d2', width=2)
    ))
    
    fig.update_layout(
        title='ECG Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude (mV)',
        height=300,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_trend_chart(predictions):
    """Create risk score trend chart."""
    if not predictions:
        st.info("No historical data available")
        return None
    
    df = pd.DataFrame([
        {
            'timestamp': p['timestamp'],
            'risk_score': p['risk_score'],
            'prediction': p['prediction']
        }
        for p in predictions
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#d32f2f', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Alert Threshold", annotation_position="right")
    
    fig.update_layout(
        title='Risk Score Trend',
        xaxis_title='Time',
        yaxis_title='Risk Score (0-1)',
        height=300,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_feature_importance_chart(explanation):
    """Create feature importance chart from SHAP."""
    
    if 'shap_explanation' not in explanation:
        st.info("SHAP explanation not available")
        return None
    
    shap_exp = explanation['shap_explanation']
    
    features = [f['feature'] for f in shap_exp['all_features'][:10]]
    values = [f['shap_value'] for f in shap_exp['all_features'][:10]]
    
    colors = ['#d32f2f' if v > 0 else '#388e3c' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=[abs(v) for v in values],
            orientation='h',
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title='Top Feature Contributors (SHAP)',
        xaxis_title='Absolute SHAP Value',
        yaxis_title='Feature',
        height=300,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app."""
    
    st.title("❤️ IoT Heart Disease Prediction Dashboard")
    st.markdown("Real-time cardiovascular health monitoring with explainable AI")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "🏥 Live Monitor",
        "📊 Analytics & History",
        "🔍 Model Explanation",
        "ℹ️ System Info"
    ])
    
    # Device selection
    device_id = st.sidebar.text_input(
        "Device/Patient ID",
        value="patient_001",
        help="Enter device ID to monitor"
    )
    
    # Refresh rate
    refresh_interval = st.sidebar.slider(
        "Auto-refresh (seconds)",
        min_value=5,
        max_value=60,
        value=10
    )
    
    # ========================================================================
    # PAGE: LIVE MONITOR
    # ========================================================================
    
    if page == "🏥 Live Monitor":
        st.header("Live Vital Signs & Risk Assessment")
        
        # Fetch latest data
        latest_data = fetch_latest_data(device_id)
        
        if latest_data is None:
            st.warning(f"No data found for device: {device_id}")
            st.info("Please make sure the server is running and predictions have been made.")
            return
        
        reading = latest_data.get('reading')
        prediction = latest_data.get('prediction')
        
        if not reading or not prediction:
            st.warning("Incomplete data received")
            return
        
        # ====== ALERT BANNER ======
        if prediction.get('alert'):
            st.error(f"🚨 ALERT: {get_risk_level(prediction['risk_score'])}")
        else:
            st.success(f"✓ Status: {get_risk_level(prediction['risk_score'])}")
        
        # ====== TOP METRICS ======
        st.subheader("Current Vital Signs")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Heart Rate",
                f"{reading['heart_rate_bpm']:.0f}",
                "bpm",
                delta=None
            )
        
        with col2:
            st.metric(
                "SpO₂",
                f"{reading['spo2_percent']:.1f}",
                "%",
                delta=None
            )
        
        with col3:
            st.metric(
                "Temperature",
                f"{reading['body_temperature_c']:.1f}",
                "°C",
                delta=None
            )
        
        with col4:
            st.metric(
                "Risk Score",
                f"{prediction['risk_score']:.1%}",
                delta=None
            )
        
        # ====== VITAL GAUGES ======
        st.subheader("Vital Sign Gauges")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = create_vital_gauge(
                reading['heart_rate_bpm'], 40, 150, 60, 100,
                'Heart Rate', ' bpm'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_vital_gauge(
                reading['spo2_percent'], 90, 100, 95, 100,
                'SpO₂', ' %'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = create_vital_gauge(
                reading['body_temperature_c'], 35, 39, 36.5, 37.5,
                'Temperature', ' °C'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ====== ECG & BLOOD PRESSURE ======
        st.subheader("ECG & Blood Pressure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_ecg_plot()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Systolic BP", f"{reading['systolic_bp_mmhg']:.0f}", "mm Hg")
            st.metric("Diastolic BP", f"{reading['diastolic_bp_mmhg']:.0f}", "mm Hg")
            
            if reading.get('ecg_signal_quality'):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("ECG Signal Quality", f"{reading['ecg_signal_quality']:.0f}", "%")
                with col_b:
                    st.metric("Activity Level", reading.get('activity_level', 0))
        
        # ====== PREDICTION DETAILS ======
        st.subheader("Risk Assessment Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediction", "Higher Risk" if prediction['prediction'] == 1 else "Lower Risk")
            st.metric("Confidence", f"{prediction['confidence']:.1%}")
        
        with col2:
            st.metric("Timestamp", reading['timestamp'])
            st.metric("Device ID", device_id)
        
        # ====== MESSAGES ======
        st.info(prediction.get('message', 'No message available'))
    
    # ========================================================================
    # PAGE: ANALYTICS & HISTORY
    # ========================================================================
    
    elif page == "📊 Analytics & History":
        st.header("Analytics & Historical Trends")
        
        # Get statistics
        stats = fetch_device_stats(device_id)
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Readings", stats.get('reading_count', 0))
            with col2:
                st.metric("High Risk Alerts", stats.get('high_risk_count', 0))
            with col3:
                st.metric("Avg Heart Rate", f"{stats.get('avg_hr', 0):.0f} bpm")
            with col4:
                st.metric("Avg SpO₂", f"{stats.get('avg_spo2', 0):.1f} %")
        
        # Risk score trend
        st.subheader("Risk Score Trend")
        predictions = fetch_device_predictions(device_id, limit=100)
        
        if predictions:
            fig = create_trend_chart(predictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available")
        
        # Predictions table
        st.subheader("Recent Predictions")
        if predictions:
            df_preds = pd.DataFrame([
                {
                    'Timestamp': p['timestamp'],
                    'Risk Score': f"{p['risk_score']:.1%}",
                    'Prediction': 'High Risk' if p['prediction'] == 1 else 'Low Risk',
                    'Confidence': f"{p['confidence']:.1%}",
                    'Alert': '🚨' if p['alert'] else '✓'
                }
                for p in predictions[:20]
            ])
            st.dataframe(df_preds, use_container_width=True)
    
    # ========================================================================
    # PAGE: MODEL EXPLANATION
    # ========================================================================
    
    elif page == "🔍 Model Explanation":
        st.header("Model Prediction Explanation (SHAP)")
        st.markdown("Understanding which features contributed to the risk prediction")
        
        latest_data = fetch_latest_data(device_id)
        
        if latest_data is None:
            st.warning(f"No data found for device: {device_id}")
            return
        
        prediction = latest_data.get('prediction')
        
        if not prediction:
            st.warning("Prediction data not available")
            return
        
        explanation = prediction.get('explanation', {})
        
        # Display prediction
        st.subheader("Current Prediction")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Class", explanation.get('prediction_class', 'Unknown'))
        with col2:
            st.metric("Risk Score", f"{explanation.get('risk_score', 0):.1%}")
        with col3:
            st.metric("Confidence", f"{explanation.get('confidence', 0):.1%}")
        
        # SHAP explanation
        st.subheader("Feature Contributions (SHAP)")
        
        if 'shap_explanation' in explanation:
            shap_exp = explanation['shap_explanation']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Positive Contributors** (increase risk)")
                for contrib in shap_exp.get('top_positive_contributors', []):
                    st.write(f"- {contrib['feature']}: {contrib['shap_value']:.4f}")
            
            with col2:
                st.markdown("**Top Negative Contributors** (decrease risk)")
                for contrib in shap_exp.get('top_negative_contributors', []):
                    st.write(f"- {contrib['feature']}: {contrib['shap_value']:.4f}")
        
        # Feature importance chart
        st.subheader("Feature Importance Visualization")
        fig = create_feature_importance_chart(explanation)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.subheader("Model Information")
        st.write(f"**Model Type:** {explanation.get('model_type', 'Unknown')}")
        
        if 'top_features' in explanation:
            st.markdown("**Top Features (by importance):**")
            for feat in explanation['top_features']:
                st.write(f"- {feat['feature']}: {feat['importance']:.4f}")
    
    # ========================================================================
    # PAGE: SYSTEM INFO
    # ========================================================================
    
    elif page == "ℹ️ System Info":
        st.header("System Information")
        
        # Check API health
        st.subheader("API Status")
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Status", "🟢 Healthy" if health['status'] == 'healthy' else "🔴 Unhealthy")
                with col2:
                    st.metric("Model", "✓ Loaded" if health['model_loaded'] else "✗ Not Loaded")
                with col3:
                    st.metric("Database", "✓ Connected" if health['database_connected'] else "✗ Not Connected")
            else:
                st.error("API responded with error")
        except:
            st.error("Cannot connect to API. Make sure the server is running on http://localhost:8000")
        
        # System documentation
        st.subheader("About This System")
        st.markdown("""
        ### IoT-Based Heart Disease Prediction System
        
        **Components:**
        - Wearable sensors (MAX30102, AD8232, temperature sensor)
        - ESP32 microcontroller for data collection
        - FastAPI backend for ML inference
        - Machine Learning models (Random Forest, XGBoost, etc.)
        - SHAP for model explainability
        - Streamlit dashboard for visualization
        
        **Features:**
        - Real-time vital sign monitoring
        - Cardiovascular risk assessment
        - Explainable AI predictions
        - Historical trend analysis
        - Automated alerts
        
        **⚠️ DISCLAIMER:**
        This is a research prototype and **NOT** a medical device.
        - Do NOT use for clinical diagnosis
        - Consult healthcare professionals for medical decisions
        - Data is for demonstration purposes only
        """)
        
        # Links
        st.subheader("Resources")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("[📖 API Documentation](http://localhost:8000/docs)")
            st.markdown("[📊 Data Files](./data/)")
        
        with col2:
            st.markdown("[🤖 Model Info](./models/)")
            st.markdown("[💾 Database](./data/predictions.db)")


if __name__ == "__main__":
    main()
