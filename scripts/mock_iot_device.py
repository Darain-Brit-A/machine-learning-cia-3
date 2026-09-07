#!/usr/bin/env python3
"""Mock IoT sensor client for Phase 4 integration testing.

This script mimics a simple ESP32-style device that sends a JSON payload to the
FastAPI prediction endpoint. It is intended to validate the device-to-server
flow described in the implementation guide.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import requests


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_DEVICE_ID = "ESP32_001"


def build_payload(device_id: str) -> Dict[str, Any]:
    """Create a realistic sensor reading payload matching the API contract."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "device_id": device_id,
        "timestamp": now,
        "heart_rate_bpm": 85.5,
        "spo2_percent": 97.2,
        "body_temperature_c": 36.8,
        "systolic_bp_mmhg": 140.2,
        "diastolic_bp_mmhg": 90.1,
        "ecg_hr_bpm": 86.0,
        "ecg_rr_interval_ms": 700.0,
        "ecg_rmssd_ms": 45.0,
        "ecg_sdnn_ms": 60.0,
        "ecg_signal_quality": 80.0,
        "activity_level": 1,
    }


def send_payload(api_url: str, device_id: str) -> Dict[str, Any]:
    """Send the payload and return the server JSON response."""
    payload = build_payload(device_id)
    response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> int:
    api_url = os.getenv("API_URL", DEFAULT_API_URL)
    device_id = os.getenv("DEVICE_ID", DEFAULT_DEVICE_ID)

    print(f"Sending mock IoT payload to {api_url} for device {device_id}...")
    try:
        result = send_payload(api_url, device_id)
    except requests.RequestException as exc:  # pragma: no cover - CLI error path
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print("Prediction response:")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
