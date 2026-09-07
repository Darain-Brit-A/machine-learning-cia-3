import unittest

from fastapi.testclient import TestClient

from server.main import app


class ApiPhaseTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertIn("model_loaded", payload)
        self.assertIn("database_connected", payload)

    def test_prediction_endpoint(self):
        payload = {
            "device_id": "phase5_test",
            "timestamp": "2026-08-31T12:00:00",
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
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["device_id"], "phase5_test")
        self.assertIn("prediction", body)
        self.assertIn("risk_score", body)
        self.assertIn("confidence", body)
        self.assertIn("explanation", body)


if __name__ == "__main__":
    unittest.main()
