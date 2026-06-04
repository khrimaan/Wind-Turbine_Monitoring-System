import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from scipy import stats
from keras.models import load_model
import warnings
import joblib
import pandas as pd
import pickle
import os

from anomaly_detector import AnomalyDetector
warnings.filterwarnings('ignore')


def realistic_power_curve(wind_speed, rated_power=2000, cut_in=3.0, rated_speed=12.0, cut_out=25.0):
    """Realistic cubic wind turbine power curve with cut-in, rated, and cut-out regions.

    Region 1 (v < cut_in):   P = 0  (below cut-in, rotor stalled)
    Region 2 (cut_in <= v <= rated): P = P_rated * ((v - v_ci) / (v_r - v_ci))^3
    Region 3 (rated < v <= cut_out): P = P_rated  (pitch-regulated constant output)
    Region 4 (v > cut_out):  P = 0  (emergency shutdown)
    """
    v = np.asarray(wind_speed, dtype=float)
    scalar = v.ndim == 0
    v = np.atleast_1d(v)
    power = np.zeros_like(v)

    cubic = (v >= cut_in) & (v <= rated_speed)
    power[cubic] = rated_power * ((v[cubic] - cut_in) / (rated_speed - cut_in)) ** 3

    rated_mask = (v > rated_speed) & (v <= cut_out)
    power[rated_mask] = rated_power

    return float(power[0]) if scalar else power


def convert_to_native_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


class TurbineMonitor:
    def __init__(self):
        self.historical_data = []
        self.prediction_errors = []
        self.max_history = 1000
        self.anomaly_detector = AnomalyDetector()

        try:
            possible_paths = [
                'model_files/wind_power_predictor.keras',
                './model_files/wind_power_predictor.keras',
                'wind_power_predictor.keras'
            ]

            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path:
                self.model = load_model(model_path)
                self.scaler = joblib.load('model_files/feature_scaler.pkl')
                with open('model_files/feature_columns.txt', 'r') as f:
                    self.feature_columns = [line.strip() for line in f.readlines()]
                print("ML models loaded successfully")
            else:
                print("Model files not found, using simulation mode")
                self.model = None
                self.scaler = None
                self.feature_columns = []

        except Exception as e:
            print(f"Error loading ML models: {e}")
            self.model = None
            self.scaler = None
            self.feature_columns = []

    def create_features(self, wind_speed, wind_direction, theoretical_power, actual_power=None, timestamp=None):
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        hour = timestamp.hour
        month = timestamp.month
        is_weekend = 1 if timestamp.weekday() >= 5 else 0

        if actual_power is not None and theoretical_power > 0:
            power_efficiency = actual_power / theoretical_power
        else:
            power_efficiency = min(0.85, 0.3 + (wind_speed / 25))

        power_lag_1 = 0.0
        power_lag_6 = 0.0
        wind_speed_lag_1 = 0.0

        if len(self.historical_data) > 0:
            power_lag_1 = self.historical_data[-1].get('power', theoretical_power * power_efficiency)
            wind_speed_lag_1 = self.historical_data[-1].get('wind_speed', wind_speed)

        if len(self.historical_data) >= 6:
            power_lag_6 = self.historical_data[-6].get('power', theoretical_power * power_efficiency)

        features = {
            'Wind Speed (m/s)': wind_speed,
            'Wind Direction (°)': wind_direction,
            'Theoretical_Power_Curve (KWh)': theoretical_power,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'is_weekend': is_weekend,
            'day_of_week': timestamp.weekday(),
            'power_efficiency': power_efficiency,
            'wind_power_density': 0.5 * 1.225 * (wind_speed ** 3),
            'power_lag_1': power_lag_1,
            'power_lag_6': power_lag_6,
            'wind_speed_lag_1': wind_speed_lag_1
        }

        current_reading = {
            'timestamp': timestamp,
            'wind_speed': wind_speed,
            'power': actual_power if actual_power is not None else theoretical_power * power_efficiency
        }
        self.historical_data.append(current_reading)

        if len(self.historical_data) > self.max_history:
            self.historical_data.pop(0)

        return features

    def predict(self, wind_speed, wind_direction, theoretical_power, actual_power=None, timestamp=None):
        try:
            features = self.create_features(wind_speed, wind_direction, theoretical_power, actual_power, timestamp)

            feature_array = []
            for col in self.feature_columns:
                feature_array.append(features.get(col, 0.0))

            scaled_features = self.scaler.transform([feature_array])
            prediction = self.model.predict(scaled_features, verbose=0)[0][0]

            return round(prediction, 2)

        except Exception as e:
            print(f"Prediction error: {e}")
            return round(realistic_power_curve(wind_speed) * min(0.85, 0.3 + (wind_speed / 25)), 2)

    def get_confidence_interval(self, predicted):
        if len(self.prediction_errors) < 5:
            margin = max(50, abs(predicted) * 0.15)
        else:
            recent = self.prediction_errors[-50:]
            margin = 1.96 * np.std(recent)
        return (max(0, predicted - margin), predicted + margin)

    def monitor(self, wind_speed, wind_direction, theoretical_power, actual_power=None, timestamp=None):
        if actual_power is None:
            curve_power = realistic_power_curve(wind_speed)
            if wind_speed < 3.0:
                actual_power = 0
            elif wind_speed < 6.0:
                actual_power = curve_power * 0.3
            elif wind_speed < 12.0:
                actual_power = curve_power * 0.7
            else:
                actual_power = curve_power * 0.85

        predicted = self.predict(wind_speed, wind_direction, theoretical_power, actual_power, timestamp)
        prediction_error = abs(actual_power - predicted)

        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > 200:
            self.prediction_errors.pop(0)

        ci_lower, ci_upper = self.get_confidence_interval(predicted)

        current_data = {
            'power': actual_power,
            'wind_speed': wind_speed,
            'theoretical_power': theoretical_power,
            'prediction_error': prediction_error
        }

        anomalies = self.anomaly_detector.comprehensive_detection(current_data)

        result = {
            'predicted_power': float(predicted),
            'actual_power_used': float(actual_power),
            'prediction_error': float(prediction_error),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'anomalies': convert_to_native_types(anomalies),
            'health_score': float(max(0, 100 - anomalies['anomaly_score'] * 15))
        }

        return result
