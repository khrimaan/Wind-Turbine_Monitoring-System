# [file name]: production_system.py
import time
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

from turbine_monitor import TurbineMonitor
from anomaly_detector import AnomalyDetector
warnings.filterwarnings('ignore')

class ProductionTurbineSystem:
    def __init__(self):
        self.monitor = TurbineMonitor()
        self.alert_system = AlertSystem()
        self.data_logger = DataLogger()
        self.analytics = PerformanceAnalytics()
        
    
    def start_real_time_monitoring(self, data_source, update_interval=300):
        print("Starting Real-time Turbine Monitoring...")
        while True:
            try:
                # Get real-time data (replace with your data source)
                current_data = pd.read_csv('turbine_data2.csv')
                
                # Process each row individually, not entire columns
                for index, row in current_data.iterrows():
                    # Monitoring turbine for each data point
                    result = self.monitor.monitor(
                        wind_speed=row['wind_speed'],           # Single value, not entire column
                        wind_direction=row['wind_direction'],   # Single value
                        theoretical_power=row['theoretical_power'],  # Single value
                        actual_power=row['actual_power']        # Single value
                    )
                   
                    # Log results
                    self.data_logger.log_reading(result)
                    
                    # Send alerts if needed
                    if result['health_score'] < 80:
                        self.alert_system.send_alert(result)
                    
                    print(f"✅ Processed data point {index+1}: Health = {result['health_score']}%")
                
                # Wait for next update after processing all data
                print(f"🕒 Waiting {update_interval} seconds for next update...")
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)  # Wait before retrying

# the alert system will tell me about the health score and notify me when in low health score
class AlertSystem:
    def send_alert(self, result):
        if result['health_score'] < 50:
            print(f"🚨 CRITICAL ALERT: Health score {result['health_score']}%")
            print(f"   Issues: {result['anomalies']}")
        elif result['health_score'] < 80:
            print(f"⚠️ WARNING: Health score {result['health_score']}%")
            
# this will create a log for me to track
class DataLogger:
    def __init__(self):
        self.log_file = 'turbine_monitoring_log.csv'
    
    def save_to_csv(self, data):
        """Save data to CSV file"""
        df = pd.DataFrame([data])
        
        # Check if file exists using modern pandas method
        try:
            # Try to read the existing file to check if it exists and has content
            existing_df = pd.read_csv(self.log_file)
            # File exists - append without header
            df.to_csv(self.log_file, mode='a', header=False, index=False)
        except FileNotFoundError:
            # File doesn't exist - create with header
            df.to_csv(self.log_file, mode='w', header=True, index=False)
            print("📁 Created new turbine_monitoring_log.csv file")
    
    def log_reading(self, result):
        timestamp = pd.Timestamp.now()
        log_entry = {
            'timestamp': timestamp,
            'predicted_power': result['predicted_power'],
            'health_score': result['health_score'],
            'prediction_error': result['prediction_error'],
            'anomalies': result['anomalies']['severity'],
            'anomaly_score': result['anomalies']['anomaly_score']
        }
        # Save to CSV/database
        self.save_to_csv(log_entry)
        print(f"📝 Logged entry: Health = {result['health_score']}%")

class PerformanceAnalytics:
    def __init__(self):
        self.monitor = TurbineMonitor()
        self.log_file = 'turbine_monitoring_log.csv'
    
    def calculate_performance_metrics(self, days=30):
        """Calculate key performance indicators"""
        try:
            if not os.path.exists(self.log_file):
                return {
                    'availability': 0,
                    'efficiency': 0,
                    'mean_time_between_failures': "No data",
                    'prediction_accuracy': 0,
                    'data_points': 0,
                    'message': 'No monitoring data available. Process some data first using the Predict & Analyze feature.'
                }
                
            df = pd.read_csv(self.log_file)
            if len(df) == 0:
                return {
                    'availability': 0,
                    'efficiency': 0,
                    'mean_time_between_failures': "No data",
                    'prediction_accuracy': 0,
                    'data_points': 0,
                    'message': 'Log file exists but is empty. Process some data first.'
                }
                
            print(f"📊 Analyzing {len(df)} records from {self.log_file}")
            
            # Ensure timestamp column exists and convert
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                recent_data = df[df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(days=days)]
            else:
                recent_data = df.tail(10)  # Use recent entries if no timestamp
            
            if len(recent_data) == 0:
                return {
                    'availability': 0,
                    'efficiency': 0,
                    'mean_time_between_failures': "No recent data",
                    'prediction_accuracy': 0,
                    'data_points': 0,
                    'message': 'No recent data available for the selected time period.'
                }
            
            metrics = {
                'availability': self.calculate_availability(recent_data),
                'efficiency': self.calculate_efficiency(recent_data),
                'mean_time_between_failures': self.calculate_mtbf(recent_data),
                'prediction_accuracy': self.calculate_prediction_accuracy(recent_data),
                'data_points': len(recent_data),
                'message': f'Analysis based on {len(recent_data)} data points'
            }
            
            return metrics
        except Exception as e:
            return {
                'availability': 0,
                'efficiency': 0,
                'mean_time_between_failures': "Error",
                'prediction_accuracy': 0,
                'data_points': 0,
                'error': str(e),
                'message': f'Error calculating metrics: {str(e)}'
            }
    
    def calculate_availability(self, data):
        """Calculate turbine availability percentage"""
        if len(data) == 0:
            return 0
        # Turbine is available if health score is good
        available_count = len(data[data['health_score'] > 70])
        total_count = len(data)
        availability = (available_count / total_count) * 100
        return round(availability, 2)
    
    def calculate_efficiency(self, data):
        """Calculate operational efficiency"""
        if len(data) == 0:
            return 0
        # Use health score as efficiency proxy
        avg_efficiency = data['health_score'].mean()
        return round(avg_efficiency, 2)
    
    def calculate_mtbf(self, data):
        """Calculate Mean Time Between Failures"""
        if len(data) == 0:
            return "No data"
        # Count failures as health score < 50
        failures = len(data[data['health_score'] < 50])
        if failures == 0:
            return "No failures detected"
        total_readings = len(data)
        mtbf = total_readings / failures
        return f"{mtbf:.1f} readings between failures"
    
    def calculate_prediction_accuracy(self, data):
        """Calculate prediction accuracy"""
        if len(data) == 0 or 'prediction_error' not in data.columns:
            return 0
        avg_error = data['prediction_error'].mean()
        # Convert error to accuracy (lower error = higher accuracy)
        # Assuming average power around 1000-2000 KWh, normalize error
        if avg_error > 1000:  # Very high error
            accuracy = 0
        else:
            accuracy = max(0, 100 - (avg_error / 10))
        return round(accuracy, 2)