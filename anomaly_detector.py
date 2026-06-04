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
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self):
        self.anomaly_history = []
        self.setup_detectors()
    
    def setup_detectors(self):
        # setting up three different anomaly detection methods to my anomaly detector. 
        
        # detecting anomaly using the z_score and the percentile methods, basically for the outliers
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        # ML Detectors
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.one_class_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
        # Clustering based
        self.dbscan = DBSCAN(eps=0.5, min_samples=10)
        # for the rolling bassed stats or the continuous data anomaly
        self.window_size = 24
        self.control_limits = {}
    
    def statistical_anomalies(self, power_readings):
        anomalies = {}
        if len(power_readings) < 2:
            return anomalies
        
        # Zscore method and the modified Zscore for the the detection of outliers properly
        z_scores = np.abs(stats.zscore(power_readings))
        anomalies['z_score_anomalies'] = np.where(z_scores > self.z_score_threshold)[0].tolist()
        median = np.median(power_readings)
        mad = np.median(np.abs(power_readings - median))
        modified_z_scores = 0.6745 * (power_readings - median) / mad if mad != 0 else 0
        anomalies['modified_z_anomalies'] = np.where(np.abs(modified_z_scores) > 3.5)[0].tolist()
        # usinf the iqr range to find out outliers in the dataset
        Q1 = np.percentile(power_readings, 25)
        Q3 = np.percentile(power_readings, 75)
        l_bound = Q1 - self.iqr_multiplier * (Q3 - Q1)
        u_bound = Q3 + self.iqr_multiplier * (Q3 - Q1)
        anomalies['iqr_anomalies'] = np.where((power_readings < l_bound) | (power_readings > u_bound))[0].tolist()
       
        return anomalies
    
    def ml_anomalies(self, features):
        anomalies = {}
        if len(features) < 10:
            return anomalies
        
        # using iso forest to detect any see if there is any unusal pattern in the data
        iso_predictions = self.iso_forest.fit_predict(features)
        anomalies['isolation_forest'] = np.where(iso_predictions == -1)[0].tolist()
        # again, for the same purpose, oneclass svm is called
        svm_predictions = self.one_class_svm.fit_predict(features)
        anomalies['one_class_svm'] = np.where(svm_predictions == -1)[0].tolist()
        
        return anomalies
    
    def temporal_anomalies(self, power_series, timestamps):
        anomalies = {}
        if len(power_series) < self.window_size:
            return anomalies
        
        # to detech sudden drops or increase in values using 
        diffs = np.diff(power_series)
        sudden_changes = np.where(np.abs(diffs) > 500)[0]  # morre than 500kW change is not normal
        anomalies['sudden_changes'] = sudden_changes.tolist()
        # to detect if there is no variation
        rolling_std = pd.Series(power_series).rolling(window=6).std()
        stuck_values = np.where(rolling_std < 10)[0]  # Very low variation
        anomalies['stuck_values'] = stuck_values.tolist()
        # for the seasonal anomalies i.e, comparing to the past patterns
        if len(power_series) >= 168: # 24 hours a day x 7 days for a total of 168 hours
            hourly_pattern = pd.Series(power_series).groupby(pd.Series(timestamps).dt.hour).mean()
            current_hour_avg = power_series[-24:].mean()
            hour = timestamps[-1].hour if hasattr(timestamps[-1], 'hour') else 12
            expected_avg = hourly_pattern.get(hour, current_hour_avg)
            
            if abs(current_hour_avg - expected_avg) > 200:  # if a 200kW deviation is seen, it is not normal
                anomalies['seasonal_anomaly'] = [len(power_series)-1]
        
        return anomalies
    
    def operational_anomalies(self, wind_speed, power, theoretical_power):
        anomalies = {}
    
    # Check for cut-in failure (ONLY at low-medium winds)
        if 4.0 <= wind_speed <= 8.0 and power < 50:
            anomalies['cut_in_failure'] = True
    
    # Check for rated power failure (at high winds)
        elif wind_speed > 12.0 and power < theoretical_power * 0.8:
            anomalies['rated_power_failure'] = True
    
    # Check for general power underperformance
        elif wind_speed > 8.0 and power < theoretical_power * 0.5:
            anomalies['power_underperformance'] = True
    
    # Calculate efficiency safely
        if theoretical_power > 0:
            efficiency = power / theoretical_power
        else:
            efficiency = 0
    
    # Efficiency checks
        if wind_speed > 6.0:
            if efficiency < 0.2:
                anomalies['low_efficiency'] = True
            elif efficiency > 1.1:
                anomalies['unrealistic_efficiency'] = True
    
        return anomalies
    
    def wind_speed_anomalies(self, wind_speed, historical_wind_speeds=None):

        anomalies = {}
    
    # Physical limits check
        if wind_speed > 30.0:  # Hurricane force winds - turbine would be shut down
            anomalies['extreme_wind_speed'] = True
    
        if wind_speed < 0:  # Impossible negative wind speed
            anomalies['negative_wind_speed'] = True
    
    # Rate of change check (if historical data available)
        if historical_wind_speeds and len(historical_wind_speeds) > 1:
            recent_change = abs(wind_speed - historical_wind_speeds[-1])
            if recent_change > 10.0:  # More than 10 m/s change in one reading
                anomalies['wind_speed_spike'] = True
    
    # Consistency check for very high winds
        if wind_speed > 20.0:
        # At these speeds, power should be near rated or decreasing
            anomalies['very_high_wind'] = True
    
        return anomalies
    
    def vibration_anomalies_simulated(self, power, wind_speed):
        anomalies = {}
        
        # mechanical isses
        if wind_speed > 6.0:
            # High vibration might cause power fluctuations
            recent_power = power[-10:] if hasattr(power, '__len__') else [power]
            if len(recent_power) > 5:
                power_std = np.std(recent_power)
                if power_std > 300:  # high deviations is a bad signw
                    anomalies['high_vibration'] = True
            # blade icing (gradual power decrease)
            if len(recent_power) > 20:
                trend = np.polyfit(range(len(recent_power)), recent_power, 1)[0]
                if trend < -20:  # Strong negative trend
                    anomalies['blade_icing'] = True
        
        return anomalies
    
    def comprehensive_detection(self, current_data, historical_data=None):
        all_anomalies = {}
    
    # Extract data
        current_power = current_data.get('power', 0)
        wind_speed = current_data.get('wind_speed', 0)
        theoretical_power = current_data.get('theoretical_power', 0)
        timestamp = current_data.get('timestamp')
    
    # Update history
        self.anomaly_history.append(current_data)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]
    
    # Get historical wind speeds for trend analysis
        historical_wind_speeds = [data.get('wind_speed', 0) for data in self.anomaly_history[-10:]]  # Last 10 readings
    
    # Run all detectors
        all_anomalies['wind_speed'] = self.wind_speed_anomalies(wind_speed, historical_wind_speeds)
        all_anomalies['statistical'] = self.statistical_anomalies([current_power])
        all_anomalies['ml'] = self.ml_anomalies([[current_power, wind_speed, theoretical_power]])
        all_anomalies['operational'] = self.operational_anomalies(wind_speed, current_power, theoretical_power)
        all_anomalies['vibration'] = self.vibration_anomalies_simulated([current_power], wind_speed)
    
    # Calculate overall anomaly score
        anomaly_score = 0
    
    # Wind speed anomalies are critical
        if all_anomalies['wind_speed'].get('extreme_wind_speed'):
            anomaly_score += 3
        if all_anomalies['wind_speed'].get('wind_speed_spike'):
            anomaly_score += 2
    
    # Operational anomalies
        if all_anomalies['operational']:
            anomaly_score += len(all_anomalies['operational']) * 1.5
    
    # Vibration anomalies
        if all_anomalies['vibration']:
            anomaly_score += len(all_anomalies['vibration']) * 1.0
    
        all_anomalies['anomaly_score'] = anomaly_score
    
    # Set severity
        if anomaly_score >= 3:
            all_anomalies['severity'] = 'critical'
        elif anomaly_score >= 1:
            all_anomalies['severity'] = 'warning'
        else:
            all_anomalies['severity'] = 'normal'
    
        return all_anomalies

