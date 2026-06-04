---
title: Wind Turbine Monitor
emoji: 🌬️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🌬️ Wind Turbine AI Monitoring System

![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

An advanced, real-time wind turbine performance monitoring dashboard and anomaly detection system. Originally developed for an Engineering Thermodynamics course (EN202), this project combines a trained Deep Learning model for power prediction with a multi-algorithmic anomaly detection engine and detailed thermodynamic analysis.

---

## ✨ Features

- **Deep Learning Power Prediction**: Utilizes a Keras neural network to predict expected power output based on current wind conditions, computing features like time-cyclic encodings and wind power density.
- **Realistic Power Curve Simulation**: Implements a cubic wind turbine power curve with accurate cut-in (3 m/s), rated (12 m/s), and cut-out (25 m/s) regions.
- **Multi-Method Anomaly Detection**: Runs 6 concurrent anomaly detection algorithms to flag statistical, operational, temporal, and vibration-related issues.
- **Engineering Thermodynamics (EN202) Integration**: Performs real-time exergy analysis, energy conversion waterfall tracking, and 2nd Law of Thermodynamics efficiency calculations relative to the Betz limit.
- **Dataset Upload & Batch Processing**: Support for uploading CSV files (e.g., SCADA datasets) with automatic column detection to process historical data.
- **Trend Analysis & Health Forecasting**: Calculates rolling averages, degradation trend lines, and extrapolates time-to-critical warnings.
- **Data Export**: Export monitoring results and filtered anomaly logs to CSV at any time.
- **Interactive UI/UX**: Dark-themed Streamlit dashboard with collapsible sections, dynamic Plotly gauge charts, 95% confidence intervals on predictions, and metric cards.

---

## 🔍 Anomaly Detection Engine

The system uses a comprehensive ensemble approach to detect faults:

1. **Machine Learning Methods**: `Isolation Forest` and `One-Class SVM` to detect unusual multivariate patterns.
2. **Statistical Methods**: Z-score, modified Z-score, and IQR to detect extreme outliers in power readings.
3. **Temporal Methods**: Detects sudden changes (>500kW drops), stuck sensor values (low rolling variance), and seasonal/hourly anomalies.
4. **Operational Logic Rules**: Physics-based checks for cut-in failures, rated power failures, and unrealistic efficiency metrics.
5. **Wind Speed Checks**: Physical limits (negative wind or hurricane forces) and rapid rate-of-change spikes.
6. **Simulated Vibration/Mechanical**: Analyzes power output standard deviation for mechanical instability and trends for blade icing.

---

## 🌡️ Thermodynamic Analysis

For deep engineering context, the system calculates real-time thermodynamics:

- **Available Wind Power**: $P = \frac{1}{2} \rho A v^3$
- **Betz Maximum Limit**: Tracks energy capture against the 59.3% theoretical limit ($C_p$).
- **Energy Waterfall**: Tracks losses through aerodynamic extraction, mechanical drivetrain (gearbox), and electromagnetic generator conversions.
- **Exergy Analysis**: Computes useful work vs. exergy destruction (irreversibilities) to provide 2nd Law Efficiency ($\eta_{II}$).

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+ (Tested on 3.13)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/khrimaan/Wind-Turbine_Monitoring-System.git
cd Wind-Turbine_Monitoring-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 🎮 Usage Guide

The dashboard sidebar offers three primary modes of operation:

### 1. Manual Input
Directly adjust **Wind Speed**, **Wind Direction**, and **Actual Power** via sliders and inputs to see how the model and anomaly detectors react to specific scenarios.

### 2. Load Dataset
Upload a real-world CSV dataset (e.g., from Kaggle). The system attempts to auto-detect the required columns (Wind Speed, Direction, Power). You can process up to 500 rows sequentially to observe historical trends and simulate real-world monitoring.

### 3. Simulation Scenarios
Generate synthetic data based on predefined operational states:
- **Normal Operation**: Standard variance around typical wind speeds.
- **High Wind Scenario**: Near cut-out speeds testing rated power limits.
- **Low Efficiency Scenario**: Simulates a dirty/degraded rotor or mechanical friction.
- **Cut-in Failure Scenario**: Simulates a turbine failing to generate power despite sufficient wind.

---

## 📂 Project Structure

```
.
├── app.py                       # Main Streamlit dashboard UI and application logic
├── turbine_monitor.py           # Core orchestrator: feature engineering & power prediction
├── anomaly_detector.py          # Ensemble anomaly detection algorithms
├── production_system.py         # Standalone CLI runner with alerts and logging
├── requirements.txt             # Python dependencies
├── .streamlit/
│   └── config.toml              # UI Theme configuration
└── model_files/                 
    ├── wind_power_predictor.keras # Pre-trained Keras Neural Network
    ├── feature_scaler.pkl         # Fitted scikit-learn StandardScaler
    └── feature_columns.txt        # Expected feature order mapping
```

---

## 📄 License
This project is open-source and available under the MIT License.
