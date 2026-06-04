import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from turbine_monitor import TurbineMonitor, realistic_power_curve
    from anomaly_detector import AnomalyDetector
    MODELS_LOADED = True
except Exception as e:
    st.warning(f"Module load error: {e}")
    MODELS_LOADED = False

    def realistic_power_curve(wind_speed, rated_power=2000, cut_in=3.0, rated_speed=12.0, cut_out=25.0):
        v = np.asarray(wind_speed, dtype=float)
        scalar = v.ndim == 0
        v = np.atleast_1d(v)
        power = np.zeros_like(v)
        cubic = (v >= cut_in) & (v <= rated_speed)
        power[cubic] = rated_power * ((v[cubic] - cut_in) / (rated_speed - cut_in)) ** 3
        rated_mask = (v > rated_speed) & (v <= cut_out)
        power[rated_mask] = rated_power
        return float(power[0]) if scalar else power

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Turbine AI Monitoring System",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        background: linear-gradient(135deg, #4fc3f7 0%, #00bcd4 50%, #26a69a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(79, 195, 247, 0.2);
    }
    .anomaly-critical {
        background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%);
        padding: 1rem; border-radius: 10px; color: white;
        border-left: 4px solid #ff5252;
        box-shadow: 0 2px 10px rgba(183, 28, 28, 0.3);
    }
    .anomaly-warning {
        background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%);
        padding: 1rem; border-radius: 10px; color: white;
        border-left: 4px solid #ffab40;
        box-shadow: 0 2px 10px rgba(230, 81, 0, 0.3);
    }
    .anomaly-normal {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 1rem; border-radius: 10px; color: white;
        border-left: 4px solid #69f0ae;
        box-shadow: 0 2px 10px rgba(27, 94, 32, 0.3);
    }
    .section-header {
        font-size: 1.35rem; color: #4fc3f7;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #1e3a5f; font-weight: 600;
    }
    .thermo-card {
        background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 100%);
        padding: 1.2rem; border-radius: 12px; color: #e0e0e0;
        border: 1px solid rgba(79, 195, 247, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 0.5rem;
    }
    .ci-text {
        font-size: 0.85rem; color: #90caf9; margin-top: 0.3rem;
    }
    .last-updated {
        font-size: 0.8rem; color: #78909c;
        text-align: right; padding: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Thermodynamic Constants ──────────────────────────────────────────────────
BETZ_LIMIT = 16 / 27          # 0.5926 — theoretical maximum Cp
AIR_DENSITY = 1.225            # kg/m³ at sea level, 15 °C (ISA conditions)
ROTOR_RADIUS = 40              # meters (typical for a 2 MW class turbine)
ROTOR_AREA = np.pi * ROTOR_RADIUS ** 2   # ~5027 m²
ETA_MECHANICAL = 0.97          # gearbox + bearings
ETA_GENERATOR = 0.96           # electromagnetic conversion

CHART_BG = 'rgba(0,0,0,0)'

# ── Column auto-detection for dataset upload ─────────────────────────────────
COLUMN_CANDIDATES = {
    'wind_speed': ['Wind Speed (m/s)', 'wind_speed', 'Wind_Speed', 'WS',
                   'windspeed', 'Wind Speed', 'wind_speed_ms'],
    'wind_direction': ['Wind Direction (°)', 'wind_direction', 'Wind_Direction',
                       'WD', 'Wind Direction', 'wind_dir'],
    'theoretical_power': ['Theoretical_Power_Curve (KWh)', 'theoretical_power',
                          'Theoretical_Power', 'theo_power', 'Theoretical Power'],
    'actual_power': ['LV ActivePower (kW)', 'actual_power', 'Active_Power',
                     'power', 'Power', 'active_power', 'power_output',
                     'ActivePower', 'LV ActivePower'],
}


def auto_detect_columns(df_columns):
    detected = {}
    cols_lower = {c.lower().replace(' ', '').replace('_', ''): c for c in df_columns}
    for field, candidates in COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate in df_columns:
                detected[field] = candidate
                break
        if field not in detected:
            key = field.replace('_', '')
            for norm, orig in cols_lower.items():
                if key in norm:
                    detected[field] = orig
                    break
    return detected


# ── Thermodynamic Calculations ───────────────────────────────────────────────
def calculate_thermodynamics(wind_speed, actual_power):
    p_available = 0.5 * AIR_DENSITY * ROTOR_AREA * (wind_speed ** 3) / 1000  # kW
    p_betz_max = p_available * BETZ_LIMIT

    cp = min(actual_power / p_available, BETZ_LIMIT) if p_available > 0 else 0

    p_electrical = actual_power
    p_mechanical = p_electrical / ETA_GENERATOR if ETA_GENERATOR > 0 else 0
    p_rotor = p_mechanical / ETA_MECHANICAL if ETA_MECHANICAL > 0 else 0

    betz_unavoidable = p_available * (1 - BETZ_LIMIT)
    aerodynamic_loss = max(0, p_betz_max - p_rotor)
    mechanical_loss = max(0, p_rotor - p_mechanical)
    generator_loss = max(0, p_mechanical - p_electrical)

    exergy_destruction = max(0, p_available - p_electrical)
    exergy_efficiency = p_electrical / p_available if p_available > 0 else 0
    second_law_eff = cp / BETZ_LIMIT if BETZ_LIMIT > 0 else 0

    return {
        'p_available': p_available,
        'p_betz_max': p_betz_max,
        'cp': cp,
        'p_rotor': p_rotor,
        'p_mechanical': p_mechanical,
        'p_electrical': p_electrical,
        'betz_unavoidable': betz_unavoidable,
        'aerodynamic_loss': aerodynamic_loss,
        'mechanical_loss': mechanical_loss,
        'generator_loss': generator_loss,
        'exergy_destruction': exergy_destruction,
        'exergy_efficiency': exergy_efficiency,
        'second_law_efficiency': second_law_eff,
    }


# ── Simulator ────────────────────────────────────────────────────────────────
class TurbineSimulator:
    def __init__(self):
        self.rated_power = 2000
        self.time = 0

    def generate_data(self, mode="normal"):
        self.time += 1

        if mode == "normal":
            wind_speed = float(np.clip(np.random.normal(8, 2), 0, 25))
            efficiency = min(0.85, 0.3 + (wind_speed / 25))
        elif mode == "high_wind":
            wind_speed = float(np.clip(np.random.normal(20, 3), 15, 30))
            efficiency = 0.85
        elif mode == "low_efficiency":
            wind_speed = float(np.clip(np.random.normal(10, 1), 4, 20))
            efficiency = float(np.clip(np.random.normal(0.2, 0.05), 0.1, 0.3))
        elif mode == "cut_in_failure":
            wind_speed = float(np.clip(np.random.normal(6, 1), 3, 10))
            efficiency = 0.05
        else:
            wind_speed = float(np.clip(np.random.normal(8, 2), 0, 25))
            efficiency = min(0.85, 0.3 + (wind_speed / 25))

        theoretical_power = realistic_power_curve(wind_speed, self.rated_power)
        actual_power = theoretical_power * efficiency
        wind_direction = float(np.random.uniform(0, 360))

        return {
            'wind_speed': round(wind_speed, 2),
            'wind_direction': round(wind_direction, 1),
            'theoretical_power': round(float(theoretical_power), 2),
            'actual_power': round(float(actual_power), 2),
            'efficiency': round(float(efficiency), 3),
            'timestamp': datetime.now()
        }


# ── Chart helpers ────────────────────────────────────────────────────────────
def create_gauge_chart(value, title, min_val=0, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#e0e0e0'}},
        delta={'reference': 80, 'increasing': {'color': "#66bb6a"},
               'decreasing': {'color': '#ef5350'}},
        number={'font': {'color': '#e0e0e0'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1,
                     'tickcolor': '#4fc3f7', 'tickfont': {'color': '#aaa'}},
            'bar': {'color': '#4fc3f7'},
            'bgcolor': '#1e2130',
            'borderwidth': 2, 'bordercolor': '#333',
            'steps': [
                {'range': [0, 60], 'color': '#b71c1c'},
                {'range': [60, 80], 'color': '#e65100'},
                {'range': [80, 100], 'color': '#1b5e20'}],
            'threshold': {'line': {'color': '#ff5252', 'width': 4},
                          'thickness': 0.75, 'value': 90}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor=CHART_BG, font={'color': '#e0e0e0'})
    return fig


def create_power_chart(historical_data):
    if not historical_data:
        return go.Figure()
    df = pd.DataFrame(historical_data)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['theoretical_power'],
        name='Theoretical', line=dict(color='#42a5f5', dash='dash', width=2)))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['actual_power'],
        name='Actual', line=dict(color='#66bb6a', width=2)))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['predicted_power'],
        name='Predicted', line=dict(color='#ffa726', dash='dot', width=2)))

    if 'ci_lower' in df.columns and 'ci_upper' in df.columns and len(df) > 1:
        fig.add_trace(go.Scatter(
            x=pd.concat([df['timestamp'], df['timestamp'][::-1]]),
            y=pd.concat([df['ci_upper'], df['ci_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255, 167, 38, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI', showlegend=True))

    if len(df) >= 5:
        df['power_ma'] = df['actual_power'].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['power_ma'],
            name='Rolling Avg (5)',
            line=dict(color='#ce93d8', width=3, dash='dashdot')))

    fig.update_layout(
        title="Power Output Over Time",
        xaxis_title="Time", yaxis_title="Power (kW)", height=400,
        template='plotly_dark', paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        legend=dict(orientation='h', yanchor='bottom', y=1.02))
    return fig


def create_wind_chart(historical_data):
    if not historical_data:
        return go.Figure()
    df = pd.DataFrame(historical_data)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['wind_speed'],
        name='Wind Speed', line=dict(color='#ef5350', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['efficiency'],
        name='Efficiency', line=dict(color='#ab47bc', width=2)), secondary_y=True)

    if len(df) >= 5:
        df['ws_ma'] = df['wind_speed'].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['ws_ma'], name='Wind MA (5)',
            line=dict(color='#e57373', width=2, dash='dashdot')), secondary_y=False)

    fig.update_layout(
        title="Wind Speed & Turbine Efficiency",
        xaxis_title="Time", height=400,
        template='plotly_dark', paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=False)
    fig.update_yaxes(title_text="Efficiency", secondary_y=True)
    return fig


def create_health_trend_chart(historical_data):
    if not historical_data or len(historical_data) < 3:
        return None
    df = pd.DataFrame(historical_data)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['health_score'],
        name='Health Score', mode='lines+markers',
        line=dict(color='#4fc3f7', width=2), marker=dict(size=5)))

    x_num = np.arange(len(df))
    z = np.polyfit(x_num, df['health_score'].values, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=p(x_num),
        name='Trend', line=dict(color='#ffa726', dash='dash', width=2)))

    if len(df) >= 5:
        last_time = df['timestamp'].iloc[-1]
        if isinstance(last_time, datetime):
            future_times = [last_time + timedelta(minutes=10 * (i + 1)) for i in range(5)]
        else:
            future_times = list(range(len(df), len(df) + 5))
        future_y = np.clip(p(np.arange(len(df), len(df) + 5)), 0, 100)
        fig.add_trace(go.Scatter(
            x=future_times, y=future_y,
            name='Forecast', mode='lines',
            line=dict(color='#ffa726', dash='dot', width=2)))

    fig.add_hrect(y0=0, y1=60, fillcolor="rgba(183, 28, 28, 0.08)", line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="rgba(230, 81, 0, 0.06)", line_width=0)

    fig.update_layout(
        title="Health Score Trend & Forecast",
        xaxis_title="Time", yaxis_title="Health Score",
        yaxis=dict(range=[0, 105]), height=350,
        template='plotly_dark', paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
    return fig


def create_betz_gauge(cp):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(cp * 100, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Power Coefficient (Cp)', 'font': {'size': 16, 'color': '#e0e0e0'}},
        number={'suffix': '%', 'font': {'size': 26, 'color': '#e0e0e0'}},
        gauge={
            'axis': {'range': [0, 65], 'ticksuffix': '%',
                     'tickfont': {'color': '#aaa'}},
            'bar': {'color': '#26a69a'},
            'bgcolor': '#1e2130',
            'borderwidth': 2, 'bordercolor': '#333',
            'steps': [
                {'range': [0, 20], 'color': '#b71c1c'},
                {'range': [20, 35], 'color': '#e65100'},
                {'range': [35, 50], 'color': '#1b5e20'},
                {'range': [50, 59.3], 'color': '#004d40'}],
            'threshold': {'line': {'color': '#ff5252', 'width': 4},
                          'thickness': 0.75, 'value': 59.3}}))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor=CHART_BG,
        annotations=[dict(x=0.5, y=-0.1, text="Betz Limit: 59.3%",
                          showarrow=False, font=dict(size=12, color='#ff5252'))])
    return fig


def create_energy_waterfall(thermo):
    labels = ["Available<br>Wind Power", "Betz<br>Loss", "Aero<br>Loss",
              "Mech<br>Loss", "Gen<br>Loss", "Electrical<br>Output"]
    values = [thermo['p_available'], -thermo['betz_unavoidable'],
              -thermo['aerodynamic_loss'], -thermo['mechanical_loss'],
              -thermo['generator_loss'], 0]
    texts = [f"{abs(v):.0f} kW" for v in values]
    texts[-1] = f"{thermo['p_electrical']:.0f} kW"

    fig = go.Figure(go.Waterfall(
        x=labels, y=values,
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        textposition="outside", text=texts,
        connector={"line": {"color": "rgba(79, 195, 247, 0.3)"}},
        increasing={"marker": {"color": "#26a69a"}},
        decreasing={"marker": {"color": "#ef5350"}},
        totals={"marker": {"color": "#4fc3f7"}}))
    fig.update_layout(
        title="Energy Conversion Breakdown",
        height=350, template='plotly_dark',
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        showlegend=False)
    return fig


def create_power_curve_plot(rated_power=2000):
    ws = np.linspace(0, 30, 300)
    turbine_power = realistic_power_curve(ws, rated_power)
    betz_power = np.clip(
        0.5 * AIR_DENSITY * ROTOR_AREA * (ws ** 3) / 1000 * BETZ_LIMIT,
        0, rated_power * 5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws, y=betz_power, name='Betz Maximum', fill='tozeroy',
        fillcolor='rgba(79, 195, 247, 0.08)',
        line=dict(color='#4fc3f7', dash='dash', width=1)))
    fig.add_trace(go.Scatter(
        x=ws, y=turbine_power, name='Turbine Power Curve',
        line=dict(color='#66bb6a', width=3)))

    fig.add_vline(x=3, line_dash="dot", line_color="#ffa726",
                  annotation_text="Cut-in (3 m/s)",
                  annotation_font_color="#ffa726")
    fig.add_vline(x=12, line_dash="dot", line_color="#ffa726",
                  annotation_text="Rated (12 m/s)",
                  annotation_font_color="#ffa726")
    fig.add_vline(x=25, line_dash="dot", line_color="#ef5350",
                  annotation_text="Cut-out (25 m/s)",
                  annotation_font_color="#ef5350")

    fig.update_layout(
        title="Turbine Power Curve vs Betz Limit",
        xaxis_title="Wind Speed (m/s)", yaxis_title="Power (kW)",
        height=350, template='plotly_dark',
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
    return fig


# ── Helper: store result in session state ────────────────────────────────────
def _store_result(wind_speed, wind_direction, theoretical_power, actual_power,
                  result, timestamp):
    effective_actual = actual_power if actual_power is not None else result.get('actual_power_used', result.get('predicted_power', 0))
    theo = theoretical_power if theoretical_power > 0 else 1

    data_point = {
        'timestamp': timestamp,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'theoretical_power': theoretical_power,
        'actual_power': effective_actual,
        'predicted_power': result.get('predicted_power', 0),
        'ci_lower': result.get('ci_lower', 0),
        'ci_upper': result.get('ci_upper', 0),
        'efficiency': effective_actual / theo if theo > 0 else 0,
        'health_score': result.get('health_score', 100),
        'anomalies': result.get('anomalies', {}),
    }
    st.session_state.historical_data.append(data_point)
    if len(st.session_state.historical_data) > 200:
        st.session_state.historical_data.pop(0)
    st.session_state.last_result = result
    st.session_state.last_updated = datetime.now()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">Wind Turbine AI Monitoring System</h1>',
                unsafe_allow_html=True)

    # ── Session state init ───────────────────────────────────────────────────
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = []
    if 'monitor' not in st.session_state and MODELS_LOADED:
        try:
            st.session_state.monitor = TurbineMonitor()
        except Exception:
            st.session_state.monitor = None
    if 'last_updated' not in st.session_state:
        st.session_state.last_updated = None

    monitor = st.session_state.get('monitor')

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Control Panel")

        operation_mode = st.selectbox(
            "Operation Mode",
            ["Manual Input", "Normal Operation", "High Wind Scenario",
             "Low Efficiency Scenario", "Cut-in Failure Scenario",
             "Load Dataset"])

        # ── Manual Input ─────────────────────────────────────────────────────
        if operation_mode == "Manual Input":
            st.markdown("### Input Parameters")
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 8.0, 0.1)
            wind_direction = st.slider("Wind Direction (deg)", 0, 360, 180, 1)

            default_theo = realistic_power_curve(wind_speed)
            theoretical_power = st.number_input(
                "Theoretical Power (kW)", value=int(default_theo), step=50,
                help=f"Realistic curve: {default_theo:.0f} kW at {wind_speed:.1f} m/s")
            actual_power = st.number_input(
                "Actual Power (kW) — leave 0 to auto-estimate", value=0, step=50)
            actual_power = actual_power if actual_power > 0 else None

            if st.button("Analyze Turbine", use_container_width=True, type="primary"):
                if MODELS_LOADED and monitor:
                    result = monitor.monitor(
                        wind_speed, wind_direction, theoretical_power, actual_power)
                    _store_result(wind_speed, wind_direction, theoretical_power,
                                 actual_power, result, datetime.now())
                else:
                    st.error("Turbine monitor not available.")

        # ── Load Dataset ─────────────────────────────────────────────────────
        elif operation_mode == "Load Dataset":
            st.markdown("### Dataset Upload")
            uploaded_file = st.file_uploader("Upload a wind turbine CSV", type=['csv'])

            if uploaded_file is not None:
                df_raw = pd.read_csv(uploaded_file)
                st.caption(f"{len(df_raw)} rows, {len(df_raw.columns)} columns")
                detected = auto_detect_columns(df_raw.columns.tolist())
                all_cols = df_raw.columns.tolist()

                def _safe_idx(field):
                    if field in detected and detected[field] in all_cols:
                        return all_cols.index(detected[field])
                    return 0

                def _safe_idx_opt(field):
                    if field in detected and detected[field] in all_cols:
                        return all_cols.index(detected[field]) + 1
                    return 0

                col_ws = st.selectbox("Wind Speed column", all_cols, index=_safe_idx('wind_speed'))
                col_wd = st.selectbox("Wind Direction column", all_cols, index=_safe_idx('wind_direction'))

                opt_cols = ["(auto-calculate)"] + all_cols
                col_tp = st.selectbox("Theoretical Power column", opt_cols,
                                      index=_safe_idx_opt('theoretical_power'))
                col_ap = st.selectbox("Actual Power column", opt_cols,
                                      index=_safe_idx_opt('actual_power'))

                max_rows = st.slider("Rows to process", 10, min(500, len(df_raw)),
                                     min(100, len(df_raw)))

                if st.button("Process Dataset", use_container_width=True, type="primary"):
                    if MODELS_LOADED and monitor:
                        st.session_state.historical_data = []
                        progress = st.progress(0, text="Processing...")

                        for i in range(max_rows):
                            row = df_raw.iloc[i]
                            ws = float(row[col_ws])
                            wd = float(row[col_wd])
                            tp = float(row[col_tp]) if col_tp != "(auto-calculate)" else float(realistic_power_curve(ws))
                            ap = float(row[col_ap]) if col_ap != "(auto-calculate)" else None

                            result = monitor.monitor(ws, wd, tp, ap)
                            ts = datetime.now() + timedelta(minutes=i * 10)
                            _store_result(ws, wd, tp, ap, result, ts)
                            progress.progress((i + 1) / max_rows,
                                              text=f"Processing row {i+1}/{max_rows}")

                        progress.empty()
                        st.success(f"Processed {max_rows} data points")
                    else:
                        st.error("Monitor not available.")

        # ── Simulation Modes ─────────────────────────────────────────────────
        else:
            st.markdown("### Simulation Mode")
            if st.button("Generate Data", use_container_width=True, type="primary"):
                simulator = TurbineSimulator()
                mode_map = {
                    "Normal Operation": "normal",
                    "High Wind Scenario": "high_wind",
                    "Low Efficiency Scenario": "low_efficiency",
                    "Cut-in Failure Scenario": "cut_in_failure",
                }
                data = simulator.generate_data(mode_map.get(operation_mode, "normal"))

                if MODELS_LOADED and monitor:
                    result = monitor.monitor(
                        data['wind_speed'], data['wind_direction'],
                        data['theoretical_power'], data['actual_power'])
                    _store_result(data['wind_speed'], data['wind_direction'],
                                 data['theoretical_power'], data['actual_power'],
                                 result, data['timestamp'])

        st.divider()

        # ── Export ───────────────────────────────────────────────────────────
        if st.session_state.historical_data:
            st.markdown("### Export Data")
            export_df = pd.DataFrame(st.session_state.historical_data)
            export_cols = [c for c in export_df.columns if c != 'anomalies']
            csv_bytes = export_df[export_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Monitoring CSV",
                data=csv_bytes,
                file_name=f"turbine_monitoring_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True)

            anomaly_rows = []
            for dp in st.session_state.historical_data:
                anomalies = dp.get('anomalies', {})
                for category, flags in anomalies.items():
                    if category in ('severity', 'anomaly_score'):
                        continue
                    if isinstance(flags, dict):
                        for flag, val in flags.items():
                            if val:
                                anomaly_rows.append({
                                    'timestamp': dp['timestamp'],
                                    'category': category,
                                    'anomaly': flag,
                                    'wind_speed': dp['wind_speed'],
                                    'health_score': dp['health_score'],
                                })
            if anomaly_rows:
                anom_csv = pd.DataFrame(anomaly_rows).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Anomaly Log",
                    data=anom_csv,
                    file_name=f"anomaly_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True)

        if st.button("Clear Historical Data", use_container_width=True):
            st.session_state.historical_data = []
            if 'last_result' in st.session_state:
                del st.session_state.last_result
            st.rerun()

        st.divider()
        st.markdown("### System Status")
        status_icon = "Loaded" if MODELS_LOADED else "Simulation Mode"
        st.info(f"ML Models: {status_icon}")
        st.info(f"Data Points: {len(st.session_state.historical_data)}")

    # ── Last Updated ─────────────────────────────────────────────────────────
    if st.session_state.get('last_updated'):
        st.markdown(
            f'<div class="last-updated">Last updated: '
            f'{st.session_state.last_updated.strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  DASHBOARD — Top Metrics
    # ═════════════════════════════════════════════════════════════════════════
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.plotly_chart(
                create_gauge_chart(result['health_score'], "Turbine Health"),
                use_container_width=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Predicted Power", f"{result['predicted_power']:.0f} kW")
            st.metric("Prediction Error", f"{result['prediction_error']:.1f} kW")
            ci_lo = result.get('ci_lower', 0)
            ci_hi = result.get('ci_upper', 0)
            st.markdown(
                f'<div class="ci-text">95% CI: [{ci_lo:.0f}, {ci_hi:.0f}] kW</div>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            severity = result['anomalies'].get('severity', 'normal')
            score = result['anomalies'].get('anomaly_score', 0)
            labels = {'critical': 'CRITICAL', 'warning': 'WARNING', 'normal': 'NORMAL'}
            st.markdown(f'<div class="anomaly-{severity}">', unsafe_allow_html=True)
            st.subheader(labels.get(severity, 'NORMAL'))
            st.write(f"Anomaly Score: {score}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown("### Detected Anomalies")
            found = False
            for cat, flags in result['anomalies'].items():
                if cat in ('severity', 'anomaly_score'):
                    continue
                if isinstance(flags, dict):
                    for name, val in flags.items():
                        if val:
                            st.error(f"{name.replace('_', ' ').title()}")
                            found = True
                elif isinstance(flags, list) and flags:
                    st.error(f"{cat.replace('_', ' ').title()}: {len(flags)}")
                    found = True
            if not found:
                st.success("No anomalies detected")
    else:
        st.info("Select a mode and generate data to begin monitoring.")

    # ═════════════════════════════════════════════════════════════════════════
    #  PERFORMANCE CHARTS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">Performance Analytics</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.historical_data:
            st.plotly_chart(create_power_chart(st.session_state.historical_data),
                           use_container_width=True)
        else:
            st.info("No data yet — generate some to see charts.")
    with col2:
        if st.session_state.historical_data:
            st.plotly_chart(create_wind_chart(st.session_state.historical_data),
                           use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TREND ANALYSIS & FORECASTING
    # ═════════════════════════════════════════════════════════════════════════
    if len(st.session_state.historical_data) >= 3:
        st.markdown('<div class="section-header">Trend Analysis & Forecasting</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            health_fig = create_health_trend_chart(st.session_state.historical_data)
            if health_fig:
                st.plotly_chart(health_fig, use_container_width=True)

                df_t = pd.DataFrame(st.session_state.historical_data)
                slope = np.polyfit(np.arange(len(df_t)),
                                   df_t['health_score'].values, 1)[0]
                current_health = df_t['health_score'].iloc[-1]
                if slope < -1:
                    readings_to_critical = max(1, int((current_health - 60) / abs(slope)))
                    st.warning(
                        f"Health declining at {slope:.2f} pts/reading. "
                        f"Estimated ~{readings_to_critical} readings until warning zone.")
                elif slope > 0.5:
                    st.success(f"Health trend improving (+{slope:.2f} pts/reading)")
                else:
                    st.info("Health trend is stable")

        with col2:
            df_e = pd.DataFrame(st.session_state.historical_data)
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(
                x=df_e['timestamp'], y=df_e['efficiency'],
                name='Efficiency', mode='lines+markers',
                line=dict(color='#ab47bc', width=2), marker=dict(size=4)))
            if len(df_e) >= 5:
                df_e['eff_ma'] = df_e['efficiency'].rolling(5, min_periods=1).mean()
                fig_eff.add_trace(go.Scatter(
                    x=df_e['timestamp'], y=df_e['eff_ma'],
                    name='Moving Avg (5)',
                    line=dict(color='#ce93d8', width=3, dash='dashdot')))
            fig_eff.update_layout(
                title="Efficiency Trend", xaxis_title="Time",
                yaxis_title="Efficiency", height=350,
                template='plotly_dark',
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG)
            st.plotly_chart(fig_eff, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  THERMODYNAMIC ANALYSIS  (EN202)
    # ═════════════════════════════════════════════════════════════════════════
    if 'last_result' in st.session_state and st.session_state.historical_data:
        st.markdown(
            '<div class="section-header">Thermodynamic Analysis (EN202)</div>',
            unsafe_allow_html=True)

        last_dp = st.session_state.historical_data[-1]
        ws = last_dp['wind_speed']
        ap = last_dp['actual_power']
        thermo = calculate_thermodynamics(ws, ap)

        # ── Key metrics row ──────────────────────────────────────────────────
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("Available Wind Power", f"{thermo['p_available']:.1f} kW")
            st.caption(f"P = ½ρAv³  |  ρ = {AIR_DENSITY} kg/m³, A = {ROTOR_AREA:.0f} m²")
            st.markdown('</div>', unsafe_allow_html=True)
        with tc2:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("Betz Maximum", f"{thermo['p_betz_max']:.1f} kW")
            st.caption(f"P_betz = P_avail x 16/27  ({BETZ_LIMIT:.1%})")
            st.markdown('</div>', unsafe_allow_html=True)
        with tc3:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("Power Coefficient (Cp)", f"{thermo['cp']:.4f}")
            st.caption("Cp = P_actual / P_available")
            st.markdown('</div>', unsafe_allow_html=True)
        with tc4:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.metric("2nd Law Efficiency", f"{thermo['second_law_efficiency']:.1%}")
            st.caption("η_II = Cp / Cp_Betz  (closeness to theoretical max)")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Charts row ───────────────────────────────────────────────────────
        tc1, tc2 = st.columns(2)
        with tc1:
            st.plotly_chart(create_betz_gauge(thermo['cp']), use_container_width=True)
        with tc2:
            if thermo['p_available'] > 0:
                st.plotly_chart(create_energy_waterfall(thermo),
                               use_container_width=True)
            else:
                st.info("Wind speed too low for energy breakdown.")

        # ── Exergy + power curve row ─────────────────────────────────────────
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown('<div class="thermo-card">', unsafe_allow_html=True)
            st.markdown("#### Exergy Analysis")
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                st.write(f"**Exergy Input:** {thermo['exergy_destruction'] + thermo['p_electrical']:.1f} kW")
                st.write(f"**Useful Work:** {thermo['p_electrical']:.1f} kW")
            with ex_col2:
                st.write(f"**Irreversibility:** {thermo['exergy_destruction']:.1f} kW")
                st.write(f"**Exergy η:** {thermo['exergy_efficiency']:.1%}")
            st.progress(min(1.0, max(0.0, thermo['exergy_efficiency'])))
            st.caption(
                "Irreversibilities include aerodynamic losses (wake, tip vortices), "
                "mechanical friction (gearbox, bearings), and electromagnetic losses "
                "(generator copper & iron losses).")
            st.markdown('</div>', unsafe_allow_html=True)
        with tc2:
            st.plotly_chart(create_power_curve_plot(), use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  ANOMALY DETAILS  (collapsible)
    # ═════════════════════════════════════════════════════════════════════════
    if 'last_result' in st.session_state:
        st.markdown(
            '<div class="section-header">Detailed Anomaly Analysis</div>',
            unsafe_allow_html=True)

        anomalies = st.session_state.last_result['anomalies']
        categories = [
            ('operational', 'Operational'),
            ('wind_speed', 'Wind Speed'),
            ('vibration', 'Vibration'),
            ('statistical', 'Statistical'),
        ]

        for key, label in categories:
            cat_data = anomalies.get(key, {})
            has_issues = (isinstance(cat_data, dict) and any(cat_data.values())) or \
                         (isinstance(cat_data, list) and len(cat_data) > 0)
            icon = "!!" if has_issues else "OK"

            with st.expander(f"[{icon}] {label}", expanded=has_issues):
                if isinstance(cat_data, dict) and cat_data:
                    for anomaly, detail in cat_data.items():
                        if detail:
                            st.error(f"**{anomaly.replace('_', ' ').title()}**")
                        else:
                            st.success(f"{anomaly.replace('_', ' ').title()} — OK")
                elif not cat_data:
                    st.success("No anomalies")
                else:
                    st.warning(f"{len(cat_data)} anomalies detected")

    # ═════════════════════════════════════════════════════════════════════════
    #  SYSTEM INFO  (collapsible)
    # ═════════════════════════════════════════════════════════════════════════
    with st.expander("System Information"):
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown("**Detection Methods**")
            for m in ["Isolation Forest", "One-Class SVM",
                       "Statistical (Z-score, IQR)",
                       "Temporal Pattern Analysis",
                       "Operational Logic Rules"]:
                st.write(f"- {m}")
        with ic2:
            st.markdown("**Monitored Parameters**")
            for p in ["Power Output", "Wind Speed & Direction",
                       "Turbine Efficiency", "Vibration Patterns",
                       "Seasonal Variations"]:
                st.write(f"- {p}")
        with ic3:
            st.markdown("**Turbine Specifications**")
            st.write(f"- Rated Power: 2,000 kW")
            st.write(f"- Rotor Radius: {ROTOR_RADIUS} m")
            st.write(f"- Swept Area: {ROTOR_AREA:,.0f} m²")
            st.write(f"- Cut-in: 3 m/s | Rated: 12 m/s | Cut-out: 25 m/s")
            st.write(f"- Air Density: {AIR_DENSITY} kg/m³ (ISA sea level)")


if __name__ == "__main__":
    main()
