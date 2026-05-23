import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data.generate_data import generate_synthetic_data
from models.forecasting import train_and_forecast_classical
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics

# Page configuration
st.set_page_config(
    page_title="Financial KPI Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (dark-mode theme with glassmorphism and gradient accents)
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Header Banner */
    .header-container {
        background: linear-gradient(135deg, #1f1235 0%, #0f081d 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    .header-title {
        background: linear-gradient(90deg, #a78bfa 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: -0.05em;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    
    /* Glassmorphic Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(167, 139, 250, 0.3);
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #34d399;
        margin-top: 0.25rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Database path definition
DB_PATH = "financial_kpi.db"

# Bootstrapping helper
@st.cache_resource
def bootstrap_database(db_path):
    """Initializes the database and seeds it with data if empty."""
    if not os.path.exists(db_path):
        init_db(db_path)
        # Generate 3 years of daily transaction data for E-commerce revenue (Issue 1)
        df_seed = generate_synthetic_data(start_date="2023-01-01", end_date="2025-12-31")
        insert_transactions(db_path, df_seed)
        return True
    return False

# Trigger bootstrap
bootstrapped = bootstrap_database(DB_PATH)

# Forecasting cache helper
@st.cache_data(show_spinner=False)
def run_forecast_pipeline(df_data):
    """Fitted models run and results are cached to ensure dashboard interactivity."""
    return train_and_forecast_classical(df_data, forecast_horizon=90)

# Main layout header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Financial KPI Intelligence</h1>
    <div class="header-subtitle">Continuous Anomaly Detection & Revenue Forecasting Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filter Section
st.sidebar.markdown("<h2 style='font-weight: 600; font-size: 1.3rem; margin-bottom: 1rem;'>Control Panel</h2>", unsafe_allow_html=True)

# Control panel filters
business_unit = st.sidebar.selectbox(
    "Business Unit",
    options=["ecommerce", "saas", "enterprise"],
    format_func=lambda x: {
        "ecommerce": "🛍️ E-commerce",
        "saas": "💻 SaaS Subscription",
        "enterprise": "🏢 Enterprise Services"
    }.get(x, x)
)

metric = st.sidebar.selectbox(
    "Financial Metric",
    options=["revenue", "cost", "volume"],
    format_func=lambda x: {
        "revenue": "💰 Revenue ($)",
        "cost": "💸 Cost ($)",
        "volume": "📦 Volume (Units)"
    }.get(x, x)
)

frequency = st.sidebar.selectbox(
    "Time Granularity",
    options=["D", "W", "M"],
    format_func=lambda x: "📅 Daily" if x == "D" else "🗓️ Weekly" if x == "W" else "🗂️ Monthly"
)

# Load the aggregated metric from database
df_metric = query_aggregated_metrics(DB_PATH, business_unit, metric, frequency)

# Compute basic statistics for cards
if not df_metric.empty:
    total_val = df_metric['Value'].sum()
    avg_val = df_metric['Value'].mean()
    max_val = df_metric['Value'].max()
    metric_unit = "$" if metric in ["revenue", "cost"] else ""
    
    # Display Stats Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Accumulation</div>
            <div class="metric-value">{metric_unit}{total_val:,.2f}</div>
            <div class="metric-sub">Full Historical Sum</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average per Period</div>
            <div class="metric-value">{metric_unit}{avg_val:,.2f}</div>
            <div class="metric-sub">Typical Performance</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Value</div>
            <div class="metric-value">{metric_unit}{max_val:,.2f}</div>
            <div class="metric-sub">Historical High</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

# Tabs definitions
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Raw Data Explorer", 
    "📈 Forecast Projection", 
    "⚠️ Anomaly Dashboard", 
    "🧠 Explainability (SHAP)"
])

# TAB 1: Raw Data & EDA Explorer
with tab1:
    st.markdown("### Exploratory Data Analysis & Raw Table")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.dataframe(
            df_metric, 
            use_container_width=True, 
            hide_index=True
        )
        
    with col_right:
        st.markdown("#### Period Statistics Summary")
        summary_stats = df_metric['Value'].describe().to_frame()
        st.table(summary_stats)
        
        st.info("💡 Note: The metrics displayed above are queried dynamically from the database and aggregated on-the-fly.")

# TAB 2: Forecast Projection (Prophet vs ARIMA models)
with tab2:
    st.markdown("### Forecasting Models Comparison & Projection")
    
    if df_metric.empty:
        st.warning("No data found for the selected options.")
    else:
        # User control to choose models
        col_opts, col_spacer = st.columns([2, 3])
        with col_opts:
            models_to_show = st.multiselect(
                "Select Models to Display", 
                options=["Prophet", "ARIMA"], 
                default=["Prophet", "ARIMA"]
            )
            
        # Run forecasting pipeline (cached)
        forecasts = run_forecast_pipeline(df_metric)
        metrics = forecasts['metrics']
        
        # Create a customized dark-themed Plotly figure
        fig = go.Figure()
        
        # 1. Historical Actuals
        fig.add_trace(go.Scatter(
            x=df_metric['Date'],
            y=df_metric['Value'],
            mode='lines',
            name='Historical Actuals',
            line=dict(color='#a78bfa', width=2.5),
            hovertemplate='Date: %{x}<br>Actual: %{y:,.2f}<extra></extra>'
        ))
        
        # 2. Prophet traces
        if "Prophet" in models_to_show:
            p_val = forecasts['prophet_val']
            p_fut = forecasts['prophet_future']
            
            # Validation period (dashed line)
            fig.add_trace(go.Scatter(
                x=p_val['Date'],
                y=p_val['Value'],
                mode='lines',
                name='Prophet (Validation Forecast)',
                line=dict(color='#06b6d4', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Prophet Val: %{y:,.2f}<extra></extra>'
            ))
            
            # Future Forecast (solid line)
            fig.add_trace(go.Scatter(
                x=p_fut['Date'],
                y=p_fut['Value'],
                mode='lines',
                name='Prophet (90d Future Projection)',
                line=dict(color='#06b6d4', width=2.5),
                hovertemplate='Date: %{x}<br>Prophet Forecast: %{y:,.2f}<extra></extra>'
            ))
            
            # Confidence Interval shading
            fig.add_trace(go.Scatter(
                x=list(p_fut['Date']) + list(p_fut['Date'])[::-1],
                y=list(p_fut['Upper']) + list(p_fut['Lower'])[::-1],
                fill='toself',
                fillcolor='rgba(6, 182, 212, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='Prophet 95% CI'
            ))
            
        # 3. ARIMA traces
        if "ARIMA" in models_to_show:
            a_val = forecasts['arima_val']
            a_fut = forecasts['arima_future']
            
            # Validation period (dashed line)
            fig.add_trace(go.Scatter(
                x=a_val['Date'],
                y=a_val['Value'],
                mode='lines',
                name='ARIMA (Validation Forecast)',
                line=dict(color='#f97316', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>ARIMA Val: %{y:,.2f}<extra></extra>'
            ))
            
            # Future Forecast (solid line)
            fig.add_trace(go.Scatter(
                x=a_fut['Date'],
                y=a_fut['Value'],
                mode='lines',
                name='ARIMA (90d Future Projection)',
                line=dict(color='#f97316', width=2.5),
                hovertemplate='Date: %{x}<br>ARIMA Forecast: %{y:,.2f}<extra></extra>'
            ))
            
            # Confidence Interval shading
            fig.add_trace(go.Scatter(
                x=list(a_fut['Date']) + list(a_fut['Date'])[::-1],
                y=list(a_fut['Upper']) + list(a_fut['Lower'])[::-1],
                fill='toself',
                fillcolor='rgba(249, 115, 22, 0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='ARIMA 95% CI'
            ))
            
        # Style layout for premium UI
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title='Timeline'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title='Value'
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistical metrics comparison
        st.markdown("<h4 style='font-weight:600; margin-top:1.5rem;'>Model Accuracy Comparison (Validation Set)</h4>", unsafe_allow_html=True)
        st.markdown("Metrics are evaluated on the final 90-day held-out quarter against clean baseline targets:")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div style="background:rgba(6, 182, 212, 0.08); border-left:4px solid #06b6d4; padding:1rem; border-radius:4px;">
                <h5 style="margin:0 0 0.5rem 0; color:#06b6d4; font-weight:600;">🌀 Facebook Prophet</h5>
                <p style="margin:2px 0;"><b>MAE:</b> {metrics['prophet']['MAE']:,.2f}</p>
                <p style="margin:2px 0;"><b>RMSE:</b> {metrics['prophet']['RMSE']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div style="background:rgba(249, 115, 22, 0.08); border-left:4px solid #f97316; padding:1rem; border-radius:4px;">
                <h5 style="margin:0 0 0.5rem 0; color:#f97316; font-weight:600;">📈 ARIMA (SARIMAX)</h5>
                <p style="margin:2px 0;"><b>MAE:</b> {metrics['arima']['MAE']:,.2f}</p>
                <p style="margin:2px 0;"><b>RMSE:</b> {metrics['arima']['RMSE']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Summary Box
        better_model = "Prophet" if metrics['prophet']['MAE'] < metrics['arima']['MAE'] else "ARIMA"
        lower_mae = min(metrics['prophet']['MAE'], metrics['arima']['MAE'])
        st.success(f"🏆 **{better_model}** has a lower MAE of **{lower_mae:,.2f}** on the validation quarter, indicating it is currently the best fit for this metric's trend.")
        
        st.info("💡 Note: The Tabular Machine Learning Forecast (Random Forest) and interactive SHAP explainability will be integrated in the next slice (Issue #5).")

# TAB 3: Anomaly Dashboard (Stub)
with tab3:
    st.markdown("### Anomaly Detection Control Room")
    st.info("⚠️ Anomaly detection layers (Isolation Forest and Z-Score thresholding) are coming soon in Issue #6 and Issue #7.")

# TAB 4: Explainability (Stub)
with tab4:
    st.markdown("### Explainable AI Predictions (SHAP)")
    st.info("🧠 Interactive SHAP explanations and waterfall charts will be integrated in Issue #5.")
