import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data.generate_data import generate_synthetic_data
from models.forecasting import train_and_forecast_all
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics
from models.anomaly_detection import compute_univariate_zscore_anomalies

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
    """Initializes the database and seeds it with data if empty or outdated."""
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM transactions WHERE business_unit = 'saas' AND type = 'cost' AND amount > 500")
            has_anoms = c.fetchone()[0] > 0
            conn.close()
            
            if not has_anoms:
                # Outdated DB without anomalies, delete and recreate
                os.remove(db_path)
        except Exception:
            # If any database error occurs, clear it
            if os.path.exists(db_path):
                os.remove(db_path)
                
    if not os.path.exists(db_path):
        init_db(db_path)
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
    return train_and_forecast_all(df_data, forecast_horizon=90)

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

# TAB 2: Forecast Projection (Prophet, ARIMA, and Random Forest models)
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
                options=["Prophet", "ARIMA", "Random Forest (ML)"], 
                default=["Prophet", "ARIMA", "Random Forest (ML)"]
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
                name='Prophet (Validation)',
                line=dict(color='#06b6d4', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Prophet Val: %{y:,.2f}<extra></extra>'
            ))
            
            # Future Forecast (solid line)
            fig.add_trace(go.Scatter(
                x=p_fut['Date'],
                y=p_fut['Value'],
                mode='lines',
                name='Prophet (90d Projection)',
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
                name='ARIMA (Validation)',
                line=dict(color='#f97316', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>ARIMA Val: %{y:,.2f}<extra></extra>'
            ))
            
            # Future Forecast (solid line)
            fig.add_trace(go.Scatter(
                x=a_fut['Date'],
                y=a_fut['Value'],
                mode='lines',
                name='ARIMA (90d Projection)',
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
            
        # 4. Random Forest traces
        if "Random Forest (ML)" in models_to_show:
            r_val = forecasts['rf_val']
            r_fut = forecasts['rf_future']
            
            # Validation period (dashed line)
            fig.add_trace(go.Scatter(
                x=r_val['Date'],
                y=r_val['Value'],
                mode='lines',
                name='Random Forest (Validation)',
                line=dict(color='#10b981', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>RF Val: %{y:,.2f}<extra></extra>'
            ))
            
            # Future Forecast (solid line)
            fig.add_trace(go.Scatter(
                x=r_fut['Date'],
                y=r_fut['Value'],
                mode='lines',
                name='Random Forest (90d Projection)',
                line=dict(color='#10b981', width=2.5),
                hovertemplate='Date: %{x}<br>RF Forecast: %{y:,.2f}<extra></extra>'
            ))
            
            # Confidence Interval shading
            fig.add_trace(go.Scatter(
                x=list(r_fut['Date']) + list(r_fut['Date'])[::-1],
                y=list(r_fut['Upper']) + list(r_fut['Lower'])[::-1],
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='Random Forest 95% CI'
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
        
        col_m1, col_m2, col_m3 = st.columns(3)
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
            
        with col_m3:
            st.markdown(f"""
            <div style="background:rgba(16, 185, 129, 0.08); border-left:4px solid #10b981; padding:1rem; border-radius:4px;">
                <h5 style="margin:0 0 0.5rem 0; color:#10b981; font-weight:600;">🌲 Random Forest (ML)</h5>
                <p style="margin:2px 0;"><b>MAE:</b> {metrics['rf']['MAE']:,.2f}</p>
                <p style="margin:2px 0;"><b>RMSE:</b> {metrics['rf']['RMSE']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Summary Box
        model_maes = {
            'Prophet': metrics['prophet']['MAE'],
            'ARIMA': metrics['arima']['MAE'],
            'Random Forest (ML)': metrics['rf']['MAE']
        }
        best_model = min(model_maes, key=model_maes.get)
        lower_mae = model_maes[best_model]
        st.success(f"🏆 **{best_model}** has the lowest MAE of **{lower_mae:,.2f}** on the validation quarter, indicating it has the lowest prediction error.")

# TAB 3: Anomaly Dashboard (Moving Z-Score)
with tab3:
    st.markdown("### Anomaly Detection Control Room")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        st.markdown("<div style='background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        threshold_sel = st.selectbox(
            "Anomaly Sensitivity Threshold",
            options=["strict", "standard", "lenient"],
            index=1,
            format_func=lambda x: {
                "strict": "🔒 Strict (Z > 3.5)",
                "standard": "⚖️ Standard (Z > 3.0)",
                "lenient": "🔓 Lenient (Z > 2.0)"
            }.get(x, x),
            key="anomaly_threshold_select"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    threshold_map = {
        "strict": 3.5,
        "standard": 3.0,
        "lenient": 2.0
    }
    z_threshold = threshold_map[threshold_sel]
    
    # Load aggregated metric with reindexing enabled to catch zero-value days/weeks/months
    df_metric_reindexed = query_aggregated_metrics(DB_PATH, business_unit, metric, frequency, reindex_all_dates=True)
    
    if df_metric_reindexed.empty:
        st.warning("No data found for the selected options.")
    else:
        # Run univariate anomaly detection
        df_anom = compute_univariate_zscore_anomalies(df_metric_reindexed, window=30, threshold=z_threshold)
        anomalies_only = df_anom[df_anom['Is_Anomaly'] == 1]
        
        # Plotly time-series chart with flagged points highlight-colored
        fig_anom = go.Figure()
        
        # Line for actual values
        fig_anom.add_trace(go.Scatter(
            x=df_anom['Date'],
            y=df_anom['Value'],
            mode='lines',
            name='Actual Level',
            line=dict(color='#8b5cf6', width=2), # Violet theme for baseline
            hovertemplate="Date: %{x}<br>Value: %{y:,.2f}<extra></extra>"
        ))
        
        # Scatter markers for anomalies
        if not anomalies_only.empty:
            fig_anom.add_trace(go.Scatter(
                x=anomalies_only['Date'],
                y=anomalies_only['Value'],
                mode='markers',
                name='Anomaly Flagged',
                marker=dict(color='#ef4444', size=10, symbol='circle-open-dot', line=dict(width=2)),
                hovertemplate="🚨 <b>Anomaly Detected</b><br>Date: %{x}<br>Value: %{y:,.2f}<br>Z-Score: %{customdata:+.2f}<extra></extra>",
                customdata=anomalies_only['Z_Score']
            ))
            
        fig_anom.update_layout(
            title=dict(
                text=f"Historical Anomalies for {business_unit.upper()} {metric.upper()}",
                font=dict(size=18, family="Outfit")
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, title='Date'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Value'),
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_anom, use_container_width=True)
        
        # List anomalies in a clean search/table
        st.markdown("<h4 style='font-weight:600;'>Flagged Anomalies Log</h4>", unsafe_allow_html=True)
        if not anomalies_only.empty:
            st.markdown(f"Detected **{len(anomalies_only)}** anomaly events:")
            # Display summary table
            display_df = anomalies_only[['Date', 'Value', 'Z_Score']].copy()
            # Rename for display
            display_df.columns = ['Date', 'Aggregated Value', 'Univariate Z-Score']
            st.dataframe(display_df.style.format({
                'Aggregated Value': '{:,.2f}',
                'Univariate Z-Score': '{:+.2f}'
            }), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No anomalies detected for the selected period under this threshold.")

# TAB 4: Explainability (SHAP)
with tab4:
    st.markdown("### Explainable AI Predictions (SHAP Waterfall Plots)")
    
    if df_metric.empty:
        st.warning("No data found for the selected options.")
    else:
        # Load pre-computed forecasts containing SHAP data
        forecasts = run_forecast_pipeline(df_metric)
        shap_data = forecasts['shap_data']
        
        st.markdown("""
        To build trust and provide transparency, this explainability layer displays a **SHAP Waterfall Plot** for any selected date.
        The plot decomposes a single day's forecast to show how much each individual engineered feature (lags, rolling averages, calendar cycles) 
        shifted the prediction away from the model's base value (average forecast).
        """)
        
        # Date selection
        selected_date = st.selectbox(
            "Select Forecast Date to Analyze",
            options=shap_data['dates'],
            index=14  # Default to middle of first month
        )
        
        # Extract indices and values
        date_idx = shap_data['dates'].index(selected_date)
        vals = shap_data['values'][date_idx]
        raw_feats = shap_data['data'][date_idx]
        base_val = shap_data['base_value']
        feature_names = shap_data['feature_names']
        
        model_pred = base_val + sum(vals)
        
        # Sort features by absolute SHAP contribution
        sorted_indices = sorted(range(len(vals)), key=lambda k: abs(vals[k]), reverse=True)
        
        measure = ["absolute"]
        x_labels = ["Model Base (Avg)"]
        y_deltas = [base_val]
        text_labels = [f"{base_val:,.2f}"]
        
        for idx in sorted_indices:
            feat_name = feature_names[idx]
            feat_val = raw_feats[idx]
            shap_val = vals[idx]
            
            friendly_name = {
                'lag_1': 'Lag 1 Day',
                'lag_7': 'Lag 7 Days',
                'lag_30': 'Lag 30 Days',
                'rolling_mean_7': '7d Rolling Avg',
                'rolling_mean_30': '30d Rolling Avg',
                'dayofweek': 'Day of Week',
                'month': 'Month'
            }.get(feat_name, feat_name)
            
            # Map raw values to readable formats for day/month
            if feat_name == 'dayofweek':
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                raw_val_str = day_names[int(feat_val)]
            elif feat_name == 'month':
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                raw_val_str = month_names[int(feat_val) - 1]
            else:
                raw_val_str = f"{feat_val:,.2f}"
                
            measure.append("relative")
            x_labels.append(f"{friendly_name}<br>({raw_val_str})")
            y_deltas.append(shap_val)
            text_labels.append(f"{'+' if shap_val > 0 else ''}{shap_val:,.2f}")
            
        measure.append("total")
        x_labels.append("Final Forecast")
        y_deltas.append(0)
        text_labels.append(f"{model_pred:,.2f}")
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP Explainability",
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_deltas,
            text=text_labels,
            textposition="outside",
            connector=dict(line=dict(color="rgba(255,255,255,0.15)", width=1)),
            decreasing=dict(marker=dict(color="#ef4444")),  # Red for negative contributions
            increasing=dict(marker=dict(color="#10b981")),  # Green for positive contributions
            totals=dict(marker=dict(color="#8b5cf6"))        # Violet for final prediction
        ))
        
        fig_waterfall.update_layout(
            title=dict(
                text=f"SHAP Waterfall Explanation for {selected_date} Prediction",
                font=dict(size=18, family="Outfit")
            ),
            waterfallgap=0.2,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(tickangle=0, title="Prediction Drivers"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title='Forecast Value ($)' if metric in ['revenue', 'cost'] else 'Forecast Volume')
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Interpretative details
        st.markdown("#### 🔍 How to read this chart:")
        st.markdown(f"""
        - The waterfall begins at the **Model Base** value (**{base_val:,.2f}**), which is the average expected value forecasted by the model.
        - Each feature bar shows its local attribution. Green bars (**positive values**) pushed the forecast higher, while red bars (**negative values**) dragged the forecast lower.
        - For example, if **Lag 1 Day** is positive, it means yesterday's strong performance drove today's forecast up.
        - The **Final Forecast** value (**{model_pred:,.2f}**) is the sum of the base value and all attributions, representing the actual value displayed in the line chart.
        """)
