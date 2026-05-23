import pandas as pd
import numpy as np
import pytest
from data.generate_data import generate_synthetic_data
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics
from models.anomaly_detection import compute_univariate_zscore_anomalies

def test_anomaly_injection(tmp_path):
    """
    Asserts that the database populated with synthetic data contains the 
    domain-specific anomalies at the specified date ranges.
    """
    db_file = tmp_path / "test_anom_gen.db"
    db_path = str(db_file)
    init_db(db_path)
    
    # Generate data with anomalies (default behaviour)
    df = generate_synthetic_data(start_date="2023-01-01", end_date="2025-12-31")
    insert_transactions(db_path, df)
    
    # 1. E-commerce Checkout Outage: 2024-06-10 to 2024-06-15
    df_ecom_rev = query_aggregated_metrics(db_path, bu="ecommerce", metric="revenue", frequency="D")
    df_outage = df_ecom_rev[(df_ecom_rev['Date'] >= '2024-06-10') & (df_ecom_rev['Date'] <= '2024-06-15')]
    # Typical baseline daily e-commerce revenue is ~1500; outage should dip it way below
    for val in df_outage['Value']:
        assert val < 300
        
    # 2. SaaS Infrastructure Cost Leak: 2024-10-05 to 2024-10-15
    df_saas_cost = query_aggregated_metrics(db_path, bu="saas", metric="cost", frequency="D")
    df_leak = df_saas_cost[(df_saas_cost['Date'] >= '2024-10-05') & (df_saas_cost['Date'] <= '2024-10-15')]
    # Typical baseline SaaS daily cost is ~$80; leak should spike it way above
    for val in df_leak['Value']:
        assert val > 800

    # 3. E-commerce Pricing Glitch: 2025-03-01 to 2025-03-05
    df_ecom_vol = query_aggregated_metrics(db_path, bu="ecommerce", metric="volume", frequency="D")
    df_glitch_vol = df_ecom_vol[(df_ecom_vol['Date'] >= '2025-03-01') & (df_ecom_vol['Date'] <= '2025-03-05')]
    df_glitch_rev = df_ecom_rev[(df_ecom_rev['Date'] >= '2025-03-01') & (df_ecom_rev['Date'] <= '2025-03-05')]
    # Pricing glitch should show a spike in volume but low revenue
    for vol in df_glitch_vol['Value']:
        assert vol > 100  # High volume (typical is ~50)
    for rev, vol in zip(df_glitch_rev['Value'], df_glitch_vol['Value']):
        # Average revenue per unit should be extremely low (~$2 instead of ~$30)
        assert (rev / vol) < 5.0

    # 4. Enterprise Services Contract Delay: 2025-08-01 to 2025-08-31
    df_ent_rev = query_aggregated_metrics(db_path, bu="enterprise", metric="revenue", frequency="D")
    df_delay = df_ent_rev[(df_ent_rev['Date'] >= '2025-08-01') & (df_ent_rev['Date'] <= '2025-08-31')]
    # No deals closed during contract delay month, so daily sum should be zero
    assert len(df_delay) == 0 or df_delay['Value'].sum() == 0

    # Assert daily Z-score detection correctly flags E-commerce Checkout Outage start
    df_ecom_detected = compute_univariate_zscore_anomalies(df_ecom_rev, window=30, threshold=3.0)
    outage_first_day = df_ecom_detected[df_ecom_detected['Date'] == '2024-06-10']
    assert outage_first_day['Is_Anomaly'].values[0] == 1

    # Assert daily Z-score detection correctly flags SaaS Cost Leak start
    df_saas_detected = compute_univariate_zscore_anomalies(df_saas_cost, window=30, threshold=3.0)
    leak_first_day = df_saas_detected[df_saas_detected['Date'] == '2024-10-05']
    assert leak_first_day['Is_Anomaly'].values[0] == 1

def test_univariate_zscore_flagging():
    """
    Asserts that moving Z-score correctly flags cost leak anomaly and handles thresholds.
    """
    # Create artificial daily data containing a cost leak spike
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    values = [80.0 + np.random.normal(0, 5) for _ in range(100)]
    # Inject cost leak at day 50
    values[50] = 1000.0
    
    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Value': values
    })
    
    # Test standard threshold (3.0)
    df_detected_std = compute_univariate_zscore_anomalies(df, window=30, threshold=3.0)
    assert 'Z_Score' in df_detected_std
    assert 'Is_Anomaly' in df_detected_std
    
    # The anomaly at day 50 should be flagged
    assert df_detected_std.loc[50, 'Is_Anomaly'] == 1
    # Check that normal days are not flagged as anomalies
    assert df_detected_std.loc[:49, 'Is_Anomaly'].sum() == 0
    
    # Test strict threshold (3.5)
    df_detected_strict = compute_univariate_zscore_anomalies(df, window=30, threshold=3.5)
    assert df_detected_strict.loc[50, 'Is_Anomaly'] == 1
    
    # Test lenient threshold (2.0)
    df_detected_lenient = compute_univariate_zscore_anomalies(df, window=30, threshold=2.0)
    assert df_detected_lenient.loc[50, 'Is_Anomaly'] == 1
