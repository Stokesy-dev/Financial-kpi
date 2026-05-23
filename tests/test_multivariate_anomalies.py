import pandas as pd
import numpy as np
import pytest
from models.anomaly_detection import compute_multivariate_anomalies, compute_combined_anomalies

def test_multivariate_isolation_forest_execution():
    """
    Asserts that the multivariate Isolation Forest correctly runs on the 3D space 
    and handles different contamination thresholds.
    """
    # Create synthetic daily data for 100 days
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    dates_str = [d.strftime('%Y-%m-%d') for d in dates]
    
    df_rev = pd.DataFrame({'Date': dates_str, 'Value': [100.0 + np.random.normal(0, 5) for _ in range(100)]})
    df_cost = pd.DataFrame({'Date': dates_str, 'Value': [50.0 + np.random.normal(0, 2) for _ in range(100)]})
    df_vol = pd.DataFrame({'Date': dates_str, 'Value': [10.0 + np.random.normal(0, 1) for _ in range(100)]})
    
    # Inject one distinct spike on day 50 across all metrics
    df_rev.loc[50, 'Value'] = 1000.0
    df_cost.loc[50, 'Value'] = 500.0
    df_vol.loc[50, 'Value'] = 100.0
    
    # Run Isolation Forest standard (contamination = 0.03)
    df_mult = compute_multivariate_anomalies(df_rev, df_cost, df_vol, contamination=0.03)
    
    assert 'IForest_Anomaly' in df_mult
    assert 'Revenue' in df_mult
    assert 'Cost' in df_mult
    assert 'Volume' in df_mult
    assert len(df_mult) == 100
    
    # The injected anomaly on day 50 should be flagged
    assert df_mult.loc[50, 'IForest_Anomaly'] == 1
    
    # Run with strict contamination (0.01) vs lenient contamination (0.05)
    df_strict = compute_multivariate_anomalies(df_rev, df_cost, df_vol, contamination=0.01)
    df_lenient = compute_multivariate_anomalies(df_rev, df_cost, df_vol, contamination=0.05)
    
    assert df_strict['IForest_Anomaly'].sum() <= df_lenient['IForest_Anomaly'].sum()
    assert df_strict.loc[50, 'IForest_Anomaly'] == 1

def test_combined_confidence_score_rules():
    """
    Asserts that confidence ratings are assigned correctly based on univariate and multivariate flags:
    - High: detected by both Z-score and Isolation Forest.
    - Medium: detected by Isolation Forest only.
    - Low: detected by Z-score only.
    - None: detected by neither.
    """
    # Create simple mock series where we can force specific flags
    dates_str = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    
    # Baseline value of 100, standard deviation of 1
    # Day 0: Normal day (no flags)
    # Day 1: High confidence (flagged by Z-score and Isolation Forest) -> spike to 1000
    # Day 2: Medium confidence (flagged by Isolation Forest only) -> we'll simulate this by choosing custom thresholds
    # Day 3: Low confidence (flagged by Z-score only) -> we'll simulate this
    
    df_rev = pd.DataFrame({'Date': dates_str, 'Value': [100.0, 1000.0, 120.0, 150.0]})
    df_cost = pd.DataFrame({'Date': dates_str, 'Value': [50.0, 500.0, 50.0, 50.0]})
    df_vol = pd.DataFrame({'Date': dates_str, 'Value': [10.0, 100.0, 10.0, 10.0]})
    
    # Call the combined scoring function
    # Let's set z_threshold=3.0, contamination=0.25 (to force 1 flagged point in IForest)
    # With these parameters:
    # - Day 1 is a massive spike (1000), so Z-score flags it, and IForest flags it -> High
    # - Day 2 (120) is a minor change, shouldn't trigger Z-score (>3 std dev is >103), but IForest might flag it if contamination is high -> Medium
    # - Day 3 (150) triggers Z-score (>103), but let's see how IForest ranks it.
    df_comb = compute_combined_anomalies(df_rev, df_cost, df_vol, z_threshold=3.0, contamination=0.25)
    
    assert 'Confidence' in df_comb
    assert 'Is_Anomaly' in df_comb
    assert 'IForest_Anomaly' in df_comb
    assert 'Z_Anomaly' in df_comb
    
    # Verify the mappings for each date
    for _, row in df_comb.iterrows():
        z_flag = row['Z_Anomaly']
        if_flag = row['IForest_Anomaly']
        conf = row['Confidence']
        
        if z_flag == 1 and if_flag == 1:
            assert conf == 'High'
        elif if_flag == 1 and z_flag == 0:
            assert conf == 'Medium'
        elif z_flag == 1 and if_flag == 0:
            assert conf == 'Low'
        else:
            assert conf == 'None'
