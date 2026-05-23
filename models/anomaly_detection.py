import pandas as pd
import numpy as np

def compute_univariate_zscore_anomalies(df: pd.DataFrame, window: int = 30, threshold: float = 3.0) -> pd.DataFrame:
    """
    Computes moving Z-scores for a time-series DataFrame with columns 'Date' and 'Value'.
    Flags anomalies where the absolute Z-score exceeds the threshold.
    
    Args:
        df: pd.DataFrame with 'Date' and 'Value' columns.
        window: Size of the rolling window.
        threshold: Z-score threshold for flagging an anomaly.
        
    Returns:
        pd.DataFrame with 'Z_Score' and 'Is_Anomaly' (0 or 1) columns added.
    """
    df_sorted = df.sort_values('Date').copy()
    
    if len(df_sorted) == 0:
        df_sorted['Z_Score'] = []
        df_sorted['Is_Anomaly'] = []
        return df_sorted
        
    # Calculate rolling mean and std
    rolling_mean = df_sorted['Value'].rolling(window=window, min_periods=7).mean()
    rolling_std = df_sorted['Value'].rolling(window=window, min_periods=7).std()
    
    # Fallback to cumulative expanding mean and std for warm-up period
    cum_mean = df_sorted['Value'].expanding().mean()
    cum_std = df_sorted['Value'].expanding().std()
    
    final_mean = rolling_mean.fillna(cum_mean).fillna(df_sorted['Value'])
    final_std = rolling_std.fillna(cum_std).fillna(0.0)
    
    # Calculate Z-scores (add small epsilon to std to prevent division by zero)
    z_scores = (df_sorted['Value'] - final_mean) / (final_std + 1e-8)
    
    # Flag anomalies
    is_anomaly = z_scores.abs() > threshold
    
    df_sorted['Z_Score'] = z_scores
    df_sorted['Is_Anomaly'] = is_anomaly.astype(int)
    
    return df_sorted
