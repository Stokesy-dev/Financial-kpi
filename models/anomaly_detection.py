import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

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

def compute_multivariate_anomalies(
    df_revenue: pd.DataFrame, 
    df_cost: pd.DataFrame, 
    df_volume: pd.DataFrame, 
    contamination: float = 0.03
) -> pd.DataFrame:
    """
    Runs Scikit-learn's IsolationForest on the combined 3D metric space (Revenue, Cost, Volume).
    
    Returns:
        pd.DataFrame with Date, Revenue, Cost, Volume, and IForest_Anomaly (0 or 1).
    """
    df_rev = df_revenue.rename(columns={'Value': 'Revenue'})[['Date', 'Revenue']]
    df_cst = df_cost.rename(columns={'Value': 'Cost'})[['Date', 'Cost']]
    df_vol = df_volume.rename(columns={'Value': 'Volume'})[['Date', 'Volume']]
    
    # Outer join to align all dates
    df_merged = df_rev.merge(df_cst, on='Date', how='outer').merge(df_vol, on='Date', how='outer')
    df_merged = df_merged.fillna(0.0)
    df_merged = df_merged.sort_values('Date').reset_index(drop=True)
    
    if len(df_merged) == 0:
        df_merged['IForest_Anomaly'] = []
        return df_merged
        
    X = df_merged[['Revenue', 'Cost', 'Volume']]
    
    # Fit Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    
    # -1 is anomaly, 1 is normal
    df_merged['IForest_Anomaly'] = (preds == -1).astype(int)
    
    return df_merged

def compute_combined_anomalies(
    df_revenue: pd.DataFrame,
    df_cost: pd.DataFrame,
    df_volume: pd.DataFrame,
    z_threshold: float = 3.0,
    contamination: float = 0.03
) -> pd.DataFrame:
    """
    Combines univariate moving Z-scores and multivariate Isolation Forest
    to compute combined Anomaly Confidence Scores: High, Medium, Low.
    """
    # 1. Run univariate Z-scores on each metric
    df_rev_z = compute_univariate_zscore_anomalies(df_revenue, window=30, threshold=z_threshold)
    df_cst_z = compute_univariate_zscore_anomalies(df_cost, window=30, threshold=z_threshold)
    df_vol_z = compute_univariate_zscore_anomalies(df_volume, window=30, threshold=z_threshold)
    
    # 2. Run multivariate Isolation Forest
    df_mult = compute_multivariate_anomalies(df_revenue, df_cost, df_volume, contamination=contamination)
    
    # 3. Rename flags for merging
    df_rev_flags = df_rev_z.rename(columns={'Is_Anomaly': 'Z_Rev_Anomaly', 'Z_Score': 'Z_Rev_Score'})[['Date', 'Z_Rev_Anomaly', 'Z_Rev_Score']]
    df_cst_flags = df_cst_z.rename(columns={'Is_Anomaly': 'Z_Cst_Anomaly', 'Z_Score': 'Z_Cst_Score'})[['Date', 'Z_Cst_Anomaly', 'Z_Cst_Score']]
    df_vol_flags = df_vol_z.rename(columns={'Is_Anomaly': 'Z_Vol_Anomaly', 'Z_Score': 'Z_Vol_Score'})[['Date', 'Z_Vol_Anomaly', 'Z_Vol_Score']]
    
    df_combined = df_mult.merge(df_rev_flags, on='Date', how='left')
    df_combined = df_combined.merge(df_cst_flags, on='Date', how='left')
    df_combined = df_combined.merge(df_vol_flags, on='Date', how='left')
    
    df_combined = df_combined.fillna(0.0)
    
    # Evaluate combined flags
    df_combined['Z_Anomaly'] = (
        (df_combined['Z_Rev_Anomaly'] == 1) |
        (df_combined['Z_Cst_Anomaly'] == 1) |
        (df_combined['Z_Vol_Anomaly'] == 1)
    ).astype(int)
    
    # Confidence Score Rules
    def get_confidence(row):
        z = row['Z_Anomaly']
        ifo = row['IForest_Anomaly']
        if z == 1 and ifo == 1:
            return 'High'
        elif ifo == 1 and z == 0:
            return 'Medium'
        elif z == 1 and ifo == 0:
            return 'Low'
        else:
            return 'None'
            
    df_combined['Confidence'] = df_combined.apply(get_confidence, axis=1)
    df_combined['Is_Anomaly'] = (df_combined['Confidence'] != 'None').astype(int)
    
    return df_combined
