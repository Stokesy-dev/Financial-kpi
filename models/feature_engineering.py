import pandas as pd
import numpy as np

def create_forecasting_features(
    df: pd.DataFrame, 
    target_col: str = 'Value', 
    lag_days: list = [1, 7, 30], 
    rolling_windows: list = [7, 30]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transforms a daily aggregated time series into tabular features.
    Features on day t use data from day t-1 and prior (zero look-ahead bias).
    The target y represents the value on day t+1.
    """
    # Sort by date to ensure proper temporal ordering
    df_sorted = df.sort_values('Date').copy()
    
    # Create the historical series (shifted by 1 day) to use as the base for rolling means.
    # This prevents any look-ahead bias on day t.
    history = df_sorted[target_col].shift(1)
    
    X = pd.DataFrame(index=df_sorted.index)
    
    # Generate lag features
    for lag in lag_days:
        X[f'lag_{lag}'] = df_sorted[target_col].shift(lag)
        
    # Generate rolling mean features (computed on history to ensure no look-ahead bias)
    for window in rolling_windows:
        X[f'rolling_mean_{window}'] = history.rolling(window).mean()
        
    # Add seasonal indicators from Date
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    X['dayofweek'] = df_sorted['Date'].dt.dayofweek
    X['month'] = df_sorted['Date'].dt.month
    
    # Align the target vector y to represent t+1 (next day's value)
    y = df_sorted[target_col].shift(-1)
    
    return X, y
