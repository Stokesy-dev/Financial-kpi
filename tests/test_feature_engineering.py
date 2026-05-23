import pandas as pd
import numpy as np
import pytest
from models.feature_engineering import create_forecasting_features

def test_feature_engineering_structure_and_leakage():
    """
    Behavior: The feature engineering module must transform daily time-series metrics into
    tabular features containing lags, rolling means, and seasonal indicators, while aligning
    the target for next-day forecasting (t+1) with zero look-ahead bias.
    """
    # Create 45 days of sequential daily values: [1, 2, ..., 45]
    dates = pd.date_range(start="2023-01-01", periods=45, freq="D")
    df = pd.DataFrame({
        'Date': dates,
        'Value': np.arange(1.0, 46.0)
    })
    
    # Act
    X, y = create_forecasting_features(
        df, 
        target_col='Value', 
        lag_days=[1, 7, 30], 
        rolling_windows=[7, 30]
    )
    
    # Assert
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # Length check: before dropping NaNs, the shape should match the input
    assert len(X) == len(df)
    assert len(y) == len(df)
    
    # Check expected feature columns
    expected_cols = [
        'lag_1', 'lag_7', 'lag_30', 
        'rolling_mean_7', 'rolling_mean_30', 
        'dayofweek', 'month'
    ]
    for col in expected_cols:
        assert col in X.columns
        
    # Check feature values at a specific index to verify lag logic (zero look-ahead bias)
    # Let's check index 35 (corresponding to 2023-02-06, which is the 36th day)
    # The input Value series at index 35 is 36.0.
    # Therefore, lag_1 at index 35 must be the value on the preceding day (index 34, value 35.0).
    assert X.loc[35, 'lag_1'] == 35.0
    
    # lag_7 at index 35 must be the value 7 days prior (index 28, value 29.0).
    assert X.loc[35, 'lag_7'] == 29.0
    
    # lag_30 at index 35 must be the value 30 days prior (index 5, value 6.0).
    assert X.loc[35, 'lag_30'] == 6.0
    
    # Check rolling mean calculation: rolling_mean_7 at index 35
    # Must be the average of Values from index 28 to 34 (29.0 to 35.0).
    # Sum: 29+30+31+32+33+34+35 = 224. Mean: 224 / 7 = 32.0.
    assert X.loc[35, 'rolling_mean_7'] == 32.0
    
    # Check target alignment: y is the next-day's value (t+1) to predict.
    # So y at index 35 (current day t) must represent the actual value at index 36 (value 37.0).
    assert y.loc[35] == 37.0
    
    # Check seasonal indicators
    # 2023-02-05 is a Sunday (dayofweek = 6)
    assert X.loc[35, 'dayofweek'] == 6
    assert X.loc[35, 'month'] == 2
    
    # Check behavior of dropping NaNs (which simulates real training data preparation)
    # The first 30 rows should have NaNs due to lag_30.
    # The last row should have NaN for y because there is no t+1 value for the final day.
    # Therefore, dropping NaNs should yield (total_days - max_lag - 1) rows.
    # 45 - 30 - 1 = 14 rows.
    combined = pd.concat([X, y], axis=1).dropna()
    assert len(combined) == 14
    assert not combined.isnull().any().any()
