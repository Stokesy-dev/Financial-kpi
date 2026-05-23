import pandas as pd
import numpy as np
import pytest
from data.generate_data import generate_synthetic_data
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics
from models.forecasting import train_and_forecast_all

def test_ml_forecasting_and_shap_pipeline(tmp_path):
    """
    Behavior: The forecasting engine must support training a Tabular Tree-based Regressor
    (Random Forest) alongside Prophet/ARIMA, evaluating its MAE/RMSE on the validation set,
    and generating SHAP explanation matrices (values, features, and base value) for future forecasts.
    """
    # 1. Generate 3 years of daily E-commerce revenue data
    db_file = tmp_path / "test_ml_forecast.db"
    db_path = str(db_file)
    init_db(db_path)
    
    df_seed = generate_synthetic_data(start_date="2023-01-01", end_date="2025-12-31")
    insert_transactions(db_path, df_seed)
    
    df_metric = query_aggregated_metrics(db_path, bu="ecommerce", metric="revenue", frequency="D")
    
    # Act
    forecast_results = train_and_forecast_all(df_metric, forecast_horizon=90)
    
    # Assert
    assert isinstance(forecast_results, dict)
    
    # Verify Random Forest forecast Series exist
    assert 'rf_val' in forecast_results
    assert 'rf_future' in forecast_results
    assert 'shap_data' in forecast_results
    
    df_rf_val = forecast_results['rf_val']
    df_rf_fut = forecast_results['rf_future']
    
    assert len(df_rf_val) == 90
    assert len(df_rf_fut) == 90
    assert list(df_rf_val.columns) == ['Date', 'Value', 'Lower', 'Upper']
    assert list(df_rf_fut.columns) == ['Date', 'Value', 'Lower', 'Upper']
    
    # Check that validation error metrics contain Random Forest (rf)
    metrics = forecast_results['metrics']
    assert 'rf' in metrics
    assert metrics['rf']['MAE'] > 0
    assert metrics['rf']['RMSE'] > 0
    
    # Verify SHAP data structure
    shap_data = forecast_results['shap_data']
    assert isinstance(shap_data, dict)
    assert 'base_value' in shap_data
    assert 'values' in shap_data
    assert 'data' in shap_data
    assert 'feature_names' in shap_data
    assert 'dates' in shap_data
    
    assert isinstance(shap_data['base_value'], (float, np.float32, np.float64))
    
    # Check shapes of SHAP values and data matrices (90 days x 7 features)
    shap_vals = np.array(shap_data['values'])
    shap_raw_data = np.array(shap_data['data'])
    
    assert shap_vals.shape == (90, 7)
    assert shap_raw_data.shape == (90, 7)
    
    # Check feature names are correct
    expected_features = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30', 'dayofweek', 'month']
    assert list(shap_data['feature_names']) == expected_features
    
    # Check dates are correct
    assert len(shap_data['dates']) == 90
    assert shap_data['dates'][0] == '2026-01-01'
    assert shap_data['dates'][-1] == '2026-03-31'
