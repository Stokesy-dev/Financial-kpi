import pandas as pd
import pytest
from data.generate_data import generate_synthetic_data
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics
from models.forecasting import train_and_forecast_classical

def test_classical_forecasting_pipeline(tmp_path):
    """
    Behavior: The forecasting module must train Prophet and ARIMA models on historical data,
    evaluate MAE/RMSE on the final 90 days (held-out quarter), and generate future forecasts 
    with confidence intervals.
    """
    # 1. Generate 3 years of daily E-commerce revenue data to simulate a real scenario
    db_file = tmp_path / "test_forecast.db"
    db_path = str(db_file)
    init_db(db_path)
    
    df_seed = generate_synthetic_data(start_date="2023-01-01", end_date="2025-12-31")
    insert_transactions(db_path, df_seed)
    
    df_metric = query_aggregated_metrics(db_path, bu="ecommerce", metric="revenue", frequency="D")
    assert len(df_metric) == 1096  # 3 years including leap year (365*3 + 1 = 1096)
    
    # Act
    forecast_results = train_and_forecast_classical(df_metric, forecast_horizon=90)
    
    # Assert
    assert isinstance(forecast_results, dict)
    expected_keys = ['prophet_val', 'arima_val', 'metrics', 'prophet_future', 'arima_future']
    for key in expected_keys:
        assert key in forecast_results
        
    # Check validation forecasts (length 90)
    for model_val in ['prophet_val', 'arima_val']:
        df_val = forecast_results[model_val]
        assert isinstance(df_val, pd.DataFrame)
        assert len(df_val) == 90
        assert list(df_val.columns) == ['Date', 'Value', 'Lower', 'Upper']
        # The validation dates should cover the last 90 days of the dataset (Oct 3, 2025 to Dec 31, 2025)
        # Note: 1096 - 90 = 1006. The 1006th row date is '2025-10-03'.
        assert df_val['Date'].iloc[0] == '2025-10-03'
        assert df_val['Date'].iloc[-1] == '2025-12-31'
        
    # Check metrics (MAE and RMSE should be valid positive floats)
    metrics = forecast_results['metrics']
    assert 'prophet' in metrics
    assert 'arima' in metrics
    for model in ['prophet', 'arima']:
        assert 'MAE' in metrics[model]
        assert 'RMSE' in metrics[model]
        assert metrics[model]['MAE'] > 0
        assert metrics[model]['RMSE'] > 0
        
    # Check future forecasts (length 90, starting after Dec 31, 2025)
    for model_fut in ['prophet_future', 'arima_future']:
        df_fut = forecast_results[model_fut]
        assert isinstance(df_fut, pd.DataFrame)
        assert len(df_fut) == 90
        assert list(df_fut.columns) == ['Date', 'Value', 'Lower', 'Upper']
        assert df_fut['Date'].iloc[0] == '2026-01-01'
        assert df_fut['Date'].iloc[-1] == '2026-03-31'
