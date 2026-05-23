import logging
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Silence warning logs from Prophet and Statsmodels
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

def train_and_forecast_classical(df: pd.DataFrame, forecast_horizon: int = 90) -> dict:
    """
    Trains Facebook Prophet and ARIMA (SARIMAX) models.
    
    Validation:
      - Split df: train is first N - forecast_horizon days, val is last forecast_horizon days.
      - Fit models on train, predict validation period, and compute MAE and RMSE.
      
    Forecast:
      - Fit models on full df and predict next forecast_horizon days into the future.
      
    Returns:
      A dictionary with:
        'prophet_val': pd.DataFrame with ['Date', 'Value', 'Lower', 'Upper']
        'arima_val': pd.DataFrame with ['Date', 'Value', 'Lower', 'Upper']
        'metrics': dict with model errors
        'prophet_future': pd.DataFrame with ['Date', 'Value', 'Lower', 'Upper']
        'arima_future': pd.DataFrame with ['Date', 'Value', 'Lower', 'Upper']
    """
    # Sort chronologically to prevent temporal leaks
    df_sorted = df.sort_values('Date').copy()
    
    # Temporal Train/Validation Split
    df_train = df_sorted.iloc[:-forecast_horizon]
    df_val = df_sorted.iloc[-forecast_horizon:]
    val_dates = df_val['Date']
    
    # -----------------------------------------------------------------
    # PROPHET MODEL
    # -----------------------------------------------------------------
    # Fit train
    df_p_train = df_train.rename(columns={'Date': 'ds', 'Value': 'y'})
    m_prophet_train = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m_prophet_train.fit(df_p_train)
    
    # Predict validation period
    df_p_val_dates = pd.DataFrame({'ds': pd.to_datetime(val_dates)})
    forecast_val_p = m_prophet_train.predict(df_p_val_dates)
    
    prophet_val = pd.DataFrame({
        'Date': val_dates.values,
        'Value': forecast_val_p['yhat'].values,
        'Lower': forecast_val_p['yhat_lower'].values,
        'Upper': forecast_val_p['yhat_upper'].values
    })
    
    # Fit full dataset
    df_p_full = df_sorted.rename(columns={'Date': 'ds', 'Value': 'y'})
    m_prophet_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m_prophet_full.fit(df_p_full)
    
    # Predict future
    future_p = m_prophet_full.make_future_dataframe(periods=forecast_horizon, freq='D')
    future_dates_only_p = future_p.iloc[-forecast_horizon:]
    forecast_future_p = m_prophet_full.predict(future_dates_only_p)
    
    prophet_future = pd.DataFrame({
        'Date': forecast_future_p['ds'].dt.strftime('%Y-%m-%d').values,
        'Value': forecast_future_p['yhat'].values,
        'Lower': forecast_future_p['yhat_lower'].values,
        'Upper': forecast_future_p['yhat_upper'].values
    })

    # -----------------------------------------------------------------
    # ARIMA (SARIMAX) MODEL
    # -----------------------------------------------------------------
    # Fit train
    m_arima_train = SARIMAX(
        df_train['Value'].values, 
        order=(1, 1, 1), 
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_arima_train = m_arima_train.fit(disp=False)
    
    # Predict validation period
    pred_val_a = res_arima_train.get_forecast(steps=forecast_horizon)
    mean_val_a = pred_val_a.predicted_mean
    conf_val_a = pred_val_a.conf_int(alpha=0.05)  # 95% Confidence Interval
    
    arima_val = pd.DataFrame({
        'Date': val_dates.values,
        'Value': mean_val_a,
        'Lower': conf_val_a[:, 0] if isinstance(conf_val_a, np.ndarray) else conf_val_a.iloc[:, 0].values,
        'Upper': conf_val_a[:, 1] if isinstance(conf_val_a, np.ndarray) else conf_val_a.iloc[:, 1].values
    })
    
    # Fit full dataset
    m_arima_full = SARIMAX(
        df_sorted['Value'].values, 
        order=(1, 1, 1), 
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_arima_full = m_arima_full.fit(disp=False)
    
    # Predict future
    pred_future_a = res_arima_full.get_forecast(steps=forecast_horizon)
    mean_future_a = pred_future_a.predicted_mean
    conf_future_a = pred_future_a.conf_int(alpha=0.05)
    
    # Generate future calendar dates
    last_date = pd.to_datetime(df_sorted['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    
    arima_future = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d').values,
        'Value': mean_future_a,
        'Lower': conf_future_a[:, 0] if isinstance(conf_future_a, np.ndarray) else conf_future_a.iloc[:, 0].values,
        'Upper': conf_future_a[:, 1] if isinstance(conf_future_a, np.ndarray) else conf_future_a.iloc[:, 1].values
    })

    # -----------------------------------------------------------------
    # METRICS EVALUATION
    # -----------------------------------------------------------------
    val_actuals = df_val['Value'].values
    
    # Prophet error
    prophet_mae = np.mean(np.abs(val_actuals - prophet_val['Value'].values))
    prophet_rmse = np.sqrt(np.mean((val_actuals - prophet_val['Value'].values) ** 2))
    
    # ARIMA error
    arima_mae = np.mean(np.abs(val_actuals - arima_val['Value'].values))
    arima_rmse = np.sqrt(np.mean((val_actuals - arima_val['Value'].values) ** 2))
    
    metrics = {
        'prophet': {
            'MAE': float(prophet_mae),
            'RMSE': float(prophet_rmse)
        },
        'arima': {
            'MAE': float(arima_mae),
            'RMSE': float(arima_rmse)
        }
    }
    
    return {
        'prophet_val': prophet_val,
        'arima_val': arima_val,
        'metrics': metrics,
        'prophet_future': prophet_future,
        'arima_future': arima_future
    }
