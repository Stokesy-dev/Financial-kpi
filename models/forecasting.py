import logging
import warnings
import numpy as np
import pandas as pd
import shap
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from models.feature_engineering import create_forecasting_features

# Silence warning logs
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

def train_and_forecast_classical(df: pd.DataFrame, forecast_horizon: int = 90) -> dict:
    """
    Trains Facebook Prophet and ARIMA (SARIMAX) models on historical data.
    Evaluates MAE/RMSE on the final held-out quarter, and predicts the next 90 days.
    """
    df_sorted = df.sort_values('Date').copy()
    df_train = df_sorted.iloc[:-forecast_horizon]
    df_val = df_sorted.iloc[-forecast_horizon:]
    val_dates = df_val['Date']
    
    # -------------------------------------------------------------
    # PROPHET MODEL
    # -------------------------------------------------------------
    df_p_train = df_train.rename(columns={'Date': 'ds', 'Value': 'y'})
    m_prophet_train = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m_prophet_train.fit(df_p_train)
    
    df_p_val_dates = pd.DataFrame({'ds': pd.to_datetime(val_dates)})
    forecast_val_p = m_prophet_train.predict(df_p_val_dates)
    
    prophet_val = pd.DataFrame({
        'Date': val_dates.values,
        'Value': forecast_val_p['yhat'].values,
        'Lower': forecast_val_p['yhat_lower'].values,
        'Upper': forecast_val_p['yhat_upper'].values
    })
    
    df_p_full = df_sorted.rename(columns={'Date': 'ds', 'Value': 'y'})
    m_prophet_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m_prophet_full.fit(df_p_full)
    
    future_p = m_prophet_full.make_future_dataframe(periods=forecast_horizon, freq='D')
    future_dates_only_p = future_p.iloc[-forecast_horizon:]
    forecast_future_p = m_prophet_full.predict(future_dates_only_p)
    
    prophet_future = pd.DataFrame({
        'Date': forecast_future_p['ds'].dt.strftime('%Y-%m-%d').values,
        'Value': forecast_future_p['yhat'].values,
        'Lower': forecast_future_p['yhat_lower'].values,
        'Upper': forecast_future_p['yhat_upper'].values
    })

    # -------------------------------------------------------------
    # ARIMA MODEL
    # -------------------------------------------------------------
    m_arima_train = SARIMAX(
        df_train['Value'].values, 
        order=(1, 1, 1), 
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_arima_train = m_arima_train.fit(disp=False)
    
    pred_val_a = res_arima_train.get_forecast(steps=forecast_horizon)
    mean_val_a = pred_val_a.predicted_mean
    conf_val_a = pred_val_a.conf_int(alpha=0.05)
    
    arima_val = pd.DataFrame({
        'Date': val_dates.values,
        'Value': mean_val_a,
        'Lower': conf_val_a[:, 0] if isinstance(conf_val_a, np.ndarray) else conf_val_a.iloc[:, 0].values,
        'Upper': conf_val_a[:, 1] if isinstance(conf_val_a, np.ndarray) else conf_val_a.iloc[:, 1].values
    })
    
    m_arima_full = SARIMAX(
        df_sorted['Value'].values, 
        order=(1, 1, 1), 
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_arima_full = m_arima_full.fit(disp=False)
    
    pred_future_a = res_arima_full.get_forecast(steps=forecast_horizon)
    mean_future_a = pred_future_a.predicted_mean
    conf_future_a = pred_future_a.conf_int(alpha=0.05)
    
    last_date = pd.to_datetime(df_sorted['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    
    arima_future = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d').values,
        'Value': mean_future_a,
        'Lower': conf_future_a[:, 0] if isinstance(conf_future_a, np.ndarray) else conf_future_a.iloc[:, 0].values,
        'Upper': conf_future_a[:, 1] if isinstance(conf_future_a, np.ndarray) else conf_future_a.iloc[:, 1].values
    })

    # Error Metrics
    val_actuals = df_val['Value'].values
    prophet_mae = np.mean(np.abs(val_actuals - prophet_val['Value'].values))
    prophet_rmse = np.sqrt(np.mean((val_actuals - prophet_val['Value'].values) ** 2))
    arima_mae = np.mean(np.abs(val_actuals - arima_val['Value'].values))
    arima_rmse = np.sqrt(np.mean((val_actuals - arima_val['Value'].values) ** 2))
    
    return {
        'prophet_val': prophet_val,
        'arima_val': arima_val,
        'metrics': {
            'prophet': {'MAE': float(prophet_mae), 'RMSE': float(prophet_rmse)},
            'arima': {'MAE': float(arima_mae), 'RMSE': float(arima_rmse)}
        },
        'prophet_future': prophet_future,
        'arima_future': arima_future
    }

def train_and_forecast_all(df: pd.DataFrame, forecast_horizon: int = 90) -> dict:
    """
    Trains Prophet, ARIMA, and a Tabular Random Forest Regressor.
    Generates local SHAP explainability matrices for the Random Forest future predictions.
    
    Returns:
        Unified dictionary containing validation metrics, forecast paths, and SHAP data.
    """
    # 1. Get Prophet & ARIMA forecasts
    results = train_and_forecast_classical(df, forecast_horizon)
    
    # Sort data chronologically
    df_sorted = df.sort_values('Date').copy()
    
    # 2. Extract features
    X, y = create_forecasting_features(df_sorted)
    
    # Drop rows containing NaNs (warm-up periods and end of series target shift)
    df_aligned = pd.concat([X, y], axis=1).dropna()
    X_aligned = df_aligned[X.columns]
    y_aligned = df_aligned[y.name]
    
    # -------------------------------------------------------------
    # RANDOM FOREST: VALIDATION RUN
    # -------------------------------------------------------------
    # Temporal Train/Validation Split on aligned features
    X_train = X_aligned.iloc[:-forecast_horizon]
    y_train = y_aligned.iloc[:-forecast_horizon]
    X_val = X_aligned.iloc[-forecast_horizon:]
    y_val = y_aligned.iloc[-forecast_horizon:]
    
    # Fit Random Forest on training set
    rf_train = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_train.fit(X_train, y_train)
    
    # Predict validation period
    y_val_pred = rf_train.predict(X_val)
    
    # Calculate confidence intervals (1.96 * standard deviation of tree predictions)
    tree_val_preds = np.array([tree.predict(X_val.values) for tree in rf_train.estimators_])
    std_val_preds = np.std(tree_val_preds, axis=0)
    lower_val_rf = y_val_pred - 1.96 * std_val_preds
    upper_val_rf = y_val_pred + 1.96 * std_val_preds
    
    # Extract validation dates matching y_val
    val_indices = y_val.index
    val_dates_rf = df_sorted.loc[val_indices, 'Date'].values
    
    rf_val = pd.DataFrame({
        'Date': val_dates_rf,
        'Value': y_val_pred,
        'Lower': np.maximum(0.0, lower_val_rf),
        'Upper': upper_val_rf
    })
    
    # Calculate validation metrics
    rf_mae = np.mean(np.abs(y_val.values - y_val_pred))
    rf_rmse = np.sqrt(np.mean((y_val.values - y_val_pred) ** 2))
    results['metrics']['rf'] = {'MAE': float(rf_mae), 'RMSE': float(rf_rmse)}
    results['rf_val'] = rf_val
    
    # -------------------------------------------------------------
    # RANDOM FOREST: FULL RUN & RECURSIVE FORECAST
    # -------------------------------------------------------------
    # Fit model on the full aligned dataset
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_full.fit(X_aligned, y_aligned)
    
    # Run recursive forecasting to generate future predictions & future feature vectors
    history_vals = list(df_sorted['Value'].values)
    history_dates = list(pd.to_datetime(df_sorted['Date'].values))
    
    predictions_fut = []
    lowers_fut = []
    uppers_fut = []
    future_dates = []
    
    # Keep track of the feature vectors constructed at each step of future predictions
    future_features = []
    current_date = history_dates[-1]
    
    for step in range(forecast_horizon):
        next_date = current_date + pd.Timedelta(days=1)
        future_dates.append(next_date)
        
        # Calculate lags and rolling means chronologically (ensuring zero look-ahead bias)
        lag_1 = history_vals[-1]
        lag_7 = history_vals[-7]
        lag_30 = history_vals[-30]
        rolling_mean_7 = np.mean(history_vals[-7:])
        rolling_mean_30 = np.mean(history_vals[-30:])
        dayofweek = next_date.dayofweek
        month = next_date.month
        
        # Store future feature vector
        feat_dict = {
            'lag_1': lag_1,
            'lag_7': lag_7,
            'lag_30': lag_30,
            'rolling_mean_7': rolling_mean_7,
            'rolling_mean_30': rolling_mean_30,
            'dayofweek': dayofweek,
            'month': month
        }
        future_features.append(feat_dict)
        
        # Format as input DataFrame row
        X_step = pd.DataFrame([feat_dict])
        
        # Predict next value
        pred_val = rf_full.predict(X_step)[0]
        
        # Calculate prediction intervals
        tree_step_preds = np.array([tree.predict(X_step.values)[0] for tree in rf_full.estimators_])
        std_step_pred = np.std(tree_step_preds)
        lower_step = pred_val - 1.96 * std_step_pred
        upper_step = pred_val + 1.96 * std_step_pred
        
        predictions_fut.append(pred_val)
        lowers_fut.append(max(0.0, lower_step))
        uppers_fut.append(upper_step)
        
        # Append back into our history to enable autoregressive lags for the next step!
        history_vals.append(pred_val)
        current_date = next_date
        
    rf_future = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Value': predictions_fut,
        'Lower': lowers_fut,
        'Upper': uppers_fut
    })
    results['rf_future'] = rf_future
    
    # -------------------------------------------------------------
    # SHAP EXPLAINABILITY (ON THE FULL RF FUTURE PROJECTIONS)
    # -------------------------------------------------------------
    X_future_df = pd.DataFrame(future_features)
    
    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf_full)
    shap_ex = explainer(X_future_df)
    
    # Extract scalar base value
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)
        
    results['shap_data'] = {
        'base_value': base_val,
        'values': shap_ex.values.tolist(),
        'data': shap_ex.data.tolist(),
        'feature_names': list(X.columns),
        'dates': rf_future['Date'].tolist()
    }
    
    return results
