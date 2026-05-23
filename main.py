import os
import argparse
import pandas as pd
from data.generate_data import generate_synthetic_data
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics
from models.forecasting import train_and_forecast_all
from models.feature_engineering import create_forecasting_features
from models.anomaly_detection import compute_univariate_zscore_anomalies

def main():
    parser = argparse.ArgumentParser(description="Financial KPI Pipeline CLI")
    
    parser.add_argument(
        "--mode", 
        choices=["generate", "forecast", "anomaly"], 
        required=True,
        help="Pipeline execution mode. 'generate' initializes and seeds the database."
    )
    
    parser.add_argument(
        "--db", 
        default="financial_kpi.db", 
        help="Path to local SQLite database file."
    )
    
    parser.add_argument(
        "--start", 
        default="2023-01-01", 
        help="Start date for synthetic data generation (YYYY-MM-DD)."
    )
    
    parser.add_argument(
        "--end", 
        default="2025-12-31", 
        help="End date for synthetic data generation (YYYY-MM-DD)."
    )
    
    # Flags for future modeling slices (to avoid breaking CLI interface schemas)
    parser.add_argument(
        "--bu", 
        choices=["ecommerce", "saas", "enterprise"],
        help="Business Unit selection for forecasting/anomaly detection."
    )
    
    parser.add_argument(
        "--metric", 
        choices=["revenue", "cost", "volume"],
        help="Financial Metric selection for forecasting/anomaly detection."
    )
    
    parser.add_argument(
        "--threshold", 
        choices=["strict", "standard", "lenient"],
        help="Anomaly detection sensitivity threshold."
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        print(f"🚀 Initializing database: {args.db}")
        init_db(args.db)
        
        print(f"📊 Generating synthetic transactions from {args.start} to {args.end}...")
        df = generate_synthetic_data(args.start, args.end)
        print(f"✅ Generated {len(df):,} transaction-level records.")
        
        print("📥 Ingesting transactions into SQLite...")
        insert_transactions(args.db, df)
        print("✅ Ingestion complete.")
        
        # Print feature engineering diagnostics
        print("\n🔧 Running Feature Engineering Pipeline Diagnostics...")
        # Extract E-commerce Revenue metrics as a diagnostic sample
        df_ecom_rev = query_aggregated_metrics(args.db, bu="ecommerce", metric="revenue", frequency="D")
        
        if not df_ecom_rev.empty:
            X, y = create_forecasting_features(df_ecom_rev)
            print(f"   Aggregated Daily E-commerce Revenue shape: {df_ecom_rev.shape}")
            print(f"   Engineered Feature Matrix (X) shape     : {X.shape}")
            print(f"   Target Forecasting Vector (y) shape     : {y.shape}")
            print(f"   Features generated: {list(X.columns)}")
            
            # Show diagnostic output (first non-NaN rows)
            combined = pd.concat([X, y], axis=1).dropna()
            print(f"   Clean aligned training dataset rows     : {len(combined)} (warm-up periods discarded)")
            print("\n🔍 Sample Feature Matrix (X) Head:")
            print(combined.head(3).to_string())
        else:
            print("❌ Diagnostic error: Could not query aggregated metrics from database.")
            
    elif args.mode == "forecast":
        if not args.bu or not args.metric:
            parser.error("--bu and --metric are required when --mode is 'forecast'.")
            
        print(f"🔮 Loading data for Business Unit: '{args.bu}', Metric: '{args.metric}'...")
        if not os.path.exists(args.db):
            print(f"❌ Error: Database file '{args.db}' not found. Run --mode generate first.")
            return
            
        df_metric = query_aggregated_metrics(args.db, bu=args.bu, metric=args.metric, frequency="D")
        if df_metric.empty:
            print("❌ Error: No records found in database matching criteria.")
            return
            
        print("🧠 Fitting Prophet, ARIMA, and Tabular Random Forest models...")
        forecast_results = train_and_forecast_all(df_metric, forecast_horizon=90)
        
        # Print metrics
        print("\n📈 Model Comparison Performance Metrics (Held-out Quarter Validation):")
        metrics = forecast_results['metrics']
        print(f"   Prophet MAE : {metrics['prophet']['MAE']:,.2f} | RMSE: {metrics['prophet']['RMSE']:,.2f}")
        print(f"   ARIMA MAE   : {metrics['arima']['MAE']:,.2f} | RMSE: {metrics['arima']['RMSE']:,.2f}")
        print(f"   RF MAE      : {metrics['rf']['MAE']:,.2f} | RMSE: {metrics['rf']['RMSE']:,.2f}")
        
        # Display sample future forecast
        print("\n🔮 Sample Future Forecast (Next 90 Days starting 2026-01-01):")
        p_fut = forecast_results['prophet_future'].head(3)
        a_fut = forecast_results['arima_future'].head(3)
        rf_fut = forecast_results['rf_future'].head(3)
        print("   --- Prophet Forecast ---")
        print(p_fut.to_string(index=False))
        print("   --- ARIMA Forecast ---")
        print(a_fut.to_string(index=False))
        print("   --- Random Forest Forecast ---")
        print(rf_fut.to_string(index=False))
        
    elif args.mode == "anomaly":
        if not os.path.exists(args.db):
            print(f"❌ Error: Database file '{args.db}' not found. Run --mode generate first.")
            return

        threshold_map = {
            'strict': 3.5,
            'standard': 3.0,
            'lenient': 2.0
        }
        threshold_val = threshold_map.get(args.threshold or 'standard', 3.0)
        
        bus = [args.bu] if args.bu else ["ecommerce", "saas", "enterprise"]
        metrics = [args.metric] if args.metric else ["revenue", "cost", "volume"]
        
        print(f"⚠️ Running Anomaly Detection Pipeline (Threshold: {args.threshold or 'standard'} [Z-Score > {threshold_val}])...")
        
        total_anomalies = 0
        for bu in bus:
            for metric in metrics:
                df_metric = query_aggregated_metrics(args.db, bu=bu, metric=metric, frequency="D", reindex_all_dates=True)
                if df_metric.empty:
                    continue
                
                df_detected = compute_univariate_zscore_anomalies(df_metric, window=30, threshold=threshold_val)
                anomalies = df_detected[df_detected['Is_Anomaly'] == 1]
                
                if not anomalies.empty:
                    print(f"\n🚨 Business Unit: '{bu.upper()}', Metric: '{metric.upper()}' ({len(anomalies)} flagged):")
                    for _, row in anomalies.iterrows():
                        print(f"   Date: {row['Date']} | Value: {row['Value']:,.2f} | Z-Score: {row['Z_Score']:+.2f}")
                    total_anomalies += len(anomalies)
                    
        if total_anomalies == 0:
            print("✅ No anomalies detected matching the threshold criteria.")

if __name__ == "__main__":
    main()
