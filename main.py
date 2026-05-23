import os
import argparse
import pandas as pd
from data.generate_data import generate_synthetic_data
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics
from models.feature_engineering import create_forecasting_features

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
            
    elif args.mode in ["forecast", "anomaly"]:
        # Placeholders for future issues
        print(f"🛠️ Mode '{args.mode}' has been recognized.")
        print(f"   Business Unit: {args.bu}")
        print(f"   Metric       : {args.metric}")
        if args.mode == "anomaly":
            print(f"   Threshold    : {args.threshold}")
        print(f"🔮 Integration for modeling is coming in the next issues (Issue #4, #5, #6).")

if __name__ == "__main__":
    main()
