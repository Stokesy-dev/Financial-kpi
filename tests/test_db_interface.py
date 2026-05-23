import os
import pandas as pd
import pytest
from database.db_interface import init_db, insert_transactions, query_aggregated_metrics

def test_database_lifecycle_and_aggregation(tmp_path):
    """
    Behavior: The database interface must support database schema initialization,
    transaction ingestion, and dynamic metric aggregation querying.
    """
    db_file = tmp_path / "test_financial_kpi.db"
    db_path = str(db_file)
    
    # Act: 1. Initialize
    init_db(db_path)
    assert os.path.exists(db_path)
    
    # Create known test transactions
    test_data = pd.DataFrame([
        {
            'TransactionID': 'TXN-001',
            'Date': '2023-01-01',
            'BusinessUnit': 'ecommerce',
            'Type': 'revenue',
            'Amount': 100.0,
            'Volume': 5
        },
        {
            'TransactionID': 'TXN-002',
            'Date': '2023-01-01',
            'BusinessUnit': 'ecommerce',
            'Type': 'revenue',
            'Amount': 250.0,
            'Volume': 10
        },
        {
            'TransactionID': 'TXN-003',
            'Date': '2023-01-02',
            'BusinessUnit': 'ecommerce',
            'Type': 'revenue',
            'Amount': 150.0,
            'Volume': 8
        }
    ])
    
    # Act: 2. Ingest
    insert_transactions(db_path, test_data)
    
    # Act: 3. Query daily revenue
    df_daily_rev = query_aggregated_metrics(db_path, bu='ecommerce', metric='revenue', frequency='D')
    
    # Assert daily revenue aggregation
    assert isinstance(df_daily_rev, pd.DataFrame)
    assert len(df_daily_rev) == 2
    
    # Check 2023-01-01 revenue sum (100.0 + 250.0 = 350.0)
    rev_jan_1 = df_daily_rev[df_daily_rev['Date'] == '2023-01-01']['Value'].values[0]
    assert rev_jan_1 == 350.0
    
    # Check 2023-01-02 revenue sum (150.0)
    rev_jan_2 = df_daily_rev[df_daily_rev['Date'] == '2023-01-02']['Value'].values[0]
    assert rev_jan_2 == 150.0
    
    # Act: 4. Query daily volume
    df_daily_vol = query_aggregated_metrics(db_path, bu='ecommerce', metric='volume', frequency='D')
    vol_jan_1 = df_daily_vol[df_daily_vol['Date'] == '2023-01-01']['Value'].values[0]
    assert vol_jan_1 == 15
