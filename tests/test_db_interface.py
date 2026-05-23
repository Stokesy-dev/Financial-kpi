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


def test_db_multi_bu_and_time_aggregations(tmp_path):
    """
    Behavior: SQLite queries must correctly aggregate across different Business Units (SaaS, Enterprise, E-commerce)
    and granularities (Daily, Weekly, Monthly) without cross-contamination.
    """
    db_file = tmp_path / "test_multi_kpi.db"
    db_path = str(db_file)
    
    init_db(db_path)
    
    # Ingest mix of BUs, metrics, and dates
    # Note: 2023-01-01 is a Sunday, 2023-01-02 is a Monday, 2023-01-08 is a Sunday
    mix_data = pd.DataFrame([
        # SaaS
        {'TransactionID': 'T-01', 'Date': '2023-01-01', 'BusinessUnit': 'saas', 'Type': 'revenue', 'Amount': 99.0, 'Volume': 1},
        {'TransactionID': 'T-02', 'Date': '2023-01-01', 'BusinessUnit': 'saas', 'Type': 'cost', 'Amount': 20.0, 'Volume': 1},
        {'TransactionID': 'T-03', 'Date': '2023-01-02', 'BusinessUnit': 'saas', 'Type': 'revenue', 'Amount': 99.0, 'Volume': 1},
        # Enterprise
        {'TransactionID': 'T-04', 'Date': '2023-01-01', 'BusinessUnit': 'enterprise', 'Type': 'revenue', 'Amount': 20000.0, 'Volume': 1},
        {'TransactionID': 'T-05', 'Date': '2023-01-08', 'BusinessUnit': 'enterprise', 'Type': 'revenue', 'Amount': 30000.0, 'Volume': 1},
        # E-commerce Cost
        {'TransactionID': 'T-06', 'Date': '2023-01-01', 'BusinessUnit': 'ecommerce', 'Type': 'cost', 'Amount': 70.0, 'Volume': 1}
    ])
    
    insert_transactions(db_path, mix_data)
    
    # Assert 1: SaaS Monthly Revenue (aggregates 99.0 + 99.0 = 198.0)
    df_saas_m = query_aggregated_metrics(db_path, bu='saas', metric='revenue', frequency='M')
    assert len(df_saas_m) == 1
    assert df_saas_m['Date'].values[0] == '2023-01-01'  # Start of Month
    assert df_saas_m['Value'].values[0] == 198.0
    
    # Assert 2: SaaS Daily Cost (20.0)
    df_saas_c = query_aggregated_metrics(db_path, bu='saas', metric='cost', frequency='D')
    assert len(df_saas_c) == 1
    assert df_saas_c['Date'].values[0] == '2023-01-01'
    assert df_saas_c['Value'].values[0] == 20.0
    
    # Assert 3: Enterprise Weekly Revenue
    # Week starts: 2023-01-01 -> Monday 2022-12-26, 2023-01-08 -> Monday 2023-01-02
    df_ent_w = query_aggregated_metrics(db_path, bu='enterprise', metric='revenue', frequency='W')
    assert len(df_ent_w) == 2
    
    val_w1 = df_ent_w[df_ent_w['Date'] == '2022-12-26']['Value'].values[0]
    assert val_w1 == 20000.0
    
    val_w2 = df_ent_w[df_ent_w['Date'] == '2023-01-02']['Value'].values[0]
    assert val_w2 == 30000.0
    
    # Assert 4: E-commerce Cost Daily (70.0)
    df_ecom_c = query_aggregated_metrics(db_path, bu='ecommerce', metric='cost', frequency='D')
    assert len(df_ecom_c) == 1
    assert df_ecom_c['Value'].values[0] == 70.0

