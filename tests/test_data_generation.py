import pandas as pd
import pytest
from data.generate_data import generate_synthetic_data

def test_generate_synthetic_data_schema_and_range():
    """
    Behavior: The generator must return a DataFrame containing daily transaction-level records
    with the correct columns, correct date ranges, and valid non-null metric values.
    """
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Act
    df = generate_synthetic_data(start_date, end_date)
    
    # Assert
    assert isinstance(df, pd.DataFrame)
    
    # Check schema
    expected_columns = ['TransactionID', 'Date', 'BusinessUnit', 'Type', 'Amount', 'Volume']
    for col in expected_columns:
        assert col in df.columns
        
    # Check date range boundary
    df['Date'] = pd.to_datetime(df['Date'])
    assert df['Date'].min() >= pd.to_datetime(start_date)
    assert df['Date'].max() <= pd.to_datetime(end_date)
    
    # Check values are populated and types are correct
    assert not df['TransactionID'].isnull().any()
    assert (df['Amount'] >= 0).all()
    assert (df['Volume'] >= 0).all()
    
    # Verify BUs and Types are within allowed vocabulary
    allowed_bus = {'ecommerce', 'saas', 'enterprise'}
    allowed_types = {'revenue', 'cost'}
    assert set(df['BusinessUnit'].unique()).issubset(allowed_bus)
    assert set(df['Type'].unique()).issubset(allowed_types)


def test_generate_synthetic_data_multi_bu():
    """
    Behavior: Verify that all business units are generated with their expected profiles.
    - SaaS: steady recurring revenue, high margins (low costs relative to revenue).
    - Enterprise Services: low frequency, high amount transactions.
    - E-commerce: high frequency, lower amount transactions.
    """
    start_date = "2023-01-01"
    end_date = "2024-12-31"  # 2 years to get representative samples
    
    # Act
    df = generate_synthetic_data(start_date, end_date)
    
    # Assert presence of all BUs and Types
    unique_bus = df['BusinessUnit'].unique()
    assert 'ecommerce' in unique_bus
    assert 'saas' in unique_bus
    assert 'enterprise' in unique_bus
    
    unique_types = df['Type'].unique()
    assert 'revenue' in unique_types
    assert 'cost' in unique_types
    
    # Check SaaS High Margin Profile
    saas_rev = df[(df['BusinessUnit'] == 'saas') & (df['Type'] == 'revenue')]['Amount'].sum()
    saas_cost = df[(df['BusinessUnit'] == 'saas') & (df['Type'] == 'cost')]['Amount'].sum()
    assert saas_rev > 0
    saas_margin = (saas_rev - saas_cost) / saas_rev
    assert saas_margin >= 0.70  # SaaS margin should be at least 70%
    
    # Check Enterprise high deal size / low frequency vs E-commerce
    enterprise_rev_txns = df[(df['BusinessUnit'] == 'enterprise') & (df['Type'] == 'revenue')]
    ecommerce_rev_txns = df[(df['BusinessUnit'] == 'ecommerce') & (df['Type'] == 'revenue')]
    
    assert len(enterprise_rev_txns) < len(ecommerce_rev_txns)
    assert enterprise_rev_txns['Amount'].mean() > 5000.0
    assert ecommerce_rev_txns['Amount'].mean() < 300.0

