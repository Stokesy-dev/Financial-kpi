import pandas as pd
import pytest
from data.generate_data import generate_synthetic_data

def test_generate_synthetic_data_schema_and_range():
    """
    Behavior: The generator must return a DataFrame containing daily transaction-level records
    with the correct columns, correct date ranges, and valid non-null metric values.
    """
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    
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
    
    # Check that for Issue 1 it contains E-commerce revenue
    assert (df['BusinessUnit'] == 'ecommerce').all()
    assert (df['Type'] == 'revenue').all()
