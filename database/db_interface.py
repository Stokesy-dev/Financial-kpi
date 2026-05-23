import sqlite3
import pandas as pd

def init_db(db_path: str) -> None:
    """
    Initializes the SQLite database and creates the transactions table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            business_unit TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            volume INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def insert_transactions(db_path: str, df: pd.DataFrame) -> None:
    """
    Inserts a DataFrame of transactions into the transactions table.
    """
    conn = sqlite3.connect(db_path)
    
    # Map the DataFrame columns to match the SQL table columns
    df_renamed = df.rename(columns={
        'TransactionID': 'transaction_id',
        'Date': 'date',
        'BusinessUnit': 'business_unit',
        'Type': 'type',
        'Amount': 'amount',
        'Volume': 'volume'
    })
    
    # Ensure correct columns exist
    db_cols = ['transaction_id', 'date', 'business_unit', 'type', 'amount', 'volume']
    df_to_insert = df_renamed[db_cols]
    
    # Append to the SQL table
    df_to_insert.to_sql('transactions', conn, if_exists='append', index=False)
    conn.close()


def query_aggregated_metrics(db_path: str, bu: str, metric: str, frequency: str, reindex_all_dates: bool = False) -> pd.DataFrame:
    """
    Dynamically aggregates metrics (Revenue, Cost, Volume) from raw Transactions
    over a specified time frequency (daily, weekly, monthly) using SQL.
    """
    conn = sqlite3.connect(db_path)
    
    # Determine which column and transaction type we filter by
    if metric in ['revenue', 'cost']:
        metric_col = 'amount'
        type_filter = metric
    elif metric == 'volume':
        metric_col = 'volume'
        # Volume metric is aggregated from sales/revenue transactions
        type_filter = 'revenue'
    else:
        conn.close()
        raise ValueError(f"Unknown metric: {metric}")
        
    # Map frequency to SQLite date expressions
    if frequency == 'D':
        date_expr = "date"
    elif frequency == 'W':
        # Align to Monday of that week
        date_expr = "date(date, 'weekday 0', '-6 days')"
    elif frequency == 'M':
        # Align to first of the month
        date_expr = "strftime('%Y-%m-01', date)"
    else:
        conn.close()
        raise ValueError(f"Unknown frequency: {frequency}")
        
    query = f"""
        SELECT {date_expr} AS Date, SUM({metric_col}) AS Value
        FROM transactions
        WHERE business_unit = ? AND type = ?
        GROUP BY {date_expr}
        ORDER BY Date ASC
    """
    
    df_result = pd.read_sql_query(query, conn, params=[bu, type_filter])
    conn.close()
    
    if reindex_all_dates and not df_result.empty:
        df_result['Date'] = pd.to_datetime(df_result['Date'])
        
        # Fetch min/max dates from transactions table to align the complete range
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(date), MAX(date) FROM transactions")
        db_min, db_max = cursor.fetchone()
        conn.close()
        
        start_date = pd.to_datetime(db_min) if db_min else df_result['Date'].min()
        end_date = pd.to_datetime(db_max) if db_max else df_result['Date'].max()
        
        if frequency == 'W':
            start_date = start_date - pd.Timedelta(days=start_date.dayofweek)
            end_date = end_date - pd.Timedelta(days=end_date.dayofweek)
        elif frequency == 'M':
            start_date = start_date.replace(day=1)
            end_date = end_date.replace(day=1)
            
        pd_freq = 'MS' if frequency == 'M' else frequency
        full_range = pd.date_range(start=start_date, end=end_date, freq=pd_freq)
        df_result = df_result.set_index('Date').reindex(full_range, fill_value=0.0).reset_index()
        df_result = df_result.rename(columns={'index': 'Date'})
        df_result['Date'] = df_result['Date'].dt.strftime('%Y-%m-%d')
        
    return df_result
