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

def query_aggregated_metrics(db_path: str, bu: str, metric: str, frequency: str) -> pd.DataFrame:
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
        GROUP BY Date
        ORDER BY Date ASC
    """
    
    df_result = pd.read_sql_query(query, conn, params=[bu, type_filter])
    conn.close()
    
    return df_result
