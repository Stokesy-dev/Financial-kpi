import pandas as pd
import numpy as np

def generate_synthetic_data(start_date: str, end_date: str, inject_anomalies: bool = True) -> pd.DataFrame:
    """
    Generates synthetic daily transaction-level records for:
    - E-commerce (high frequency, lower deal size, weekly/annual seasonality, high cost of goods).
    - SaaS (medium frequency, steady growth, low recurring cost, high margin).
    - Enterprise Services (low frequency, lumpy contract wins with very large transaction amounts).
    
    Deterministic seeding ensures reproducible data across test runs and dashboard sessions.
    """
    # Seed numpy for determinism
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    records = []
    txn_id_counter = 1
    
    # Base MRR count for SaaS that increases over time
    saas_subscriber_base = 100.0
    
    for date in dates:
        # Get date variables for seasonality
        day_of_week = date.dayofweek
        month = date.month
        year = date.year
        
        # Calculate growth trend factor (since Jan 1, 2023)
        days_passed = (date - pd.to_datetime("2023-01-01")).days
        growth_factor = 1.0 + (days_passed * 0.0003)  # linear growth of ~10% per year
        
        # Seasonality factors
        weekend_factor = 1.3 if day_of_week in [4, 5] else 0.85
        holiday_factor = 1.5 if month in [11, 12] else 0.9
        
        # Determine anomaly flags if inject_anomalies is enabled
        is_checkout_outage = False
        is_cost_leak = False
        is_pricing_glitch = False
        is_contract_delay = False
        
        if inject_anomalies:
            is_checkout_outage = (pd.Timestamp("2024-06-10") <= date <= pd.Timestamp("2024-06-15"))
            is_cost_leak = (pd.Timestamp("2024-10-05") <= date <= pd.Timestamp("2024-10-15"))
            is_pricing_glitch = (pd.Timestamp("2025-03-01") <= date <= pd.Timestamp("2025-03-05"))
            is_contract_delay = (pd.Timestamp("2025-08-01") <= date <= pd.Timestamp("2025-08-31"))
            
        # -------------------------------------------------------------
        # 1. E-COMMERCE GENERATION
        # -------------------------------------------------------------
        # High frequency, low value, strong weekly/annual seasonality
        cogs_rate = 0.70  # cost is ~70% of revenue
        
        # Revenue transactions
        poisson_rate = 25 * weekend_factor * holiday_factor * growth_factor
        if is_checkout_outage:
            poisson_rate *= 0.05  # 95% dip in transactions
        elif is_pricing_glitch:
            poisson_rate *= 5.0   # 5x volume spike
            
        num_ecom_rev = int(np.random.poisson(poisson_rate))
        
        # In a checkout outage, we run a very low loop count. 
        # Otherwise we run max(1, num_ecom_rev) to ensure activity.
        loop_count = num_ecom_rev if is_checkout_outage else max(1, num_ecom_rev)
        
        for _ in range(loop_count):
            if is_pricing_glitch:
                amount = np.random.uniform(1.0, 3.0)  # low price glitch ($1 - $3)
                volume = int(np.random.randint(1, 4))
            else:
                amount = np.random.normal(65.0, 15.0)
                volume = int(np.random.randint(1, 4))
                
            records.append({
                'TransactionID': f"TXN-ECOMR-{date.strftime('%Y%m%d')}-{txn_id_counter:05d}",
                'Date': date.strftime('%Y-%m-%d'),
                'BusinessUnit': 'ecommerce',
                'Type': 'revenue',
                'Amount': max(0.50 if is_pricing_glitch else 5.0, round(amount, 2)),
                'Volume': max(1, volume)
            })
            txn_id_counter += 1
            
        # Cost transactions (COGS, shipping, marketing)
        num_ecom_cost = int(np.random.poisson(10 * growth_factor))
        for _ in range(max(1, num_ecom_cost)):
            amount = np.random.normal(110.0, 20.0)  # aggregated operational costs
            records.append({
                'TransactionID': f"TXN-ECOMC-{date.strftime('%Y%m%d')}-{txn_id_counter:05d}",
                'Date': date.strftime('%Y-%m-%d'),
                'BusinessUnit': 'ecommerce',
                'Type': 'cost',
                'Amount': max(5.0, round(amount, 2)),
                'Volume': 1
            })
            txn_id_counter += 1

        # -------------------------------------------------------------
        # 2. SAAS GENERATION
        # -------------------------------------------------------------
        # Medium frequency renewals/subscriptions, high margins, steady growth
        saas_subscriber_base += 0.05  # steady net new subscribers daily
        
        # Revenue transactions (Renewals/Signups)
        num_saas_renewals = int(np.random.poisson(8 * growth_factor))
        for _ in range(max(1, num_saas_renewals)):
            # Standard SaaS tiers: $49, $99, $199
            tier = np.random.choice([49.0, 99.0, 199.0], p=[0.5, 0.4, 0.1])
            records.append({
                'TransactionID': f"TXN-SAASR-{date.strftime('%Y%m%d')}-{txn_id_counter:05d}",
                'Date': date.strftime('%Y-%m-%d'),
                'BusinessUnit': 'saas',
                'Type': 'revenue',
                'Amount': tier,
                'Volume': 1
            })
            txn_id_counter += 1
            
        # Cost transactions (Hosting/Infrastructure) - very low, stable
        # 1 transaction per day
        cost_amount = np.random.normal(80.0, 5.0)
        if is_cost_leak:
            # 10x cost spike + flat fee
            cost_amount = cost_amount * 10.0 + np.random.uniform(200, 300)
            
        records.append({
            'TransactionID': f"TXN-SAASC-{date.strftime('%Y%m%d')}-{txn_id_counter:05d}",
            'Date': date.strftime('%Y-%m-%d'),
            'BusinessUnit': 'saas',
            'Type': 'cost',
            'Amount': max(10.0, round(cost_amount, 2)),
            'Volume': 1
        })
        txn_id_counter += 1

        # -------------------------------------------------------------
        # 3. ENTERPRISE SERVICES GENERATION
        # -------------------------------------------------------------
        # Low frequency lumpy contract wins, very high transaction amounts
        
        # Revenue transactions (probability of closing a deal is 3.5% daily, approx 1 deal/month)
        deal_prob = 0.0 if is_contract_delay else 0.035
        if np.random.random() < deal_prob:
            amount = np.random.uniform(15000.0, 45000.0)
            records.append({
                'TransactionID': f"TXN-ENTR-{date.strftime('%Y%m%d')}-{txn_id_counter:05d}",
                'Date': date.strftime('%Y-%m-%d'),
                'BusinessUnit': 'enterprise',
                'Type': 'revenue',
                'Amount': round(amount, 2),
                'Volume': 1
            })
            txn_id_counter += 1
            
        # Cost transactions (Contractor fees & delivery expenses, occurs every Friday)
        if day_of_week == 4:  # Friday
            amount = np.random.normal(4500.0, 300.0)
            records.append({
                'TransactionID': f"TXN-ENTC-{date.strftime('%Y%m%d')}-{txn_id_counter:05d}",
                'Date': date.strftime('%Y-%m-%d'),
                'BusinessUnit': 'enterprise',
                'Type': 'cost',
                'Amount': round(amount, 2),
                'Volume': 1
            })
            txn_id_counter += 1

    return pd.DataFrame(records)
