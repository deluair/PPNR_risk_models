"""
Sample Data Generator for PPNR Risk Models

This script generates realistic sample data files for testing and demonstration
of the PPNR risk models system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary data directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/scenarios'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def generate_macro_data():
    """Generate macroeconomic data"""
    logger.info("Generating macroeconomic data...")
    
    # Generate 5 years of quarterly data
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 1, 1)
    quarters = pd.date_range(start=start_date, end=end_date, freq='Q')
    
    np.random.seed(42)  # For reproducible results
    
    # Base economic indicators
    gdp_growth = np.random.normal(2.5, 1.5, len(quarters))
    unemployment_rate = np.maximum(3.0, np.random.normal(5.5, 2.0, len(quarters)))
    inflation_rate = np.maximum(0.0, np.random.normal(2.0, 1.0, len(quarters)))
    fed_funds_rate = np.maximum(0.0, np.random.normal(2.5, 1.5, len(quarters)))
    
    # Add some correlation and trends
    for i in range(1, len(quarters)):
        gdp_growth[i] = 0.7 * gdp_growth[i-1] + 0.3 * gdp_growth[i]
        unemployment_rate[i] = 0.8 * unemployment_rate[i-1] + 0.2 * unemployment_rate[i]
        inflation_rate[i] = 0.6 * inflation_rate[i-1] + 0.4 * inflation_rate[i]
        fed_funds_rate[i] = 0.9 * fed_funds_rate[i-1] + 0.1 * fed_funds_rate[i]
    
    macro_data = pd.DataFrame({
        'date': quarters,
        'gdp_growth_rate': gdp_growth,
        'unemployment_rate': unemployment_rate,
        'inflation_rate': inflation_rate,
        'fed_funds_rate': fed_funds_rate,
        'house_price_index': 100 * np.cumprod(1 + np.random.normal(0.02, 0.05, len(quarters))),
        'vix': np.maximum(10, np.random.normal(20, 8, len(quarters))),
        'credit_spread': np.maximum(0.5, np.random.normal(2.0, 1.0, len(quarters))),
        'dollar_index': 100 + np.cumsum(np.random.normal(0, 2, len(quarters)))
    })
    
    macro_data.to_csv('data/raw/macro_data.csv', index=False)
    logger.info(f"Generated macro data with {len(macro_data)} records")
    return macro_data

def generate_market_data():
    """Generate market data"""
    logger.info("Generating market data...")
    
    # Generate daily data for 2 years
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(123)
    
    # Stock indices
    sp500_returns = np.random.normal(0.0005, 0.015, len(dates))
    sp500_prices = 4000 * np.cumprod(1 + sp500_returns)
    
    # Interest rates
    treasury_10y = 2.0 + np.cumsum(np.random.normal(0, 0.05, len(dates)))
    treasury_2y = treasury_10y - 0.5 + np.random.normal(0, 0.3, len(dates))
    
    # Credit spreads
    investment_grade_spread = np.maximum(0.5, 1.5 + np.cumsum(np.random.normal(0, 0.02, len(dates))))
    high_yield_spread = investment_grade_spread + 3.0 + np.random.normal(0, 0.5, len(dates))
    
    # FX rates
    eur_usd = 1.10 + np.cumsum(np.random.normal(0, 0.005, len(dates)))
    
    market_data = pd.DataFrame({
        'date': dates,
        'sp500_price': sp500_prices,
        'sp500_return': sp500_returns,
        'treasury_10y': treasury_10y,
        'treasury_2y': treasury_2y,
        'investment_grade_spread': investment_grade_spread,
        'high_yield_spread': high_yield_spread,
        'eur_usd': eur_usd,
        'crude_oil': 70 + np.cumsum(np.random.normal(0, 2, len(dates))),
        'gold_price': 1800 + np.cumsum(np.random.normal(0, 20, len(dates)))
    })
    
    market_data.to_csv('data/raw/market_data.csv', index=False)
    logger.info(f"Generated market data with {len(market_data)} records")
    return market_data

def generate_portfolio_data():
    """Generate portfolio/loan data"""
    logger.info("Generating portfolio data...")
    
    np.random.seed(456)
    
    # Generate loan portfolio
    n_loans = 10000
    
    # Loan characteristics
    loan_types = ['residential_mortgage', 'commercial_real_estate', 'commercial_industrial', 
                  'consumer', 'credit_card', 'auto_loan']
    
    portfolio_data = pd.DataFrame({
        'loan_id': [f'LOAN_{i:06d}' for i in range(1, n_loans + 1)],
        'loan_type': np.random.choice(loan_types, n_loans),
        'origination_date': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1460, n_loans), unit='D'),
        'maturity_date': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 3650, n_loans), unit='D'),
        'original_balance': np.random.lognormal(11, 1, n_loans),  # Mean around $60k
        'current_balance': lambda x: x * np.random.uniform(0.3, 1.0, n_loans),
        'interest_rate': np.random.normal(4.5, 2.0, n_loans),
        'ltv_ratio': np.random.uniform(0.3, 0.95, n_loans),
        'borrower_fico': np.random.normal(720, 80, n_loans).astype(int),
        'debt_to_income': np.random.uniform(0.1, 0.6, n_loans),
        'geography': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_loans),
        'industry': np.random.choice(['retail', 'manufacturing', 'real_estate', 'healthcare', 
                                    'technology', 'energy', 'financial', 'other'], n_loans),
        'current_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], n_loans, 
                                         p=[0.05, 0.15, 0.25, 0.30, 0.15, 0.08, 0.02]),
        'pd_1y': np.random.beta(1, 50, n_loans),  # Probability of default
        'lgd': np.random.beta(2, 3, n_loans),     # Loss given default
        'ead': lambda x: x * np.random.uniform(0.8, 1.2, n_loans)  # Exposure at default
    })
    
    # Calculate current balance and EAD properly
    portfolio_data['current_balance'] = portfolio_data['original_balance'] * np.random.uniform(0.3, 1.0, n_loans)
    portfolio_data['ead'] = portfolio_data['current_balance'] * np.random.uniform(0.8, 1.2, n_loans)
    
    portfolio_data.to_csv('data/raw/portfolio_data.csv', index=False)
    logger.info(f"Generated portfolio data with {len(portfolio_data)} records")
    return portfolio_data

def generate_stress_scenarios():
    """Generate stress test scenarios"""
    logger.info("Generating stress test scenarios...")
    
    # Baseline scenario
    baseline = pd.DataFrame({
        'quarter': range(1, 13),  # 3 years of quarterly projections
        'gdp_growth': [2.5, 2.3, 2.4, 2.6, 2.5, 2.4, 2.3, 2.5, 2.4, 2.3, 2.2, 2.1],
        'unemployment_rate': [4.0, 4.1, 4.0, 3.9, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.2, 4.1],
        'fed_funds_rate': [2.5, 2.75, 3.0, 3.25, 3.5, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0],
        'house_price_growth': [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        'equity_price_growth': [8.0, 6.0, 7.0, 5.0, 6.0, 7.0, 8.0, 6.0, 5.0, 7.0, 8.0, 9.0]
    })
    
    # Adverse scenario
    adverse = pd.DataFrame({
        'quarter': range(1, 13),
        'gdp_growth': [-1.0, -2.0, -1.5, 0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.1, 2.0, 1.9],
        'unemployment_rate': [5.0, 6.5, 8.0, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0],
        'fed_funds_rate': [2.0, 1.5, 1.0, 0.5, 0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5],
        'house_price_growth': [-5.0, -8.0, -10.0, -8.0, -5.0, -2.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.5],
        'equity_price_growth': [-25.0, -15.0, -10.0, 5.0, 10.0, 8.0, 6.0, 7.0, 8.0, 9.0, 10.0, 8.0]
    })
    
    # Severely adverse scenario
    severely_adverse = pd.DataFrame({
        'quarter': range(1, 13),
        'gdp_growth': [-3.0, -5.0, -4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.1],
        'unemployment_rate': [6.0, 8.0, 10.5, 12.0, 11.5, 11.0, 10.0, 9.0, 8.0, 7.0, 6.5, 6.0],
        'fed_funds_rate': [1.5, 1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        'house_price_growth': [-15.0, -20.0, -18.0, -15.0, -10.0, -5.0, -2.0, 0.0, 1.0, 2.0, 2.5, 3.0],
        'equity_price_growth': [-40.0, -30.0, -20.0, -10.0, 0.0, 5.0, 8.0, 10.0, 12.0, 10.0, 8.0, 9.0]
    })
    
    # Save scenarios
    baseline.to_csv('data/scenarios/baseline_scenario.csv', index=False)
    adverse.to_csv('data/scenarios/adverse_scenario.csv', index=False)
    severely_adverse.to_csv('data/scenarios/severely_adverse_scenario.csv', index=False)
    
    logger.info("Generated stress test scenarios: baseline, adverse, severely adverse")
    return baseline, adverse, severely_adverse

def generate_bank_metrics():
    """Generate bank-specific metrics and balance sheet data"""
    logger.info("Generating bank metrics...")
    
    # Quarterly data for 3 years
    quarters = pd.date_range(start='2021-01-01', end='2024-01-01', freq='Q')
    
    np.random.seed(789)
    
    bank_metrics = pd.DataFrame({
        'date': quarters,
        'total_assets': 500000 + np.cumsum(np.random.normal(5000, 10000, len(quarters))),
        'total_loans': 350000 + np.cumsum(np.random.normal(3000, 8000, len(quarters))),
        'total_deposits': 400000 + np.cumsum(np.random.normal(4000, 9000, len(quarters))),
        'tier1_capital': 45000 + np.cumsum(np.random.normal(500, 1000, len(quarters))),
        'total_capital': 55000 + np.cumsum(np.random.normal(600, 1200, len(quarters))),
        'net_interest_income': 3000 + np.random.normal(0, 300, len(quarters)),
        'noninterest_income': 1500 + np.random.normal(0, 200, len(quarters)),
        'noninterest_expense': 2800 + np.random.normal(0, 250, len(quarters)),
        'provision_expense': np.maximum(0, np.random.normal(200, 150, len(quarters))),
        'trading_revenue': np.random.normal(300, 200, len(quarters)),
        'fee_income': 800 + np.random.normal(0, 100, len(quarters)),
        'charge_offs': np.maximum(0, np.random.normal(150, 100, len(quarters))),
        'recoveries': np.maximum(0, np.random.normal(50, 30, len(quarters)))
    })
    
    # Calculate derived metrics
    bank_metrics['rwa'] = bank_metrics['total_assets'] * 0.75  # Simplified RWA
    bank_metrics['tier1_ratio'] = bank_metrics['tier1_capital'] / bank_metrics['rwa']
    bank_metrics['total_capital_ratio'] = bank_metrics['total_capital'] / bank_metrics['rwa']
    bank_metrics['roe'] = (bank_metrics['net_interest_income'] + bank_metrics['noninterest_income'] - 
                          bank_metrics['noninterest_expense'] - bank_metrics['provision_expense']) / bank_metrics['total_capital'] * 4
    bank_metrics['roa'] = bank_metrics['roe'] * bank_metrics['total_capital'] / bank_metrics['total_assets']
    
    bank_metrics.to_csv('data/processed/bank_metrics.csv', index=False)
    logger.info(f"Generated bank metrics with {len(bank_metrics)} records")
    return bank_metrics

def main():
    """Generate all sample data files"""
    logger.info("Starting sample data generation...")
    
    # Create directories
    create_directories()
    
    # Generate all data files
    macro_data = generate_macro_data()
    market_data = generate_market_data()
    portfolio_data = generate_portfolio_data()
    scenarios = generate_stress_scenarios()
    bank_metrics = generate_bank_metrics()
    
    logger.info("Sample data generation completed successfully!")
    logger.info("Generated files:")
    logger.info("  - data/raw/macro_data.csv")
    logger.info("  - data/raw/market_data.csv") 
    logger.info("  - data/raw/portfolio_data.csv")
    logger.info("  - data/scenarios/baseline_scenario.csv")
    logger.info("  - data/scenarios/adverse_scenario.csv")
    logger.info("  - data/scenarios/severely_adverse_scenario.csv")
    logger.info("  - data/processed/bank_metrics.csv")

if __name__ == "__main__":
    main()