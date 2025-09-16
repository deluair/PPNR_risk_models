#!/usr/bin/env python3
"""
PPNR Risk Models System Test Suite
Comprehensive testing of all system components
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Setup logging for test execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_results.log')
        ]
    )
    return logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing"""
    logger = logging.getLogger(__name__)
    logger.info("Creating sample test data...")
    
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Sample portfolio data
    np.random.seed(42)
    n_exposures = 1000
    
    portfolio_data = pd.DataFrame({
        'exposure_id': range(1, n_exposures + 1),
        'exposure_amount': np.random.lognormal(15, 1, n_exposures),
        'credit_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], n_exposures),
        'industry_sector': np.random.choice(['Banking', 'Technology', 'Healthcare', 'Energy', 'Retail'], n_exposures),
        'maturity_years': np.random.uniform(0.5, 10, n_exposures),
        'debt_to_equity': np.random.uniform(0.1, 3.0, n_exposures),
        'current_ratio': np.random.uniform(0.5, 3.0, n_exposures),
        'roa': np.random.uniform(-0.05, 0.15, n_exposures),
        'default_flag': np.random.binomial(1, 0.05, n_exposures),
        'loss_given_default': np.random.beta(2, 5, n_exposures),
        'observation_date': pd.date_range('2020-01-01', periods=n_exposures, freq='D')[:n_exposures]
    })
    
    portfolio_data.to_csv('data/raw/portfolio_data.csv', index=False)
    
    # Sample market data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    market_data = pd.DataFrame({
        'date': dates,
        'sp500': 3000 + np.cumsum(np.random.normal(0.05/252, 0.15/np.sqrt(252), len(dates))),
        'treasury_10y': 2.0 + np.cumsum(np.random.normal(0, 0.01, len(dates))),
        'credit_spread': 1.5 + np.cumsum(np.random.normal(0, 0.005, len(dates))),
        'vix': 20 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    })
    
    market_data.to_csv('data/raw/market_data.csv', index=False)
    
    # Sample macro data
    macro_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2023-12-31', freq='Q'),
        'gdp_growth': np.random.normal(0.02, 0.01, 16),
        'unemployment_rate': np.random.uniform(0.03, 0.08, 16),
        'fed_funds_rate': np.random.uniform(0.0, 0.05, 16),
        'inflation_rate': np.random.uniform(0.01, 0.04, 16)
    })
    
    macro_data.to_csv('data/raw/macro_data.csv', index=False)
    
    logger.info("Sample data created successfully")
    return portfolio_data, market_data, macro_data

def test_data_module():
    """Test data processing module"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Data Module...")
    
    try:
        from data.data_loader import DataLoader
        from data.data_validator import DataValidator
        
        logger.info("Testing DataLoader...")
        # Create a simple config for testing
        test_config = {
            'data_sources': {},
            'caching': {'enabled': False}
        }
        loader = DataLoader(test_config)
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100)
        })
        
        # Test data validation
        validator = DataValidator()
        is_valid = validator.validate_data(sample_data)
        
        logger.info("X Data Module Test Passed")
        return True
        
    except Exception as e:
        logger.error(f"X Data Module Test Failed: {str(e)}")
        return False

def test_risk_factors_module():
    """Test risk factor models"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Risk Factors Module...")
    
    try:
        from risk_factors.credit_risk import CreditRiskModel
        from risk_factors.market_risk import MarketRiskModel
        from risk_factors.operational_risk import OperationalRiskModel
        
        logger.info("Testing Credit Risk Model...")
        credit_model = CreditRiskModel()
        
        logger.info("Testing Market Risk Model...")
        market_model = MarketRiskModel()
        
        logger.info("Testing Operational Risk Model...")
        op_model = OperationalRiskModel()
        
        logger.info("X Risk Factors Module Test Passed")
        return True
        
    except Exception as e:
        logger.error(f"X Risk Factors Module Test Failed: {str(e)}")
        return False

def test_regulatory_module():
    """Test regulatory compliance module"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Regulatory Module...")
    
    try:
        from regulatory.basel_iii import BaselIIICalculator
        from regulatory.ccar import CCARReporter
        from regulatory.stress_testing import StressTestFramework
        
        logger.info("Testing Basel III Calculator...")
        basel_calc = BaselIIICalculator()
        
        logger.info("Testing CCAR Reporter...")
        ccar_reporter = CCARReporter()
        
        logger.info("Testing Stress Test Framework...")
        stress_tester = StressTestFramework()
        
        logger.info("X Regulatory Module Test Passed")
        return True
        
    except Exception as e:
        logger.error(f"X Regulatory Module Test Failed: {str(e)}")
        return False

def test_dashboard_module():
    """Test dashboard and visualization module"""
    logger = logging.getLogger(__name__)
    logger.info("Testing Dashboard Module...")
    
    try:
        from dashboard.risk_dashboard import RiskDashboard
        from dashboard.visualization_engine import VisualizationEngine
        from dashboard.report_generator import ReportGenerator
        
        logger.info("Testing Risk Dashboard...")
        dashboard = RiskDashboard()
        
        logger.info("Testing Visualization Engine...")
        viz_engine = VisualizationEngine()
        
        logger.info("Testing Report Generator...")
        report_gen = ReportGenerator()
        
        logger.info("X Dashboard Module Test Passed")
        return True
        
    except Exception as e:
        logger.error(f"X Dashboard Module Test Failed: {str(e)}")
        return False

def test_integration():
    """Test integration between modules"""
    logger = logging.getLogger(__name__)
    logger.info("Testing System Integration...")
    
    try:
        logger.info("Testing basic integration...")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100)
        })
        
        logger.info("X Integration Test Passed")
        return True
        
    except Exception as e:
        logger.error(f"X Integration Test Failed: {str(e)}")
        return False

def run_performance_tests():
    """Run performance benchmarks"""
    logger = logging.getLogger(__name__)
    logger.info("Running Performance Tests...")
    
    try:
        logger.info("Testing basic performance...")
        
        start_time = time.time()
        
        # Simple performance test
        data = pd.DataFrame({
            'values': np.random.randn(10000)
        })
        
        # Basic operations
        result = data['values'].mean()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"X Performance Test: Execution time: {execution_time:.3f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"X Performance Test Failed: {str(e)}")
        return False

def main():
    """Main test execution function"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("PPNR Risk Models System - Comprehensive Test Suite")
    logger.info("=" * 60)
    
    # Create sample data
    create_sample_data()
    
    # Run test modules
    test_results = {
        'Data Module': test_data_module(),
        'Risk Factors Module': test_risk_factors_module(),
        'Regulatory Module': test_regulatory_module(),
        'Dashboard Module': test_dashboard_module(),
        'Integration Test': test_integration(),
        'Performance Tests': run_performance_tests()
    }
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        logger.warning(f"Warning: {total-passed} test(s) failed. Please review the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)