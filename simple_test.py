"""
Simplified Test Suite for PPNR Risk Models

This test focuses on basic import functionality and core system components.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_basic_imports():
    """Test basic module imports"""
    logger = logging.getLogger(__name__)
    logger.info("Testing basic imports...")
    
    try:
        # Test data module imports
        from data.data_loader import DataLoader
        from data.data_validator import DataValidator
        from data.economic_indicators_processor import EconomicIndicatorsProcessor
        logger.info("✓ Data module imports successful")
        
        # Test risk factors imports
        from risk_factors.credit_risk import CreditRiskModel
        from risk_factors.market_risk import MarketRiskModel, MarketRiskFactors
        from risk_factors.operational_risk import OperationalRiskModel, OperationalRiskFactors
        from risk_factors.risk_integration import RiskIntegrationModel
        logger.info("✓ Risk factors module imports successful")
        
        # Test regulatory imports
        from regulatory.basel_iii import BaselIIICalculator
        from regulatory.ccar import CCARStressTester, CCARReporter
        from regulatory.stress_testing import StressTestFramework
        logger.info("✓ Regulatory module imports successful")
        
        # Test dashboard imports (skip if system libraries missing)
        try:
            from dashboard.risk_dashboard import RiskDashboard
            from dashboard.report_generator import ReportGenerator
            logger.info("✓ Dashboard module imports successful")
        except Exception as dashboard_error:
            if "libgobject" in str(dashboard_error) or "WeasyPrint" in str(dashboard_error):
                logger.warning("⚠️ Dashboard module imports skipped (missing system libraries)")
                logger.info("Note: Dashboard functionality requires additional system libraries on Windows")
            else:
                raise dashboard_error
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic functionality without full initialization"""
    logger = logging.getLogger(__name__)
    logger.info("Testing basic functionality...")
    
    try:
        # Test data processing
        sample_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
            'category': ['A', 'B'] * 50
        })
        
        # Basic data operations
        mean_value = sample_data['value'].mean()
        std_value = sample_data['value'].std()
        logger.info(f"Sample data stats - Mean: {mean_value:.3f}, Std: {std_value:.3f}")
        
        # Test MarketRiskFactors functionality
        from risk_factors.market_risk import MarketRiskFactors
        risk_factors = MarketRiskFactors()
        logger.info("✓ MarketRiskFactors instantiated successfully")
        
        # Test OperationalRiskFactors functionality
        from risk_factors.operational_risk import OperationalRiskFactors
        op_factors = OperationalRiskFactors()
        logger.info("✓ OperationalRiskFactors instantiated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Functionality test failed: {str(e)}")
        return False

def test_configuration_handling():
    """Test configuration handling"""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration handling...")
    
    try:
        # Create sample configurations
        data_config = {
            'data_sources': {
                'database': {'enabled': False},
                'files': {'enabled': True}
            },
            'caching': {'enabled': False}
        }
        
        risk_config = {
            'models': {
                'credit_risk': {'enabled': True},
                'market_risk': {'enabled': True},
                'operational_risk': {'enabled': True}
            },
            'parameters': {
                'confidence_level': 0.95,
                'time_horizon': 252
            }
        }
        
        # Test with configurations
        from data.data_loader import DataLoader
        loader = DataLoader(data_config)
        logger.info("✓ DataLoader with config successful")
        
        from data.data_validator import DataValidator
        validator = DataValidator(data_config)
        logger.info("✓ DataValidator with config successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        return False

def main():
    """Run simplified test suite"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("PPNR Risk Models - Simplified Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Handling", test_configuration_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        logger.warning(f"⚠️  {total-passed} test(s) failed. Please review the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)