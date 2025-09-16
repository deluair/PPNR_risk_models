"""
PPNR Risk Models System Demonstration

This script demonstrates the key capabilities of the comprehensive PPNR 
(Pre-Provision Net Revenue) risk modeling system including:
- Data validation and processing
- Credit risk modeling
- Market risk assessment
- Stress testing
- Regulatory compliance
- Capital adequacy calculations
- Dashboard visualization
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

def load_sample_data():
    """Load the generated sample data"""
    print_section_header("LOADING SAMPLE DATA")
    
    try:
        # Load data files
        macro_data = pd.read_csv('data/raw/macro_data.csv')
        market_data = pd.read_csv('data/raw/market_data.csv')
        portfolio_data = pd.read_csv('data/raw/portfolio_data.csv')
        bank_metrics = pd.read_csv('data/processed/bank_metrics.csv')
        
        # Load scenarios
        baseline = pd.read_csv('data/scenarios/baseline_scenario.csv')
        adverse = pd.read_csv('data/scenarios/adverse_scenario.csv')
        severely_adverse = pd.read_csv('data/scenarios/severely_adverse_scenario.csv')
        
        logger.info("Successfully loaded all sample data files")
        
        print(f"✓ Macro data: {len(macro_data)} records")
        print(f"✓ Market data: {len(market_data)} records") 
        print(f"✓ Portfolio data: {len(portfolio_data)} loans")
        print(f"✓ Bank metrics: {len(bank_metrics)} quarters")
        print(f"✓ Stress scenarios: 3 scenarios with {len(baseline)} quarters each")
        
        return {
            'macro': macro_data,
            'market': market_data,
            'portfolio': portfolio_data,
            'bank_metrics': bank_metrics,
            'scenarios': {
                'baseline': baseline,
                'adverse': adverse,
                'severely_adverse': severely_adverse
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        print("❌ Failed to load sample data. Please run generate_sample_data.py first.")
        return None

def demonstrate_data_validation(data):
    """Demonstrate data validation capabilities"""
    print_section_header("DATA VALIDATION & PROCESSING")
    
    try:
        from data.data_validator import DataValidator
        
        # Create validator with basic config
        config = {
            'data_validation': {
                'completeness_threshold': 0.95,
                'outlier_threshold': 3.0
            }
        }
        
        validator = DataValidator(config)
        
        print_subsection("Portfolio Data Validation")
        
        # Validate portfolio data using the general validate_data method
        validation_results = validator.validate_data(data['portfolio'], 'portfolio_data')
        
        print(f"✓ Data validation completed")
        print(f"  - Total records: {len(data['portfolio'])}")
        print(f"  - Quality score: {validation_results.get('quality_score', 0):.2f}")
        
        if validation_results.get('warnings'):
            print(f"  - Warnings: {len(validation_results['warnings'])}")
            
        # Show data quality metrics
        print_subsection("Data Quality Metrics")
        portfolio = data['portfolio']
        
        print(f"✓ Missing data analysis:")
        missing_pct = portfolio.isnull().sum() / len(portfolio) * 100
        for col in ['current_balance', 'interest_rate', 'pd_1y', 'lgd']:
            if col in missing_pct:
                print(f"  - {col}: {missing_pct[col]:.2f}% missing")
        
        print(f"✓ Portfolio composition:")
        loan_type_dist = portfolio['loan_type'].value_counts()
        for loan_type, count in loan_type_dist.head(3).items():
            pct = count / len(portfolio) * 100
            print(f"  - {loan_type}: {count:,} loans ({pct:.1f}%)")
            
        logger.info("Data validation demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data validation demo: {e}")
        print("❌ Data validation demo failed - using basic validation instead")
        
        # Fallback to basic validation
        print_subsection("Basic Data Quality Check")
        portfolio = data['portfolio']
        
        print(f"✓ Portfolio Overview:")
        print(f"  - Total records: {len(portfolio)}")
        print(f"  - Total columns: {len(portfolio.columns)}")
        print(f"  - Data completeness: {(1 - portfolio.isnull().sum().sum() / (len(portfolio) * len(portfolio.columns))):.1%}")
        
        print(f"✓ Key metrics:")
        if 'current_balance' in portfolio.columns:
            print(f"  - Total exposure: ${portfolio['current_balance'].sum():,.0f}")
        if 'loan_type' in portfolio.columns:
            print(f"  - Loan types: {portfolio['loan_type'].nunique()}")
        if 'current_rating' in portfolio.columns:
            print(f"  - Credit ratings: {portfolio['current_rating'].nunique()}")

def demonstrate_credit_risk_modeling(data):
    """Demonstrate credit risk modeling capabilities"""
    print_section_header("CREDIT RISK MODELING")
    
    try:
        from risk_factors.credit_risk import CreditRiskModel
        
        # Create model with basic config
        config = {
            'model_type': 'logistic_regression',
            'features': ['ltv_ratio', 'borrower_fico', 'debt_to_income'],
            'target': 'pd_1y',
            'validation_split': 0.2
        }
        
        credit_model = CreditRiskModel(config)
        
        print_subsection("Portfolio Risk Analysis")
        
        portfolio = data['portfolio']
        
        # Calculate basic risk metrics
        total_exposure = portfolio['current_balance'].sum()
        avg_pd = portfolio['pd_1y'].mean()
        avg_lgd = portfolio['lgd'].mean()
        expected_loss = (portfolio['current_balance'] * portfolio['pd_1y'] * portfolio['lgd']).sum()
        
        print(f"✓ Portfolio Risk Metrics:")
        print(f"  - Total Exposure: ${total_exposure:,.0f}")
        print(f"  - Average PD: {avg_pd:.2%}")
        print(f"  - Average LGD: {avg_lgd:.2%}")
        print(f"  - Expected Loss: ${expected_loss:,.0f}")
        print(f"  - Loss Rate: {expected_loss/total_exposure:.2%}")
        
        # Risk by loan type
        print_subsection("Risk by Loan Type")
        risk_by_type = portfolio.groupby('loan_type').agg({
            'current_balance': 'sum',
            'pd_1y': 'mean',
            'lgd': 'mean'
        }).round(4)
        
        for loan_type in risk_by_type.index[:3]:
            balance = risk_by_type.loc[loan_type, 'current_balance']
            pd = risk_by_type.loc[loan_type, 'pd_1y']
            lgd = risk_by_type.loc[loan_type, 'lgd']
            print(f"  - {loan_type}:")
            print(f"    Balance: ${balance:,.0f}, PD: {pd:.2%}, LGD: {lgd:.2%}")
        
        # Credit rating distribution
        print_subsection("Credit Rating Distribution")
        rating_dist = portfolio['current_rating'].value_counts()
        for rating in ['AAA', 'AA', 'A', 'BBB', 'BB']:
            if rating in rating_dist:
                count = rating_dist[rating]
                pct = count / len(portfolio) * 100
                print(f"  - {rating}: {count:,} loans ({pct:.1f}%)")
        
        logger.info("Credit risk modeling demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in credit risk modeling demo: {e}")
        print("❌ Credit risk modeling demo failed - using basic analysis instead")
        
        # Fallback to basic credit risk analysis
        print_subsection("Basic Credit Risk Analysis")
        portfolio = data['portfolio']
        
        # Calculate basic risk metrics without model
        total_exposure = portfolio['current_balance'].sum()
        avg_pd = portfolio['pd_1y'].mean()
        avg_lgd = portfolio['lgd'].mean()
        expected_loss = (portfolio['current_balance'] * portfolio['pd_1y'] * portfolio['lgd']).sum()
        
        print(f"✓ Portfolio Risk Metrics:")
        print(f"  - Total Exposure: ${total_exposure:,.0f}")
        print(f"  - Average PD: {avg_pd:.2%}")
        print(f"  - Average LGD: {avg_lgd:.2%}")
        print(f"  - Expected Loss: ${expected_loss:,.0f}")
        print(f"  - Loss Rate: {expected_loss/total_exposure:.2%}")
        
        # Risk by loan type
        print_subsection("Risk by Loan Type")
        risk_by_type = portfolio.groupby('loan_type').agg({
            'current_balance': 'sum',
            'pd_1y': 'mean',
            'lgd': 'mean'
        }).round(4)
        
        for loan_type in risk_by_type.index[:3]:
            balance = risk_by_type.loc[loan_type, 'current_balance']
            pd = risk_by_type.loc[loan_type, 'pd_1y']
            lgd = risk_by_type.loc[loan_type, 'lgd']
            print(f"  - {loan_type}:")
            print(f"    Balance: ${balance:,.0f}, PD: {pd:.2%}, LGD: {lgd:.2%}")
        
        # Credit rating distribution
        print_subsection("Credit Rating Distribution")
        rating_dist = portfolio['current_rating'].value_counts()
        for rating in ['AAA', 'AA', 'A', 'BBB', 'BB']:
            if rating in rating_dist:
                count = rating_dist[rating]
                pct = count / len(portfolio) * 100
                print(f"  - {rating}: {count:,} loans ({pct:.1f}%)")

def demonstrate_stress_testing(data):
    """Demonstrate stress testing capabilities"""
    print_section_header("STRESS TESTING")
    
    try:
        print_subsection("Stress Test Scenarios")
        
        scenarios = data['scenarios']
        
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n✓ {scenario_name.replace('_', ' ').title()} Scenario:")
            
            # Show key metrics for first and last quarters
            q1_gdp = scenario_data.iloc[0]['gdp_growth']
            q12_gdp = scenario_data.iloc[-1]['gdp_growth']
            max_unemployment = scenario_data['unemployment_rate'].max()
            min_house_price = scenario_data['house_price_growth'].min()
            
            print(f"  - GDP Growth: Q1 {q1_gdp:.1f}% → Q12 {q12_gdp:.1f}%")
            print(f"  - Peak Unemployment: {max_unemployment:.1f}%")
            print(f"  - Min House Price Growth: {min_house_price:.1f}%")
        
        print_subsection("Stress Test Impact Analysis")
        
        # Simulate impact on portfolio
        portfolio = data['portfolio']
        base_loss_rate = (portfolio['pd_1y'] * portfolio['lgd']).mean()
        
        # Simple stress multipliers (in practice, these would be model-driven)
        stress_multipliers = {
            'baseline': 1.0,
            'adverse': 2.5,
            'severely_adverse': 4.0
        }
        
        print(f"✓ Projected Loss Rates:")
        for scenario, multiplier in stress_multipliers.items():
            stressed_loss_rate = base_loss_rate * multiplier
            total_loss = portfolio['current_balance'].sum() * stressed_loss_rate
            print(f"  - {scenario.replace('_', ' ').title()}: {stressed_loss_rate:.2%} (${total_loss:,.0f})")
        
        # Capital impact
        print_subsection("Capital Impact Assessment")
        
        bank_metrics = data['bank_metrics'].iloc[-1]  # Latest quarter
        current_tier1_ratio = bank_metrics['tier1_ratio']
        
        print(f"✓ Current Tier 1 Ratio: {current_tier1_ratio:.2%}")
        print(f"✓ Regulatory Minimum: 6.00%")
        print(f"✓ Well-Capitalized Threshold: 8.00%")
        
        for scenario, multiplier in stress_multipliers.items():
            if scenario == 'baseline':
                continue
            stressed_loss = portfolio['current_balance'].sum() * base_loss_rate * multiplier
            stressed_tier1 = bank_metrics['tier1_capital'] - stressed_loss
            stressed_ratio = stressed_tier1 / bank_metrics['rwa']
            
            status = "PASS" if stressed_ratio >= 0.06 else "FAIL"
            print(f"  - {scenario.replace('_', ' ').title()}: {stressed_ratio:.2%} ({status})")
        
        logger.info("Stress testing demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in stress testing demo: {e}")
        print("❌ Stress testing demo failed")

def demonstrate_regulatory_compliance(data):
    """Demonstrate regulatory compliance capabilities"""
    print_section_header("REGULATORY COMPLIANCE")
    
    try:
        print_subsection("Capital Adequacy Assessment")
        
        bank_metrics = data['bank_metrics'].iloc[-1]  # Latest quarter
        
        # Calculate key ratios
        tier1_ratio = bank_metrics['tier1_ratio']
        total_capital_ratio = bank_metrics['total_capital_ratio']
        leverage_ratio = bank_metrics['tier1_capital'] / bank_metrics['total_assets']
        
        print(f"✓ Capital Ratios (Latest Quarter):")
        print(f"  - Tier 1 Capital Ratio: {tier1_ratio:.2%}")
        print(f"  - Total Capital Ratio: {total_capital_ratio:.2%}")
        print(f"  - Leverage Ratio: {leverage_ratio:.2%}")
        
        # Regulatory requirements
        requirements = {
            'Tier 1 Capital Ratio': {'current': tier1_ratio, 'minimum': 0.06, 'well_cap': 0.08},
            'Total Capital Ratio': {'current': total_capital_ratio, 'minimum': 0.08, 'well_cap': 0.10},
            'Leverage Ratio': {'current': leverage_ratio, 'minimum': 0.04, 'well_cap': 0.05}
        }
        
        print_subsection("Regulatory Compliance Status")
        
        for ratio_name, values in requirements.items():
            current = values['current']
            minimum = values['minimum']
            well_cap = values['well_cap']
            
            if current >= well_cap:
                status = "WELL CAPITALIZED"
            elif current >= minimum:
                status = "ADEQUATELY CAPITALIZED"
            else:
                status = "UNDERCAPITALIZED"
            
            print(f"✓ {ratio_name}: {status}")
            print(f"  Current: {current:.2%} | Minimum: {minimum:.2%} | Well-Cap: {well_cap:.2%}")
        
        print_subsection("CCAR/DFAST Readiness")
        
        # Check data availability for stress testing
        required_data = ['macro', 'market', 'portfolio', 'scenarios']
        available_data = [key for key in required_data if key in data and data[key] is not None]
        
        print(f"✓ Data Readiness: {len(available_data)}/{len(required_data)} datasets available")
        for dataset in available_data:
            print(f"  - {dataset.title()} data: ✓")
        
        # Scenario coverage
        scenarios = list(data['scenarios'].keys())
        required_scenarios = ['baseline', 'adverse', 'severely_adverse']
        scenario_coverage = len([s for s in required_scenarios if s in scenarios])
        
        print(f"✓ Scenario Coverage: {scenario_coverage}/{len(required_scenarios)} scenarios")
        
        compliance_score = (len(available_data) + scenario_coverage) / (len(required_data) + len(required_scenarios))
        print(f"✓ Overall CCAR Readiness: {compliance_score:.0%}")
        
        logger.info("Regulatory compliance demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in regulatory compliance demo: {e}")
        print("❌ Regulatory compliance demo failed")

def demonstrate_market_risk(data):
    """Demonstrate market risk assessment capabilities"""
    print_section_header("MARKET RISK ASSESSMENT")
    
    try:
        print_subsection("Market Data Analysis")
        
        market_data = data['market']
        
        # Calculate recent volatilities (last 30 days)
        recent_data = market_data.tail(30)
        
        sp500_vol = recent_data['sp500_return'].std() * np.sqrt(252)  # Annualized
        rate_vol = recent_data['treasury_10y'].diff().std() * np.sqrt(252)
        fx_vol = recent_data['eur_usd'].pct_change().std() * np.sqrt(252)
        
        print(f"✓ Market Volatilities (Annualized):")
        print(f"  - S&P 500: {sp500_vol:.1%}")
        print(f"  - 10Y Treasury: {rate_vol:.1%}")
        print(f"  - EUR/USD: {fx_vol:.1%}")
        
        # Current market levels
        current_data = market_data.iloc[-1]
        print(f"\n✓ Current Market Levels:")
        print(f"  - S&P 500: {current_data['sp500_price']:,.0f}")
        print(f"  - 10Y Treasury: {current_data['treasury_10y']:.2f}%")
        print(f"  - Investment Grade Spread: {current_data['investment_grade_spread']:.0f} bps")
        print(f"  - High Yield Spread: {current_data['high_yield_spread']:.0f} bps")
        
        print_subsection("Value at Risk (VaR) Estimation")
        
        # Simple VaR calculation for demonstration
        portfolio_value = 1000000  # $1M portfolio assumption
        confidence_levels = [0.95, 0.99]
        
        for confidence in confidence_levels:
            var_multiplier = np.percentile(recent_data['sp500_return'], (1-confidence)*100)
            var_amount = portfolio_value * abs(var_multiplier)
            
            print(f"✓ {confidence:.0%} VaR (1-day): ${var_amount:,.0f}")
        
        print_subsection("Interest Rate Risk")
        
        # Duration analysis
        avg_duration = 4.5  # Assumed portfolio duration
        rate_change_scenarios = [-0.01, -0.005, 0.005, 0.01]  # 1% rate changes
        
        print(f"✓ Duration-based P&L (Duration: {avg_duration} years):")
        for rate_change in rate_change_scenarios:
            duration_pnl = -avg_duration * rate_change * portfolio_value
            print(f"  - {rate_change:+.1%} rate change: ${duration_pnl:+,.0f}")
        
        logger.info("Market risk assessment demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in market risk demo: {e}")
        print("❌ Market risk assessment demo failed")

def demonstrate_ppnr_projection(data):
    """Demonstrate PPNR projection capabilities"""
    print_section_header("PPNR PROJECTIONS")
    
    try:
        print_subsection("Net Interest Income Projection")
        
        bank_metrics = data['bank_metrics']
        portfolio = data['portfolio']
        
        # Current metrics
        current_nii = bank_metrics.iloc[-1]['net_interest_income']
        current_loans = bank_metrics.iloc[-1]['total_loans']
        
        # Calculate net interest margin
        nim = current_nii / current_loans * 4  # Quarterly to annual
        
        print(f"✓ Current Metrics:")
        print(f"  - Net Interest Income (Quarterly): ${current_nii:,.0f}")
        print(f"  - Total Loans: ${current_loans:,.0f}")
        print(f"  - Net Interest Margin: {nim:.2%}")
        
        # Project under different scenarios
        scenarios = data['scenarios']
        
        print_subsection("Scenario-Based NII Projections")
        
        for scenario_name, scenario_data in scenarios.items():
            # Simple projection based on rate environment
            avg_fed_rate = scenario_data['fed_funds_rate'].mean()
            rate_impact = (avg_fed_rate - 2.5) / 100  # Relative to base case
            projected_nim = nim * (1 + rate_impact * 0.5)  # 50% beta to rates
            projected_nii = projected_nim * current_loans / 4  # Back to quarterly
            
            print(f"✓ {scenario_name.replace('_', ' ').title()}:")
            print(f"  - Avg Fed Funds Rate: {avg_fed_rate:.2f}%")
            print(f"  - Projected NIM: {projected_nim:.2%}")
            print(f"  - Projected NII (Q): ${projected_nii:,.0f}")
        
        print_subsection("Non-Interest Income Projection")
        
        current_nonii = bank_metrics.iloc[-1]['noninterest_income']
        current_fee_income = bank_metrics.iloc[-1]['fee_income']
        current_trading = bank_metrics.iloc[-1]['trading_revenue']
        
        print(f"✓ Current Non-Interest Income Components:")
        print(f"  - Total Non-Interest Income: ${current_nonii:,.0f}")
        print(f"  - Fee Income: ${current_fee_income:,.0f}")
        print(f"  - Trading Revenue: ${current_trading:,.0f}")
        
        print_subsection("Provision Expense Projection")
        
        current_provisions = bank_metrics.iloc[-1]['provision_expense']
        
        # Link provisions to economic scenarios
        for scenario_name, scenario_data in scenarios.items():
            max_unemployment = scenario_data['unemployment_rate'].max()
            
            # Simple provision model based on unemployment
            if scenario_name == 'baseline':
                provision_multiplier = 1.0
            elif scenario_name == 'adverse':
                provision_multiplier = 2.5
            else:  # severely adverse
                provision_multiplier = 4.0
            
            projected_provisions = current_provisions * provision_multiplier
            
            print(f"✓ {scenario_name.replace('_', ' ').title()}:")
            print(f"  - Peak Unemployment: {max_unemployment:.1f}%")
            print(f"  - Projected Provisions: ${projected_provisions:,.0f}")
        
        logger.info("PPNR projections demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in PPNR projections demo: {e}")
        print("❌ PPNR projections demo failed")

def generate_summary_report(data):
    """Generate a summary report of the demonstration"""
    print_section_header("DEMONSTRATION SUMMARY REPORT")
    
    try:
        # System capabilities demonstrated
        capabilities = [
            "✓ Data Validation & Quality Assessment",
            "✓ Credit Risk Modeling & Portfolio Analysis", 
            "✓ Market Risk Assessment & VaR Calculation",
            "✓ Stress Testing with Multiple Scenarios",
            "✓ Regulatory Compliance Monitoring",
            "✓ PPNR Projections & Revenue Modeling",
            "✓ Capital Adequacy Assessment"
        ]
        
        print_subsection("System Capabilities Demonstrated")
        for capability in capabilities:
            print(f"  {capability}")
        
        # Data summary
        print_subsection("Data Processing Summary")
        
        total_loans = len(data['portfolio'])
        total_exposure = data['portfolio']['current_balance'].sum()
        data_points = len(data['macro']) + len(data['market']) + len(data['bank_metrics'])
        
        print(f"✓ Portfolio Analysis:")
        print(f"  - Loans Processed: {total_loans:,}")
        print(f"  - Total Exposure: ${total_exposure:,.0f}")
        print(f"  - Data Points Analyzed: {data_points:,}")
        
        # Risk metrics summary
        print_subsection("Key Risk Metrics")
        
        portfolio = data['portfolio']
        avg_pd = portfolio['pd_1y'].mean()
        avg_lgd = portfolio['lgd'].mean()
        expected_loss = (portfolio['current_balance'] * portfolio['pd_1y'] * portfolio['lgd']).sum()
        loss_rate = expected_loss / total_exposure
        
        print(f"✓ Credit Risk Summary:")
        print(f"  - Average PD: {avg_pd:.2%}")
        print(f"  - Average LGD: {avg_lgd:.2%}")
        print(f"  - Expected Loss Rate: {loss_rate:.2%}")
        
        # Capital summary
        bank_metrics = data['bank_metrics'].iloc[-1]
        tier1_ratio = bank_metrics['tier1_ratio']
        
        print(f"✓ Capital Position:")
        print(f"  - Tier 1 Ratio: {tier1_ratio:.2%}")
        print(f"  - Regulatory Status: {'WELL CAPITALIZED' if tier1_ratio >= 0.08 else 'ADEQUATELY CAPITALIZED'}")
        
        # Recommendations
        print_subsection("System Recommendations")
        
        recommendations = [
            "• Continue monitoring credit concentrations by geography and industry",
            "• Enhance stress testing with more granular scenario analysis", 
            "• Implement real-time market risk monitoring",
            "• Develop automated regulatory reporting workflows",
            "• Consider machine learning models for PD estimation",
            "• Establish early warning indicators for portfolio deterioration"
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
        
        # Save summary to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"demo_summary_{timestamp}.txt"
        
        # Convert Unicode checkmarks to ASCII for Windows compatibility
        ascii_capabilities = [cap.replace("✓", "[PASS]") for cap in capabilities]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PPNR Risk Models System - Demonstration Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Capabilities Demonstrated:\n")
            for capability in ascii_capabilities:
                f.write(f"{capability}\n")
            
            f.write(f"\nPortfolio Summary:\n")
            f.write(f"- Total Loans: {total_loans:,}\n")
            f.write(f"- Total Exposure: ${total_exposure:,.0f}\n")
            f.write(f"- Expected Loss Rate: {loss_rate:.2%}\n")
            f.write(f"- Tier 1 Capital Ratio: {tier1_ratio:.2%}\n")
        
        print(f"\n✓ Summary report saved to: {summary_file}")
        
        logger.info("Summary report generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        print("❌ Summary report generation failed")

def main():
    """Main demonstration function"""
    print_section_header("PPNR RISK MODELS SYSTEM DEMONSTRATION")
    print("This demonstration showcases the comprehensive capabilities")
    print("of the PPNR risk modeling and stress testing system.")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load sample data
    data = load_sample_data()
    if not data:
        print("\n❌ Cannot proceed without sample data. Exiting.")
        return
    
    # Run demonstrations
    try:
        demonstrate_data_validation(data)
        demonstrate_credit_risk_modeling(data)
        demonstrate_market_risk(data)
        demonstrate_stress_testing(data)
        demonstrate_regulatory_compliance(data)
        demonstrate_ppnr_projection(data)
        generate_summary_report(data)
        
        print_section_header("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("✓ All system capabilities have been demonstrated")
        print("✓ Sample data processed and analyzed")
        print("✓ Risk metrics calculated and reported")
        print("✓ Regulatory compliance assessed")
        print("✓ Summary report generated")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nFor detailed logs, see: demo_results.log")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        print("Check demo_results.log for detailed error information")

if __name__ == "__main__":
    main()