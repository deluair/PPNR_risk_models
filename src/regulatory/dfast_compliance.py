"""
DFAST (Dodd-Frank Act Stress Testing) Compliance Module

Implements DFAST requirements for banks with assets over $100 billion:
- Company-run stress testing
- Supervisory stress testing coordination
- Public disclosure requirements
- Mid-cycle stress testing
- Regulatory reporting formats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import json

@dataclass
class DFASTRequirements:
    """DFAST regulatory requirements and thresholds."""
    asset_threshold: float = 100e9  # $100 billion
    stress_test_frequency: str = 'annual'  # annual, semi_annual for large banks
    public_disclosure_required: bool = True
    supervisory_coordination: bool = True
    mid_cycle_testing: bool = False  # For banks >$250B
    
@dataclass
class DFASTScenario:
    """DFAST stress test scenario definition."""
    name: str
    description: str
    scenario_type: str
    duration_quarters: int
    macroeconomic_variables: Dict[str, List[float]]
    market_variables: Dict[str, List[float]]
    regulatory_source: str  # 'company', 'supervisory'

class DFASTCompliance:
    """
    DFAST compliance framework for stress testing.
    
    Features:
    - Company-run stress testing
    - Supervisory scenario implementation
    - Public disclosure preparation
    - Mid-cycle stress testing (for applicable banks)
    - Regulatory reporting formats
    - Capital planning integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DFAST compliance framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dfast_config = config.get('dfast', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.DFASTCompliance")
        
        # Initialize requirements
        self.requirements = DFASTRequirements()
        
        # Bank characteristics
        self.total_assets = self.dfast_config.get('total_assets', 150e9)
        self.bank_category = self._determine_bank_category()
        
        # Load scenarios
        self.company_scenarios = self._load_company_scenarios()
        self.supervisory_scenarios = self._load_supervisory_scenarios()
        
        # Results storage
        self.stress_test_results = {}
        self.disclosure_data = {}
        self.regulatory_submissions = {}
        
        # Testing parameters
        self.projection_horizon = self.dfast_config.get('projection_horizon', 9)
        self.as_of_date = self.dfast_config.get('as_of_date', datetime(2024, 12, 31))
        
        self.logger.info(f"DFAST compliance framework initialized for {self.bank_category} bank")
    
    def _determine_bank_category(self) -> str:
        """Determine bank category for DFAST requirements."""
        if self.total_assets >= 700e9:
            return 'category_i'  # G-SIBs
        elif self.total_assets >= 250e9:
            return 'category_ii'  # Large banks
        elif self.total_assets >= 100e9:
            return 'category_iii'  # Regional banks
        else:
            return 'not_subject'  # Below DFAST threshold
    
    def _load_company_scenarios(self) -> Dict[str, DFASTScenario]:
        """Load company-developed stress scenarios."""
        # Company scenarios should reflect the bank's unique risk profile
        scenarios = {
            'baseline': DFASTScenario(
                name='Company Baseline',
                description='Company baseline economic scenario',
                scenario_type='baseline',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [2.2, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.9, 1.9],
                    'unemployment_rate': [3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.1, 4.1, 4.1],
                    'cpi_inflation': [2.3, 2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    'fed_funds_rate': [5.5, 5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.75],
                    '10y_treasury': [4.3, 4.1, 3.9, 3.7, 3.6, 3.5, 3.4, 3.3, 3.3],
                    'house_price_index': [2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3, 0.3],
                    'commercial_real_estate_prices': [1.5, 1.2, 1.0, 0.8, 0.5, 0.3, 0.0, -0.2, -0.2]
                },
                market_variables={
                    'equity_prices': [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -1.0, -1.0],
                    'vix_level': [17, 17, 18, 18, 19, 19, 20, 20, 20],
                    'corporate_bond_spreads': [0.75, 0.75, 0.8, 0.8, 0.85, 0.85, 0.9, 0.9, 0.9]
                },
                regulatory_source='company'
            ),
            'adverse': DFASTScenario(
                name='Company Adverse',
                description='Company adverse economic scenario',
                scenario_type='adverse',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [-0.8, -1.5, -0.5, 0.8, 1.5, 2.0, 2.2, 2.3, 2.3],
                    'unemployment_rate': [3.6, 4.2, 5.5, 6.8, 7.2, 6.8, 6.2, 5.8, 5.5],
                    'cpi_inflation': [2.3, 1.9, 1.4, 1.0, 1.2, 1.6, 1.9, 2.0, 2.0],
                    'fed_funds_rate': [5.5, 4.5, 3.0, 1.5, 1.0, 1.0, 1.5, 2.0, 2.5],
                    '10y_treasury': [4.3, 3.5, 2.5, 1.8, 2.0, 2.4, 2.7, 3.0, 3.2],
                    'house_price_index': [2.0, 0.5, -2.0, -4.5, -3.8, -2.5, -1.0, 0.5, 1.0],
                    'commercial_real_estate_prices': [1.5, -1.0, -4.0, -6.5, -5.2, -3.0, -1.5, 0.0, 1.0]
                },
                market_variables={
                    'equity_prices': [5.0, -10.0, -20.0, -15.0, -8.0, 2.0, 8.0, 12.0, 15.0],
                    'vix_level': [17, 25, 32, 28, 23, 20, 18, 17, 16],
                    'corporate_bond_spreads': [0.75, 1.2, 1.8, 2.1, 1.8, 1.4, 1.1, 0.9, 0.8]
                },
                regulatory_source='company'
            ),
            'severely_adverse': DFASTScenario(
                name='Company Severely Adverse',
                description='Company severely adverse scenario with idiosyncratic risks',
                scenario_type='severely_adverse',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [-2.5, -3.8, -1.8, -0.2, 1.2, 2.5, 3.0, 2.7, 2.5],
                    'unemployment_rate': [3.6, 4.8, 7.2, 9.5, 10.2, 9.8, 8.8, 7.8, 7.0],
                    'cpi_inflation': [2.3, 1.4, 0.5, -0.3, 0.0, 0.5, 1.2, 1.7, 2.0],
                    'fed_funds_rate': [5.5, 3.0, 1.0, 0.1, 0.1, 0.1, 0.5, 1.0, 1.5],
                    '10y_treasury': [4.3, 3.0, 1.5, 1.0, 1.3, 1.8, 2.2, 2.6, 2.9],
                    'house_price_index': [2.0, -2.5, -8.0, -12.0, -10.5, -7.0, -4.0, -1.0, 1.5],
                    'commercial_real_estate_prices': [1.5, -3.0, -10.0, -15.0, -12.5, -8.0, -4.5, -1.0, 2.0]
                },
                market_variables={
                    'equity_prices': [5.0, -20.0, -40.0, -35.0, -20.0, -5.0, 10.0, 18.0, 22.0],
                    'vix_level': [17, 32, 50, 45, 32, 25, 22, 19, 17],
                    'corporate_bond_spreads': [0.75, 1.8, 3.5, 4.2, 3.5, 2.5, 1.8, 1.3, 1.0]
                },
                regulatory_source='company'
            )
        }
        
        return scenarios
    
    def _load_supervisory_scenarios(self) -> Dict[str, DFASTScenario]:
        """Load Federal Reserve supervisory scenarios for coordination."""
        # These would typically be provided by the Federal Reserve
        # For demonstration, using similar structure to CCAR scenarios
        scenarios = {
            'supervisory_baseline': DFASTScenario(
                name='Supervisory Baseline',
                description='Federal Reserve baseline scenario',
                scenario_type='baseline',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [2.1, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.8, 1.8],
                    'unemployment_rate': [3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.2, 4.2, 4.2],
                    'cpi_inflation': [2.4, 2.2, 2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    'fed_funds_rate': [5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.5, 3.5],
                    '10y_treasury': [4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.2]
                },
                market_variables={
                    'equity_prices': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'vix_level': [18, 18, 18, 18, 18, 18, 18, 18, 18]
                },
                regulatory_source='supervisory'
            ),
            'supervisory_severely_adverse': DFASTScenario(
                name='Supervisory Severely Adverse',
                description='Federal Reserve severely adverse scenario',
                scenario_type='severely_adverse',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [-2.8, -4.2, -2.1, -0.5, 1.0, 2.2, 2.8, 2.5, 2.3],
                    'unemployment_rate': [3.7, 5.2, 7.8, 10.0, 10.8, 10.2, 9.1, 8.0, 7.2],
                    'cpi_inflation': [2.4, 1.2, 0.2, -0.5, -0.2, 0.8, 1.5, 1.8, 2.0],
                    'fed_funds_rate': [5.25, 2.5, 0.5, 0.1, 0.1, 0.1, 0.5, 1.0, 1.5],
                    '10y_treasury': [4.2, 2.8, 1.2, 0.8, 1.2, 1.8, 2.2, 2.5, 2.8]
                },
                market_variables={
                    'equity_prices': [0.0, -25.0, -45.0, -40.0, -25.0, -10.0, 5.0, 12.0, 15.0],
                    'vix_level': [18, 35, 55, 48, 35, 28, 25, 22, 20]
                },
                regulatory_source='supervisory'
            )
        }
        
        return scenarios
    
    def run_dfast_stress_test(self, ppnr_models: Dict[str, Any],
                            initial_capital: Dict[str, float],
                            balance_sheet_data: pd.DataFrame,
                            test_type: str = 'annual') -> Dict[str, Any]:
        """
        Run DFAST stress test.
        
        Args:
            ppnr_models: Dictionary of PPNR models
            initial_capital: Initial capital positions
            balance_sheet_data: Historical balance sheet data
            test_type: Type of test ('annual', 'mid_cycle')
            
        Returns:
            DFAST stress test results
        """
        self.logger.info(f"Running DFAST {test_type} stress test...")
        
        results = {
            'test_metadata': {
                'test_type': test_type,
                'test_date': datetime.now().isoformat(),
                'as_of_date': self.as_of_date.isoformat(),
                'bank_category': self.bank_category,
                'total_assets': self.total_assets
            },
            'company_run_results': {},
            'supervisory_coordination': {},
            'capital_projections': {},
            'public_disclosure_data': {},
            'regulatory_submissions': {}
        }
        
        # Run company-developed scenarios
        self.logger.info("Running company-developed scenarios...")
        for scenario_name, scenario in self.company_scenarios.items():
            scenario_results = self._run_scenario_stress_test(
                scenario, ppnr_models, initial_capital, balance_sheet_data
            )
            results['company_run_results'][scenario_name] = scenario_results
        
        # Coordinate with supervisory scenarios (if applicable)
        if self.requirements.supervisory_coordination:
            self.logger.info("Processing supervisory scenarios...")
            for scenario_name, scenario in self.supervisory_scenarios.items():
                scenario_results = self._run_scenario_stress_test(
                    scenario, ppnr_models, initial_capital, balance_sheet_data
                )
                results['supervisory_coordination'][scenario_name] = scenario_results
        
        # Calculate capital projections
        results['capital_projections'] = self._calculate_capital_projections(
            results['company_run_results'], initial_capital
        )
        
        # Prepare public disclosure data
        if self.requirements.public_disclosure_required:
            results['public_disclosure_data'] = self._prepare_public_disclosure(results)
        
        # Prepare regulatory submissions
        results['regulatory_submissions'] = self._prepare_regulatory_submissions(results)
        
        # Store results
        self.stress_test_results = results
        
        self.logger.info("DFAST stress test completed")
        return results
    
    def _run_scenario_stress_test(self, scenario: DFASTScenario,
                                ppnr_models: Dict[str, Any],
                                initial_capital: Dict[str, float],
                                balance_sheet_data: pd.DataFrame) -> Dict[str, Any]:
        """Run stress test for a specific DFAST scenario."""
        scenario_results = {
            'scenario_info': {
                'name': scenario.name,
                'type': scenario.scenario_type,
                'source': scenario.regulatory_source,
                'duration': scenario.duration_quarters
            },
            'quarterly_projections': [],
            'ppnr_summary': {},
            'loss_summary': {},
            'capital_impact': {}
        }
        
        # Generate quarterly projections
        for quarter in range(scenario.duration_quarters):
            # Extract scenario variables
            scenario_vars = self._extract_scenario_variables(scenario, quarter)
            
            # Project PPNR components
            ppnr_projection = self._project_ppnr_components(
                ppnr_models, scenario_vars, quarter
            )
            
            # Project balance sheet
            balance_sheet_projection = self._project_balance_sheet(
                balance_sheet_data, scenario_vars, quarter
            )
            
            # Project losses (including provisions and charge-offs)
            loss_projection = self._project_comprehensive_losses(
                balance_sheet_projection, scenario_vars, quarter
            )
            
            # Calculate net income and capital impact
            net_income = self._calculate_net_income(
                ppnr_projection, loss_projection, scenario_vars
            )
            
            quarterly_projections.append({
                'quarter': quarter + 1,
                'scenario_variables': scenario_vars,
                'ppnr': ppnr_projection,
                'balance_sheet': balance_sheet_projection,
                'losses': loss_projection,
                'net_income': net_income
            })
        
        scenario_results['quarterly_projections'] = quarterly_projections
        
        # Aggregate results
        scenario_results['ppnr_summary'] = self._aggregate_ppnr_results(quarterly_projections)
        scenario_results['loss_summary'] = self._aggregate_loss_results(quarterly_projections)
        scenario_results['capital_impact'] = self._calculate_scenario_capital_impact(
            quarterly_projections, initial_capital
        )
        
        return scenario_results
    
    def _extract_scenario_variables(self, scenario: DFASTScenario, quarter: int) -> Dict[str, float]:
        """Extract scenario variables for a specific quarter."""
        variables = {}
        
        # Extract macroeconomic variables
        for var_name, values in scenario.macroeconomic_variables.items():
            if quarter < len(values):
                variables[var_name] = values[quarter]
            else:
                variables[var_name] = values[-1]
        
        # Extract market variables
        for var_name, values in scenario.market_variables.items():
            if quarter < len(values):
                variables[var_name] = values[quarter]
            else:
                variables[var_name] = values[-1]
        
        return variables
    
    def _project_ppnr_components(self, ppnr_models: Dict[str, Any],
                               scenario_vars: Dict[str, float],
                               quarter: int) -> Dict[str, float]:
        """Project PPNR components under DFAST scenario conditions."""
        ppnr_projection = {}
        
        # Net Interest Income
        if 'nii_model' in ppnr_models:
            nii_features = self._prepare_nii_features(scenario_vars, quarter)
            nii_projection = ppnr_models['nii_model'].predict_scenario(nii_features, scenario_vars)
            ppnr_projection['net_interest_income'] = float(nii_projection)
        
        # Non-Interest Income components
        if 'fee_income_model' in ppnr_models:
            fee_features = self._prepare_fee_features(scenario_vars, quarter)
            fee_projection = ppnr_models['fee_income_model'].predict_scenario(fee_features, scenario_vars)
            ppnr_projection.update({
                'service_charges': float(fee_projection * 0.3),
                'investment_banking_fees': float(fee_projection * 0.25),
                'trading_account_profits': float(fee_projection * 0.2),
                'other_noninterest_income': float(fee_projection * 0.25)
            })
        
        # Trading Revenue (separate from fee income)
        if 'trading_model' in ppnr_models:
            trading_features = self._prepare_trading_features(scenario_vars, quarter)
            trading_projection = ppnr_models['trading_model'].predict_scenario(trading_features, scenario_vars)
            ppnr_projection['trading_revenue'] = float(trading_projection)
        
        # Calculate total non-interest income
        noninterest_income_components = [
            'service_charges', 'investment_banking_fees', 
            'trading_account_profits', 'other_noninterest_income', 'trading_revenue'
        ]
        ppnr_projection['total_noninterest_income'] = sum([
            ppnr_projection.get(component, 0) for component in noninterest_income_components
        ])
        
        # Calculate total PPNR
        ppnr_projection['total_ppnr'] = (
            ppnr_projection.get('net_interest_income', 0) + 
            ppnr_projection.get('total_noninterest_income', 0)
        )
        
        return ppnr_projection
    
    def _project_balance_sheet(self, balance_sheet_data: pd.DataFrame,
                             scenario_vars: Dict[str, float],
                             quarter: int) -> Dict[str, float]:
        """Project balance sheet under DFAST scenario conditions."""
        latest_bs = balance_sheet_data.iloc[-1].to_dict()
        
        # Economic drivers
        gdp_growth = scenario_vars.get('real_gdp_growth', 0) / 100
        unemployment_rate = scenario_vars.get('unemployment_rate', 0) / 100
        house_price_growth = scenario_vars.get('house_price_index', 0) / 100
        
        # Loan portfolio projections
        # Consumer loans (sensitive to unemployment)
        consumer_loan_factor = 1 + (gdp_growth * 0.4 - (unemployment_rate - 0.04) * 0.6) / 4
        
        # Commercial loans (sensitive to GDP and CRE prices)
        cre_price_growth = scenario_vars.get('commercial_real_estate_prices', 0) / 100
        commercial_loan_factor = 1 + (gdp_growth * 0.6 + cre_price_growth * 0.3) / 4
        
        # Residential mortgages (sensitive to house prices)
        mortgage_factor = 1 + (house_price_growth * 0.4 + gdp_growth * 0.2) / 4
        
        projection = {
            'total_assets': latest_bs.get('total_assets', 0) * (1 + gdp_growth / 4),
            'consumer_loans': latest_bs.get('consumer_loans', 0) * consumer_loan_factor,
            'commercial_loans': latest_bs.get('commercial_loans', 0) * commercial_loan_factor,
            'residential_mortgages': latest_bs.get('residential_mortgages', 0) * mortgage_factor,
            'securities': latest_bs.get('securities', 0) * (1 + gdp_growth / 8),
            'deposits': latest_bs.get('deposits', 0) * (1 + gdp_growth / 4),
            'shareholders_equity': latest_bs.get('shareholders_equity', 0)
        }
        
        # Calculate total loans
        projection['total_loans'] = (
            projection['consumer_loans'] + 
            projection['commercial_loans'] + 
            projection['residential_mortgages']
        )
        
        return projection
    
    def _project_comprehensive_losses(self, balance_sheet_projection: Dict[str, float],
                                    scenario_vars: Dict[str, float],
                                    quarter: int) -> Dict[str, float]:
        """Project comprehensive credit losses under DFAST scenarios."""
        # Economic stress indicators
        unemployment_rate = scenario_vars.get('unemployment_rate', 0) / 100
        gdp_growth = scenario_vars.get('real_gdp_growth', 0) / 100
        house_price_growth = scenario_vars.get('house_price_index', 0) / 100
        cre_price_growth = scenario_vars.get('commercial_real_estate_prices', 0) / 100
        
        # Base loss rates by portfolio
        base_rates = {
            'consumer_loans': 0.008,  # 80 bps quarterly
            'commercial_loans': 0.004,  # 40 bps quarterly
            'residential_mortgages': 0.002  # 20 bps quarterly
        }
        
        # Stress adjustments
        unemployment_stress = max(0, unemployment_rate - 0.04) * 3  # 3x multiplier above 4%
        gdp_stress = max(0, -gdp_growth) * 2  # 2x multiplier for negative growth
        house_price_stress = max(0, -house_price_growth) * 1.5  # 1.5x for house price declines
        cre_stress = max(0, -cre_price_growth) * 2  # 2x for CRE price declines
        
        # Portfolio-specific loss rates
        consumer_loss_rate = base_rates['consumer_loans'] + unemployment_stress + gdp_stress
        commercial_loss_rate = base_rates['commercial_loans'] + gdp_stress + cre_stress
        mortgage_loss_rate = base_rates['residential_mortgages'] + house_price_stress + unemployment_stress * 0.5
        
        # Calculate losses by portfolio
        consumer_losses = balance_sheet_projection['consumer_loans'] * consumer_loss_rate
        commercial_losses = balance_sheet_projection['commercial_loans'] * commercial_loss_rate
        mortgage_losses = balance_sheet_projection['residential_mortgages'] * mortgage_loss_rate
        
        total_provision = consumer_losses + commercial_losses + mortgage_losses
        
        return {
            'provision_for_credit_losses': total_provision,
            'consumer_loan_losses': consumer_losses,
            'commercial_loan_losses': commercial_losses,
            'mortgage_losses': mortgage_losses,
            'net_charge_offs': total_provision * 0.85,  # 85% of provisions become charge-offs
            'loss_rates': {
                'consumer': consumer_loss_rate,
                'commercial': commercial_loss_rate,
                'mortgage': mortgage_loss_rate,
                'total': total_provision / balance_sheet_projection['total_loans']
            }
        }
    
    def _calculate_net_income(self, ppnr_projection: Dict[str, float],
                            loss_projection: Dict[str, float],
                            scenario_vars: Dict[str, float]) -> Dict[str, float]:
        """Calculate net income components."""
        # Pre-provision net revenue
        ppnr = ppnr_projection['total_ppnr']
        
        # Provision for credit losses
        provision = loss_projection['provision_for_credit_losses']
        
        # Pre-tax income
        pre_tax_income = ppnr - provision
        
        # Tax calculation (21% corporate tax rate)
        tax_rate = 0.21
        tax_expense = max(0, pre_tax_income * tax_rate)  # No tax benefit if loss
        
        # Net income
        net_income = pre_tax_income - tax_expense
        
        return {
            'pre_provision_net_revenue': ppnr,
            'provision_for_credit_losses': provision,
            'pre_tax_income': pre_tax_income,
            'tax_expense': tax_expense,
            'net_income': net_income
        }
    
    def _prepare_nii_features(self, scenario_vars: Dict[str, float], quarter: int) -> Dict[str, float]:
        """Prepare features for NII model prediction."""
        return {
            'fed_funds_rate': scenario_vars.get('fed_funds_rate', 0),
            '10y_treasury': scenario_vars.get('10y_treasury', 0),
            'yield_curve_slope': scenario_vars.get('10y_treasury', 0) - scenario_vars.get('fed_funds_rate', 0),
            'quarter': quarter + 1,
            'gdp_growth': scenario_vars.get('real_gdp_growth', 0),
            'unemployment_rate': scenario_vars.get('unemployment_rate', 0)
        }
    
    def _prepare_fee_features(self, scenario_vars: Dict[str, float], quarter: int) -> Dict[str, float]:
        """Prepare features for fee income model prediction."""
        return {
            'gdp_growth': scenario_vars.get('real_gdp_growth', 0),
            'unemployment_rate': scenario_vars.get('unemployment_rate', 0),
            'equity_shock': scenario_vars.get('equity_prices', 0),
            'vix_level': scenario_vars.get('vix_level', 0),
            'quarter': quarter + 1,
            'fed_funds_rate': scenario_vars.get('fed_funds_rate', 0)
        }
    
    def _prepare_trading_features(self, scenario_vars: Dict[str, float], quarter: int) -> Dict[str, float]:
        """Prepare features for trading revenue model prediction."""
        return {
            'equity_shock': scenario_vars.get('equity_prices', 0),
            'vix_level': scenario_vars.get('vix_level', 0),
            'credit_spread': scenario_vars.get('corporate_bond_spreads', 0),
            'interest_rate_shock': scenario_vars.get('fed_funds_rate', 0) - 5.5,  # Shock from baseline
            'quarter': quarter + 1,
            'gdp_growth': scenario_vars.get('real_gdp_growth', 0)
        }
    
    def _aggregate_ppnr_results(self, quarterly_projections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate PPNR results across quarters."""
        aggregated = {
            'net_interest_income': [],
            'total_noninterest_income': [],
            'total_ppnr': [],
            'quarterly_growth_rates': []
        }
        
        for i, projection in enumerate(quarterly_projections):
            ppnr = projection['ppnr']
            aggregated['net_interest_income'].append(ppnr.get('net_interest_income', 0))
            aggregated['total_noninterest_income'].append(ppnr.get('total_noninterest_income', 0))
            aggregated['total_ppnr'].append(ppnr.get('total_ppnr', 0))
            
            # Calculate quarter-over-quarter growth
            if i > 0:
                prev_ppnr = quarterly_projections[i-1]['ppnr']['total_ppnr']
                current_ppnr = ppnr['total_ppnr']
                growth_rate = (current_ppnr - prev_ppnr) / abs(prev_ppnr) if prev_ppnr != 0 else 0
                aggregated['quarterly_growth_rates'].append(growth_rate)
            else:
                aggregated['quarterly_growth_rates'].append(0)
        
        # Calculate summary statistics
        aggregated['total_9_quarter'] = sum(aggregated['total_ppnr'])
        aggregated['average_quarterly'] = np.mean(aggregated['total_ppnr'])
        aggregated['minimum_quarterly'] = min(aggregated['total_ppnr'])
        aggregated['maximum_quarterly'] = max(aggregated['total_ppnr'])
        
        return aggregated
    
    def _aggregate_loss_results(self, quarterly_projections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate loss results across quarters."""
        aggregated = {
            'provision_for_credit_losses': [],
            'net_charge_offs': [],
            'cumulative_losses': [],
            'loss_rates': []
        }
        
        cumulative_losses = 0
        
        for projection in quarterly_projections:
            losses = projection['losses']
            provision = losses.get('provision_for_credit_losses', 0)
            charge_offs = losses.get('net_charge_offs', 0)
            
            aggregated['provision_for_credit_losses'].append(provision)
            aggregated['net_charge_offs'].append(charge_offs)
            
            cumulative_losses += charge_offs
            aggregated['cumulative_losses'].append(cumulative_losses)
            aggregated['loss_rates'].append(losses.get('loss_rates', {}).get('total', 0))
        
        # Calculate summary statistics
        aggregated['total_9_quarter_provision'] = sum(aggregated['provision_for_credit_losses'])
        aggregated['total_9_quarter_charge_offs'] = sum(aggregated['net_charge_offs'])
        aggregated['peak_quarterly_loss_rate'] = max(aggregated['loss_rates'])
        aggregated['average_loss_rate'] = np.mean(aggregated['loss_rates'])
        
        return aggregated
    
    def _calculate_scenario_capital_impact(self, quarterly_projections: List[Dict[str, Any]],
                                         initial_capital: Dict[str, float]) -> Dict[str, Any]:
        """Calculate capital impact for a scenario."""
        capital_evolution = {
            'cet1_capital': [initial_capital.get('cet1_capital', 0)],
            'tier1_capital': [initial_capital.get('tier1_capital', 0)],
            'total_capital': [initial_capital.get('total_capital', 0)],
            'retained_earnings_impact': []
        }
        
        # Dividend payout ratio (assume 40% for DFAST)
        dividend_payout_ratio = 0.4
        
        for projection in quarterly_projections:
            net_income = projection['net_income']['net_income']
            
            # Calculate retained earnings (after dividends)
            retained_earnings = net_income * (1 - dividend_payout_ratio)
            capital_evolution['retained_earnings_impact'].append(retained_earnings)
            
            # Update capital levels
            new_cet1 = capital_evolution['cet1_capital'][-1] + retained_earnings
            new_tier1 = capital_evolution['tier1_capital'][-1] + retained_earnings
            new_total = capital_evolution['total_capital'][-1] + retained_earnings
            
            capital_evolution['cet1_capital'].append(new_cet1)
            capital_evolution['tier1_capital'].append(new_tier1)
            capital_evolution['total_capital'].append(new_total)
        
        return {
            'capital_evolution': capital_evolution,
            'total_capital_impact': sum(capital_evolution['retained_earnings_impact']),
            'final_capital_levels': {
                'cet1_capital': capital_evolution['cet1_capital'][-1],
                'tier1_capital': capital_evolution['tier1_capital'][-1],
                'total_capital': capital_evolution['total_capital'][-1]
            }
        }
    
    def _calculate_capital_projections(self, company_results: Dict[str, Any],
                                     initial_capital: Dict[str, float]) -> Dict[str, Any]:
        """Calculate capital projections across all company scenarios."""
        projections = {}
        
        for scenario_name, results in company_results.items():
            capital_impact = results['capital_impact']
            projections[scenario_name] = capital_impact['capital_evolution']
        
        return projections
    
    def _prepare_public_disclosure(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare public disclosure data per DFAST requirements."""
        disclosure_data = {
            'disclosure_date': datetime.now().isoformat(),
            'summary_results': {},
            'methodology_description': self._get_methodology_description(),
            'scenario_descriptions': {},
            'key_assumptions': self._get_key_assumptions()
        }
        
        # Summary results for public disclosure
        for scenario_name, scenario_results in results['company_run_results'].items():
            if scenario_results['scenario_info']['type'] in ['baseline', 'severely_adverse']:
                ppnr_summary = scenario_results['ppnr_summary']
                loss_summary = scenario_results['loss_summary']
                capital_impact = scenario_results['capital_impact']
                
                disclosure_data['summary_results'][scenario_name] = {
                    'total_ppnr_9_quarters': ppnr_summary['total_9_quarter'],
                    'total_losses_9_quarters': loss_summary['total_9_quarter_provision'],
                    'minimum_capital_ratio': self._calculate_minimum_capital_ratio(capital_impact),
                    'capital_impact': capital_impact['total_capital_impact']
                }
        
        # Scenario descriptions (high-level only for public)
        for scenario_name, scenario in self.company_scenarios.items():
            if scenario.scenario_type in ['baseline', 'severely_adverse']:
                disclosure_data['scenario_descriptions'][scenario_name] = {
                    'name': scenario.name,
                    'description': scenario.description,
                    'key_features': self._extract_scenario_key_features(scenario)
                }
        
        return disclosure_data
    
    def _prepare_regulatory_submissions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare regulatory submission data."""
        submissions = {
            'submission_date': datetime.now().isoformat(),
            'fr_y14a_data': self._prepare_fr_y14a_data(results),
            'fr_y14q_data': self._prepare_fr_y14q_data(results),
            'narrative_summary': self._prepare_narrative_summary(results),
            'model_documentation': self._prepare_model_documentation()
        }
        
        return submissions
    
    def _prepare_fr_y14a_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare FR Y-14A regulatory submission data."""
        # FR Y-14A: Annual Capital Assessments and Stress Testing
        fr_y14a = {
            'summary_schedule': {},
            'scenario_results': {},
            'capital_actions': {},
            'business_plan_changes': {}
        }
        
        # Summary schedule with key metrics
        for scenario_name, scenario_results in results['company_run_results'].items():
            fr_y14a['scenario_results'][scenario_name] = {
                'ppnr_projections': scenario_results['ppnr_summary'],
                'loss_projections': scenario_results['loss_summary'],
                'capital_projections': scenario_results['capital_impact']
            }
        
        return fr_y14a
    
    def _prepare_fr_y14q_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare FR Y-14Q regulatory submission data."""
        # FR Y-14Q: Quarterly Capital Assessments and Stress Testing
        fr_y14q = {
            'trading_and_counterparty': {},
            'securities': {},
            'regulatory_capital': {},
            'balances': {}
        }
        
        # Extract quarterly data for regulatory reporting
        # This would include detailed breakdowns by business line, geography, etc.
        
        return fr_y14q
    
    def _prepare_narrative_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare narrative summary for regulatory submission."""
        return {
            'executive_summary': 'DFAST stress test results summary',
            'methodology_overview': 'Overview of stress testing methodology',
            'scenario_analysis': 'Analysis of scenario impacts',
            'model_performance': 'Assessment of model performance',
            'key_risks_identified': 'Key risks and vulnerabilities identified',
            'management_actions': 'Planned management actions based on results'
        }
    
    def _prepare_model_documentation(self) -> Dict[str, Any]:
        """Prepare model documentation for regulatory review."""
        return {
            'model_inventory': 'List of models used in stress testing',
            'model_validation': 'Model validation results and findings',
            'governance_framework': 'Model governance and oversight framework',
            'limitations_and_assumptions': 'Key model limitations and assumptions',
            'benchmarking_results': 'Model benchmarking and challenger model results'
        }
    
    def _get_methodology_description(self) -> str:
        """Get high-level methodology description for public disclosure."""
        return """
        The bank's DFAST stress testing methodology incorporates:
        - Econometric models for PPNR projections
        - Credit risk models for loss estimation
        - Balance sheet projection models
        - Capital planning and optimization models
        
        The methodology is designed to assess the bank's capital adequacy
        under adverse economic conditions while maintaining compliance
        with regulatory requirements.
        """
    
    def _get_key_assumptions(self) -> List[str]:
        """Get key assumptions for public disclosure."""
        return [
            "Business plan assumptions remain constant except for scenario-driven changes",
            "No extraordinary capital actions beyond planned dividends",
            "Regulatory capital requirements remain at current levels",
            "Balance sheet composition evolves based on economic conditions",
            "Credit loss models reflect current portfolio composition"
        ]
    
    def _extract_scenario_key_features(self, scenario: DFASTScenario) -> List[str]:
        """Extract key features of a scenario for public disclosure."""
        features = []
        
        if scenario.scenario_type == 'severely_adverse':
            features.extend([
                "Severe economic recession",
                "Significant increase in unemployment",
                "Substantial decline in asset prices",
                "Elevated market volatility"
            ])
        elif scenario.scenario_type == 'baseline':
            features.extend([
                "Continued economic growth",
                "Stable employment conditions",
                "Moderate asset price appreciation",
                "Normal market conditions"
            ])
        
        return features
    
    def _calculate_minimum_capital_ratio(self, capital_impact: Dict[str, Any]) -> float:
        """Calculate minimum capital ratio during stress period."""
        # Simplified calculation - would use actual RWA projections
        cet1_levels = capital_impact['capital_evolution']['cet1_capital']
        rwa = self.dfast_config.get('risk_weighted_assets', 200e9)
        
        capital_ratios = [capital / rwa for capital in cet1_levels]
        return min(capital_ratios)
    
    def generate_dfast_report(self) -> Dict[str, Any]:
        """Generate comprehensive DFAST compliance report."""
        if not self.stress_test_results:
            raise ValueError("No stress test results available. Run stress test first.")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'bank_category': self.bank_category,
                'total_assets': self.total_assets,
                'test_type': self.stress_test_results['test_metadata']['test_type']
            },
            'executive_summary': self._generate_executive_summary(),
            'company_run_results': self.stress_test_results['company_run_results'],
            'supervisory_coordination': self.stress_test_results.get('supervisory_coordination', {}),
            'public_disclosure_summary': self.stress_test_results.get('public_disclosure_data', {}),
            'regulatory_compliance': self._assess_dfast_compliance(),
            'recommendations': self._generate_dfast_recommendations()
        }
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of DFAST results."""
        company_results = self.stress_test_results['company_run_results']
        
        # Find most severe scenario results
        severely_adverse_results = None
        for scenario_name, results in company_results.items():
            if results['scenario_info']['type'] == 'severely_adverse':
                severely_adverse_results = results
                break
        
        if severely_adverse_results:
            capital_impact = severely_adverse_results['capital_impact']
            min_capital_ratio = self._calculate_minimum_capital_ratio(capital_impact)
            
            return {
                'overall_assessment': 'ADEQUATE' if min_capital_ratio > 0.045 else 'NEEDS_ATTENTION',
                'minimum_capital_ratio': min_capital_ratio,
                'total_capital_impact': capital_impact['total_capital_impact'],
                'key_vulnerabilities': self._identify_key_vulnerabilities(),
                'stress_test_date': self.stress_test_results['test_metadata']['test_date']
            }
        
        return {'message': 'Severely adverse scenario results not available'}
    
    def _assess_dfast_compliance(self) -> Dict[str, Any]:
        """Assess DFAST regulatory compliance."""
        compliance = {
            'compliance_date': datetime.now().isoformat(),
            'requirements_met': {},
            'deficiencies': [],
            'overall_status': 'COMPLIANT'
        }
        
        # Check asset threshold compliance
        compliance['requirements_met']['asset_threshold'] = self.total_assets >= self.requirements.asset_threshold
        if self.total_assets < self.requirements.asset_threshold:
            compliance['deficiencies'].append("Bank below DFAST asset threshold")
            compliance['overall_status'] = 'NOT_SUBJECT'
        
        # Check scenario coverage
        company_scenarios = set(self.stress_test_results['company_run_results'].keys())
        required_scenarios = {'baseline', 'adverse', 'severely_adverse'}
        
        compliance['requirements_met']['scenario_coverage'] = required_scenarios.issubset(company_scenarios)
        if not required_scenarios.issubset(company_scenarios):
            missing = required_scenarios - company_scenarios
            compliance['deficiencies'].append(f"Missing required scenarios: {missing}")
            compliance['overall_status'] = 'NON_COMPLIANT'
        
        # Check public disclosure preparation
        if self.requirements.public_disclosure_required:
            has_disclosure_data = 'public_disclosure_data' in self.stress_test_results
            compliance['requirements_met']['public_disclosure'] = has_disclosure_data
            if not has_disclosure_data:
                compliance['deficiencies'].append("Public disclosure data not prepared")
                compliance['overall_status'] = 'NON_COMPLIANT'
        
        return compliance
    
    def _generate_dfast_recommendations(self) -> List[str]:
        """Generate DFAST-specific recommendations."""
        recommendations = []
        
        # Analyze results for recommendations
        company_results = self.stress_test_results['company_run_results']
        
        for scenario_name, results in company_results.items():
            capital_impact = results['capital_impact']
            min_ratio = self._calculate_minimum_capital_ratio(capital_impact)
            
            if min_ratio < 0.065:  # Below well-capitalized threshold
                recommendations.append(f"Consider capital strengthening actions for {scenario_name} scenario")
            
            loss_summary = results['loss_summary']
            if loss_summary['peak_quarterly_loss_rate'] > 0.02:  # 2% quarterly loss rate
                recommendations.append(f"Review credit risk management for {scenario_name} scenario")
        
        # General DFAST recommendations
        recommendations.extend([
            "Enhance scenario development capabilities",
            "Strengthen model validation processes",
            "Improve public disclosure processes",
            "Coordinate with supervisory stress testing requirements"
        ])
        
        return recommendations
    
    def _identify_key_vulnerabilities(self) -> List[str]:
        """Identify key vulnerabilities from DFAST analysis."""
        return [
            "Credit losses from economic downturn",
            "Revenue pressure from adverse conditions",
            "Capital depletion under stress scenarios",
            "Market risk from trading activities"
        ]
    
    def export_dfast_results(self, filepath: str, export_type: str = 'summary') -> None:
        """Export DFAST results to file."""
        if not self.stress_test_results:
            raise ValueError("No stress test results to export")
        
        if export_type == 'full':
            report = self.generate_dfast_report()
        else:
            report = self._generate_summary_export()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            # Export as CSV summary
            self._export_csv_summary(filepath)
        
        self.logger.info(f"DFAST results exported to {filepath}")
    
    def _generate_summary_export(self) -> Dict[str, Any]:
        """Generate summary export data."""
        return {
            'test_metadata': self.stress_test_results['test_metadata'],
            'scenario_summaries': {
                scenario_name: {
                    'total_ppnr': results['ppnr_summary']['total_9_quarter'],
                    'total_losses': results['loss_summary']['total_9_quarter_provision'],
                    'capital_impact': results['capital_impact']['total_capital_impact']
                }
                for scenario_name, results in self.stress_test_results['company_run_results'].items()
            }
        }
    
    def _export_csv_summary(self, filepath: str) -> None:
        """Export summary results as CSV."""
        summary_data = []
        
        for scenario_name, results in self.stress_test_results['company_run_results'].items():
            summary_data.append({
                'scenario': scenario_name,
                'scenario_type': results['scenario_info']['type'],
                'total_ppnr_9q': results['ppnr_summary']['total_9_quarter'],
                'total_provision_9q': results['loss_summary']['total_9_quarter_provision'],
                'capital_impact': results['capital_impact']['total_capital_impact'],
                'min_capital_ratio': self._calculate_minimum_capital_ratio(results['capital_impact'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)