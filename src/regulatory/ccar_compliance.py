"""
CCAR (Comprehensive Capital Analysis and Review) Compliance Module

Implements Federal Reserve's CCAR requirements for large bank holding companies:
- Capital planning and stress testing
- PPNR projections under supervisory scenarios
- Capital adequacy assessment
- Regulatory reporting requirements
- Qualitative assessment support
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
class CCARScenario:
    """CCAR stress test scenario definition."""
    name: str
    description: str
    scenario_type: str  # 'baseline', 'adverse', 'severely_adverse'
    duration_quarters: int
    macroeconomic_variables: Dict[str, List[float]]
    market_shocks: Dict[str, List[float]]
    effective_date: datetime

@dataclass
class CCARCapitalRequirements:
    """CCAR capital requirements and thresholds."""
    cet1_minimum: float = 0.045  # 4.5%
    tier1_minimum: float = 0.06   # 6.0%
    total_capital_minimum: float = 0.08  # 8.0%
    leverage_ratio_minimum: float = 0.04  # 4.0%
    capital_conservation_buffer: float = 0.025  # 2.5%
    countercyclical_buffer: float = 0.0  # Variable
    gsib_buffer: float = 0.0  # Variable based on G-SIB score

class CCARCompliance:
    """
    CCAR compliance framework for PPNR models.
    
    Features:
    - Supervisory scenario implementation
    - Capital adequacy calculations
    - PPNR projections under stress
    - Regulatory reporting formats
    - Qualitative assessment support
    - Model validation requirements
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CCAR compliance framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ccar_config = config.get('ccar', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.CCARCompliance")
        
        # Initialize capital requirements
        self.capital_requirements = CCARCapitalRequirements()
        
        # Load supervisory scenarios
        self.supervisory_scenarios = self._load_supervisory_scenarios()
        
        # Results storage
        self.stress_test_results = {}
        self.capital_projections = {}
        self.compliance_status = {}
        
        # CCAR timeline (typically 9 quarters)
        self.projection_horizon = self.ccar_config.get('projection_horizon', 9)
        
        # Bank-specific parameters
        self.bank_tier = self.ccar_config.get('bank_tier', 'large_bank')  # large_bank, gsib
        self.total_assets = self.ccar_config.get('total_assets', 250e9)  # $250B threshold
        
        self.logger.info("CCAR compliance framework initialized")
    
    def _load_supervisory_scenarios(self) -> Dict[str, CCARScenario]:
        """Load Federal Reserve supervisory scenarios."""
        # 2024 CCAR supervisory scenarios (example)
        scenarios = {
            'baseline': CCARScenario(
                name='Baseline Scenario',
                description='Baseline economic conditions',
                scenario_type='baseline',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [2.1, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.8, 1.8],
                    'unemployment_rate': [3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.2, 4.2, 4.2],
                    'cpi_inflation': [2.4, 2.2, 2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    'fed_funds_rate': [5.25, 5.0, 4.75, 4.5, 4.25, 4.0, 3.75, 3.5, 3.5],
                    '10y_treasury': [4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.2],
                    'bbg_aaa_spread': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
                },
                market_shocks={
                    'equity_shock': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'vix_level': [18, 18, 18, 18, 18, 18, 18, 18, 18]
                },
                effective_date=datetime(2024, 1, 1)
            ),
            'adverse': CCARScenario(
                name='Adverse Scenario',
                description='Moderate recession with elevated unemployment',
                scenario_type='adverse',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [-1.2, -2.1, -0.8, 0.5, 1.2, 1.8, 2.0, 2.1, 2.1],
                    'unemployment_rate': [3.7, 4.5, 5.8, 7.2, 7.8, 7.5, 7.0, 6.5, 6.0],
                    'cpi_inflation': [2.4, 1.8, 1.2, 0.8, 1.0, 1.5, 1.8, 2.0, 2.0],
                    'fed_funds_rate': [5.25, 4.0, 2.5, 1.0, 0.5, 0.5, 1.0, 1.5, 2.0],
                    '10y_treasury': [4.2, 3.2, 2.1, 1.5, 1.8, 2.2, 2.5, 2.8, 3.0],
                    'bbg_aaa_spread': [0.8, 1.2, 1.8, 2.2, 2.0, 1.6, 1.3, 1.1, 1.0]
                },
                market_shocks={
                    'equity_shock': [0.0, -15.0, -25.0, -20.0, -10.0, 0.0, 5.0, 8.0, 10.0],
                    'vix_level': [18, 28, 35, 32, 25, 22, 20, 19, 18]
                },
                effective_date=datetime(2024, 1, 1)
            ),
            'severely_adverse': CCARScenario(
                name='Severely Adverse Scenario',
                description='Severe global recession with financial market stress',
                scenario_type='severely_adverse',
                duration_quarters=9,
                macroeconomic_variables={
                    'real_gdp_growth': [-2.8, -4.2, -2.1, -0.5, 1.0, 2.2, 2.8, 2.5, 2.3],
                    'unemployment_rate': [3.7, 5.2, 7.8, 10.0, 10.8, 10.2, 9.1, 8.0, 7.2],
                    'cpi_inflation': [2.4, 1.2, 0.2, -0.5, -0.2, 0.8, 1.5, 1.8, 2.0],
                    'fed_funds_rate': [5.25, 2.5, 0.5, 0.1, 0.1, 0.1, 0.5, 1.0, 1.5],
                    '10y_treasury': [4.2, 2.8, 1.2, 0.8, 1.2, 1.8, 2.2, 2.5, 2.8],
                    'bbg_aaa_spread': [0.8, 1.8, 3.2, 4.5, 3.8, 2.8, 2.2, 1.8, 1.5]
                },
                market_shocks={
                    'equity_shock': [0.0, -25.0, -45.0, -40.0, -25.0, -10.0, 5.0, 12.0, 15.0],
                    'vix_level': [18, 35, 55, 48, 35, 28, 25, 22, 20]
                },
                effective_date=datetime(2024, 1, 1)
            )
        }
        
        return scenarios
    
    def run_ccar_stress_test(self, ppnr_models: Dict[str, Any],
                           initial_capital: Dict[str, float],
                           balance_sheet_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive CCAR stress test.
        
        Args:
            ppnr_models: Dictionary of PPNR models (NII, fee income, trading)
            initial_capital: Initial capital positions
            balance_sheet_data: Historical balance sheet data
            
        Returns:
            CCAR stress test results
        """
        self.logger.info("Running CCAR stress test...")
        
        results = {
            'test_date': datetime.now().isoformat(),
            'scenarios': {},
            'capital_projections': {},
            'compliance_assessment': {},
            'regulatory_ratios': {},
            'summary_metrics': {}
        }
        
        # Run stress test for each supervisory scenario
        for scenario_name, scenario in self.supervisory_scenarios.items():
            self.logger.info(f"Processing {scenario_name} scenario...")
            
            scenario_results = self._run_scenario_stress_test(
                scenario, ppnr_models, initial_capital, balance_sheet_data
            )
            
            results['scenarios'][scenario_name] = scenario_results
        
        # Calculate capital projections
        results['capital_projections'] = self._calculate_capital_projections(
            results['scenarios'], initial_capital
        )
        
        # Assess regulatory compliance
        results['compliance_assessment'] = self._assess_regulatory_compliance(
            results['capital_projections']
        )
        
        # Calculate regulatory ratios
        results['regulatory_ratios'] = self._calculate_regulatory_ratios(
            results['capital_projections']
        )
        
        # Generate summary metrics
        results['summary_metrics'] = self._generate_summary_metrics(results)
        
        # Store results
        self.stress_test_results = results
        
        self.logger.info("CCAR stress test completed")
        return results
    
    def _run_scenario_stress_test(self, scenario: CCARScenario,
                                ppnr_models: Dict[str, Any],
                                initial_capital: Dict[str, float],
                                balance_sheet_data: pd.DataFrame) -> Dict[str, Any]:
        """Run stress test for a specific scenario."""
        scenario_results = {
            'scenario_info': {
                'name': scenario.name,
                'type': scenario.scenario_type,
                'duration': scenario.duration_quarters
            },
            'ppnr_projections': {},
            'balance_sheet_projections': {},
            'loss_projections': {},
            'capital_impact': {}
        }
        
        # Generate quarterly projections
        quarterly_projections = []
        
        for quarter in range(scenario.duration_quarters):
            # Extract scenario variables for this quarter
            scenario_vars = self._extract_scenario_variables(scenario, quarter)
            
            # Project PPNR components
            ppnr_projection = self._project_ppnr_components(
                ppnr_models, scenario_vars, quarter
            )
            
            # Project balance sheet
            balance_sheet_projection = self._project_balance_sheet(
                balance_sheet_data, scenario_vars, quarter
            )
            
            # Project losses
            loss_projection = self._project_losses(
                balance_sheet_projection, scenario_vars, quarter
            )
            
            quarterly_projections.append({
                'quarter': quarter + 1,
                'scenario_variables': scenario_vars,
                'ppnr': ppnr_projection,
                'balance_sheet': balance_sheet_projection,
                'losses': loss_projection
            })
        
        scenario_results['quarterly_projections'] = quarterly_projections
        
        # Aggregate results
        scenario_results['ppnr_projections'] = self._aggregate_ppnr_projections(quarterly_projections)
        scenario_results['loss_projections'] = self._aggregate_loss_projections(quarterly_projections)
        scenario_results['capital_impact'] = self._calculate_capital_impact(
            scenario_results['ppnr_projections'],
            scenario_results['loss_projections'],
            initial_capital
        )
        
        return scenario_results
    
    def _extract_scenario_variables(self, scenario: CCARScenario, quarter: int) -> Dict[str, float]:
        """Extract scenario variables for a specific quarter."""
        variables = {}
        
        # Extract macroeconomic variables
        for var_name, values in scenario.macroeconomic_variables.items():
            if quarter < len(values):
                variables[var_name] = values[quarter]
            else:
                variables[var_name] = values[-1]  # Use last available value
        
        # Extract market shocks
        for var_name, values in scenario.market_shocks.items():
            if quarter < len(values):
                variables[var_name] = values[quarter]
            else:
                variables[var_name] = values[-1]
        
        return variables
    
    def _project_ppnr_components(self, ppnr_models: Dict[str, Any],
                               scenario_vars: Dict[str, float],
                               quarter: int) -> Dict[str, float]:
        """Project PPNR components under scenario conditions."""
        ppnr_projection = {}
        
        # Net Interest Income projection
        if 'nii_model' in ppnr_models:
            nii_features = self._prepare_nii_features(scenario_vars, quarter)
            nii_projection = ppnr_models['nii_model'].predict_scenario(nii_features, scenario_vars)
            ppnr_projection['net_interest_income'] = float(nii_projection)
        
        # Fee Income projection
        if 'fee_income_model' in ppnr_models:
            fee_features = self._prepare_fee_features(scenario_vars, quarter)
            fee_projection = ppnr_models['fee_income_model'].predict_scenario(fee_features, scenario_vars)
            ppnr_projection['fee_income'] = float(fee_projection)
        
        # Trading Revenue projection
        if 'trading_model' in ppnr_models:
            trading_features = self._prepare_trading_features(scenario_vars, quarter)
            trading_projection = ppnr_models['trading_model'].predict_scenario(trading_features, scenario_vars)
            ppnr_projection['trading_revenue'] = float(trading_projection)
        
        # Calculate total PPNR
        ppnr_projection['total_ppnr'] = sum([
            ppnr_projection.get('net_interest_income', 0),
            ppnr_projection.get('fee_income', 0),
            ppnr_projection.get('trading_revenue', 0)
        ])
        
        return ppnr_projection
    
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
            'equity_shock': scenario_vars.get('equity_shock', 0),
            'vix_level': scenario_vars.get('vix_level', 0),
            'quarter': quarter + 1,
            'fed_funds_rate': scenario_vars.get('fed_funds_rate', 0)
        }
    
    def _prepare_trading_features(self, scenario_vars: Dict[str, float], quarter: int) -> Dict[str, float]:
        """Prepare features for trading revenue model prediction."""
        return {
            'equity_shock': scenario_vars.get('equity_shock', 0),
            'vix_level': scenario_vars.get('vix_level', 0),
            'credit_spread': scenario_vars.get('bbg_aaa_spread', 0),
            'interest_rate_shock': scenario_vars.get('fed_funds_rate', 0) - 5.25,  # Shock from current level
            'quarter': quarter + 1,
            'gdp_growth': scenario_vars.get('real_gdp_growth', 0)
        }
    
    def _project_balance_sheet(self, balance_sheet_data: pd.DataFrame,
                             scenario_vars: Dict[str, float],
                             quarter: int) -> Dict[str, float]:
        """Project balance sheet items under scenario conditions."""
        # Simplified balance sheet projection
        # In practice, this would use sophisticated balance sheet models
        
        latest_bs = balance_sheet_data.iloc[-1].to_dict()
        
        # Apply growth/contraction based on economic conditions
        gdp_growth = scenario_vars.get('real_gdp_growth', 0) / 100
        unemployment_rate = scenario_vars.get('unemployment_rate', 0) / 100
        
        # Loan growth adjustment
        loan_growth_factor = 1 + (gdp_growth * 0.5 - (unemployment_rate - 0.04) * 0.3) / 4  # Quarterly
        
        projection = {
            'total_assets': latest_bs.get('total_assets', 0) * (1 + gdp_growth / 4),
            'total_loans': latest_bs.get('total_loans', 0) * loan_growth_factor,
            'securities': latest_bs.get('securities', 0) * (1 + gdp_growth / 8),
            'deposits': latest_bs.get('deposits', 0) * (1 + gdp_growth / 4),
            'shareholders_equity': latest_bs.get('shareholders_equity', 0)  # Will be updated with earnings
        }
        
        return projection
    
    def _project_losses(self, balance_sheet_projection: Dict[str, float],
                       scenario_vars: Dict[str, float],
                       quarter: int) -> Dict[str, float]:
        """Project credit losses under scenario conditions."""
        # Simplified loss projection
        # In practice, this would use detailed credit risk models
        
        unemployment_rate = scenario_vars.get('unemployment_rate', 0) / 100
        gdp_growth = scenario_vars.get('real_gdp_growth', 0) / 100
        
        # Base loss rate adjustment
        base_loss_rate = 0.005  # 50 bps base quarterly loss rate
        
        # Stress adjustment
        unemployment_stress = max(0, unemployment_rate - 0.04) * 2  # Stress above 4%
        gdp_stress = max(0, -gdp_growth) * 1.5  # Stress for negative growth
        
        stressed_loss_rate = base_loss_rate + unemployment_stress + gdp_stress
        
        total_loans = balance_sheet_projection.get('total_loans', 0)
        
        return {
            'provision_for_credit_losses': total_loans * stressed_loss_rate,
            'net_charge_offs': total_loans * stressed_loss_rate * 0.8,
            'loss_rate': stressed_loss_rate
        }
    
    def _aggregate_ppnr_projections(self, quarterly_projections: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Aggregate PPNR projections across quarters."""
        aggregated = {
            'net_interest_income': [],
            'fee_income': [],
            'trading_revenue': [],
            'total_ppnr': []
        }
        
        for projection in quarterly_projections:
            ppnr = projection['ppnr']
            for component in aggregated.keys():
                aggregated[component].append(ppnr.get(component, 0))
        
        return aggregated
    
    def _aggregate_loss_projections(self, quarterly_projections: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Aggregate loss projections across quarters."""
        aggregated = {
            'provision_for_credit_losses': [],
            'net_charge_offs': [],
            'loss_rate': []
        }
        
        for projection in quarterly_projections:
            losses = projection['losses']
            for component in aggregated.keys():
                aggregated[component].append(losses.get(component, 0))
        
        return aggregated
    
    def _calculate_capital_impact(self, ppnr_projections: Dict[str, List[float]],
                                loss_projections: Dict[str, List[float]],
                                initial_capital: Dict[str, float]) -> Dict[str, Any]:
        """Calculate capital impact from PPNR and losses."""
        quarters = len(ppnr_projections['total_ppnr'])
        
        # Initialize capital tracking
        capital_evolution = {
            'cet1_capital': [initial_capital.get('cet1_capital', 0)],
            'tier1_capital': [initial_capital.get('tier1_capital', 0)],
            'total_capital': [initial_capital.get('total_capital', 0)],
            'retained_earnings': [0]
        }
        
        # Tax rate
        tax_rate = 0.21
        
        # Calculate quarterly capital evolution
        for q in range(quarters):
            # Pre-tax income
            pre_tax_income = (
                ppnr_projections['total_ppnr'][q] - 
                loss_projections['provision_for_credit_losses'][q]
            )
            
            # After-tax income
            after_tax_income = pre_tax_income * (1 - tax_rate)
            
            # Retained earnings (assume 50% payout ratio)
            retained_earnings = after_tax_income * 0.5
            capital_evolution['retained_earnings'].append(retained_earnings)
            
            # Update capital levels
            new_cet1 = capital_evolution['cet1_capital'][-1] + retained_earnings
            new_tier1 = capital_evolution['tier1_capital'][-1] + retained_earnings
            new_total = capital_evolution['total_capital'][-1] + retained_earnings
            
            capital_evolution['cet1_capital'].append(new_cet1)
            capital_evolution['tier1_capital'].append(new_tier1)
            capital_evolution['total_capital'].append(new_total)
        
        return {
            'capital_evolution': capital_evolution,
            'total_retained_earnings': sum(capital_evolution['retained_earnings']),
            'final_capital_levels': {
                'cet1_capital': capital_evolution['cet1_capital'][-1],
                'tier1_capital': capital_evolution['tier1_capital'][-1],
                'total_capital': capital_evolution['total_capital'][-1]
            }
        }
    
    def _calculate_capital_projections(self, scenario_results: Dict[str, Any],
                                     initial_capital: Dict[str, float]) -> Dict[str, Any]:
        """Calculate capital projections across all scenarios."""
        projections = {}
        
        for scenario_name, results in scenario_results.items():
            capital_impact = results['capital_impact']
            projections[scenario_name] = capital_impact['capital_evolution']
        
        return projections
    
    def _assess_regulatory_compliance(self, capital_projections: Dict[str, Any]) -> Dict[str, Any]:
        """Assess regulatory compliance across scenarios."""
        compliance = {}
        
        # Assumed risk-weighted assets (would be calculated from balance sheet)
        rwa = self.ccar_config.get('risk_weighted_assets', 200e9)
        total_assets = self.total_assets
        
        for scenario_name, projections in capital_projections.items():
            scenario_compliance = {
                'cet1_ratio_compliance': [],
                'tier1_ratio_compliance': [],
                'total_capital_ratio_compliance': [],
                'leverage_ratio_compliance': [],
                'minimum_ratios_met': []
            }
            
            # Check compliance for each quarter
            for q in range(len(projections['cet1_capital'])):
                cet1_capital = projections['cet1_capital'][q]
                tier1_capital = projections['tier1_capital'][q]
                total_capital = projections['total_capital'][q]
                
                # Calculate ratios
                cet1_ratio = cet1_capital / rwa
                tier1_ratio = tier1_capital / rwa
                total_capital_ratio = total_capital / rwa
                leverage_ratio = tier1_capital / total_assets
                
                # Check compliance
                cet1_compliant = cet1_ratio >= (self.capital_requirements.cet1_minimum + 
                                              self.capital_requirements.capital_conservation_buffer)
                tier1_compliant = tier1_ratio >= (self.capital_requirements.tier1_minimum + 
                                                self.capital_requirements.capital_conservation_buffer)
                total_capital_compliant = total_capital_ratio >= (self.capital_requirements.total_capital_minimum + 
                                                                self.capital_requirements.capital_conservation_buffer)
                leverage_compliant = leverage_ratio >= self.capital_requirements.leverage_ratio_minimum
                
                scenario_compliance['cet1_ratio_compliance'].append(cet1_compliant)
                scenario_compliance['tier1_ratio_compliance'].append(tier1_compliant)
                scenario_compliance['total_capital_ratio_compliance'].append(total_capital_compliant)
                scenario_compliance['leverage_ratio_compliance'].append(leverage_compliant)
                scenario_compliance['minimum_ratios_met'].append(
                    all([cet1_compliant, tier1_compliant, total_capital_compliant, leverage_compliant])
                )
            
            compliance[scenario_name] = scenario_compliance
        
        return compliance
    
    def _calculate_regulatory_ratios(self, capital_projections: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate regulatory capital ratios."""
        ratios = {}
        
        # Assumed values (would be calculated from actual data)
        rwa = self.ccar_config.get('risk_weighted_assets', 200e9)
        total_assets = self.total_assets
        
        for scenario_name, projections in capital_projections.items():
            scenario_ratios = {
                'cet1_ratios': [],
                'tier1_ratios': [],
                'total_capital_ratios': [],
                'leverage_ratios': []
            }
            
            for q in range(len(projections['cet1_capital'])):
                cet1_capital = projections['cet1_capital'][q]
                tier1_capital = projections['tier1_capital'][q]
                total_capital = projections['total_capital'][q]
                
                scenario_ratios['cet1_ratios'].append(cet1_capital / rwa)
                scenario_ratios['tier1_ratios'].append(tier1_capital / rwa)
                scenario_ratios['total_capital_ratios'].append(total_capital / rwa)
                scenario_ratios['leverage_ratios'].append(tier1_capital / total_assets)
            
            ratios[scenario_name] = scenario_ratios
        
        return ratios
    
    def _generate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary metrics for CCAR results."""
        summary = {
            'scenarios_tested': len(results['scenarios']),
            'projection_horizon_quarters': self.projection_horizon,
            'minimum_capital_ratios': {},
            'capital_depletion': {},
            'stress_impact': {}
        }
        
        # Find minimum capital ratios across scenarios
        for scenario_name in results['regulatory_ratios'].keys():
            ratios = results['regulatory_ratios'][scenario_name]
            summary['minimum_capital_ratios'][scenario_name] = {
                'min_cet1_ratio': min(ratios['cet1_ratios']),
                'min_tier1_ratio': min(ratios['tier1_ratios']),
                'min_total_capital_ratio': min(ratios['total_capital_ratios']),
                'min_leverage_ratio': min(ratios['leverage_ratios'])
            }
        
        # Calculate capital depletion
        for scenario_name in results['capital_projections'].keys():
            projections = results['capital_projections'][scenario_name]
            initial_cet1 = projections['cet1_capital'][0]
            final_cet1 = projections['cet1_capital'][-1]
            
            summary['capital_depletion'][scenario_name] = {
                'absolute_change': final_cet1 - initial_cet1,
                'percentage_change': (final_cet1 - initial_cet1) / initial_cet1 * 100
            }
        
        return summary
    
    def generate_ccar_report(self) -> Dict[str, Any]:
        """Generate comprehensive CCAR compliance report."""
        if not self.stress_test_results:
            raise ValueError("No stress test results available. Run stress test first.")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'bank_tier': self.bank_tier,
                'total_assets': self.total_assets,
                'projection_horizon': self.projection_horizon
            },
            'executive_summary': self._generate_executive_summary(),
            'scenario_results': self.stress_test_results['scenarios'],
            'capital_adequacy': self._generate_capital_adequacy_summary(),
            'compliance_status': self._generate_compliance_summary(),
            'risk_assessment': self._generate_risk_assessment(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of CCAR results."""
        summary_metrics = self.stress_test_results['summary_metrics']
        
        # Find most stressed scenario
        min_cet1_ratios = {
            scenario: metrics['min_cet1_ratio'] 
            for scenario, metrics in summary_metrics['minimum_capital_ratios'].items()
        }
        most_stressed_scenario = min(min_cet1_ratios, key=min_cet1_ratios.get)
        min_cet1_ratio = min_cet1_ratios[most_stressed_scenario]
        
        return {
            'overall_assessment': 'PASS' if min_cet1_ratio > 0.07 else 'CONDITIONAL_PASS' if min_cet1_ratio > 0.045 else 'FAIL',
            'minimum_cet1_ratio': min_cet1_ratio,
            'most_stressed_scenario': most_stressed_scenario,
            'capital_buffer_above_minimum': min_cet1_ratio - 0.045,
            'scenarios_tested': summary_metrics['scenarios_tested'],
            'key_risks_identified': self._identify_key_risks()
        }
    
    def _generate_capital_adequacy_summary(self) -> Dict[str, Any]:
        """Generate capital adequacy summary."""
        regulatory_ratios = self.stress_test_results['regulatory_ratios']
        
        adequacy_summary = {}
        
        for scenario_name, ratios in regulatory_ratios.items():
            adequacy_summary[scenario_name] = {
                'minimum_cet1_ratio': min(ratios['cet1_ratios']),
                'minimum_tier1_ratio': min(ratios['tier1_ratios']),
                'minimum_total_capital_ratio': min(ratios['total_capital_ratios']),
                'minimum_leverage_ratio': min(ratios['leverage_ratios']),
                'quarters_below_well_capitalized': sum([
                    1 for ratio in ratios['cet1_ratios'] if ratio < 0.065
                ])
            }
        
        return adequacy_summary
    
    def _generate_compliance_summary(self) -> Dict[str, Any]:
        """Generate regulatory compliance summary."""
        compliance = self.stress_test_results['compliance_assessment']
        
        compliance_summary = {}
        
        for scenario_name, scenario_compliance in compliance.items():
            compliance_summary[scenario_name] = {
                'quarters_compliant': sum(scenario_compliance['minimum_ratios_met']),
                'total_quarters': len(scenario_compliance['minimum_ratios_met']),
                'compliance_rate': sum(scenario_compliance['minimum_ratios_met']) / len(scenario_compliance['minimum_ratios_met']),
                'first_breach_quarter': next(
                    (i+1 for i, compliant in enumerate(scenario_compliance['minimum_ratios_met']) if not compliant),
                    None
                )
            }
        
        return compliance_summary
    
    def _generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate risk assessment based on CCAR results."""
        return {
            'credit_risk_assessment': 'Moderate stress from unemployment increases',
            'market_risk_assessment': 'Significant trading revenue volatility under adverse scenarios',
            'interest_rate_risk_assessment': 'NII pressure from rate cuts in stress scenarios',
            'operational_risk_assessment': 'Stable operational risk profile',
            'concentration_risk_assessment': 'Diversified revenue streams provide resilience'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on CCAR results."""
        recommendations = []
        
        # Check minimum ratios
        summary_metrics = self.stress_test_results['summary_metrics']
        min_cet1_ratios = summary_metrics['minimum_capital_ratios']
        
        for scenario, metrics in min_cet1_ratios.items():
            if metrics['min_cet1_ratio'] < 0.07:
                recommendations.append(f"Consider capital actions to improve CET1 ratio in {scenario} scenario")
        
        # Check capital depletion
        capital_depletion = summary_metrics['capital_depletion']
        for scenario, depletion in capital_depletion.items():
            if depletion['percentage_change'] < -20:
                recommendations.append(f"Review capital planning for {scenario} scenario (high depletion)")
        
        # General recommendations
        recommendations.extend([
            "Maintain strong capital buffers above regulatory minimums",
            "Continue monitoring economic indicators for early warning signals",
            "Enhance stress testing capabilities for emerging risks",
            "Review dividend policy in light of stress test results"
        ])
        
        return recommendations
    
    def _identify_key_risks(self) -> List[str]:
        """Identify key risks from CCAR analysis."""
        return [
            "Credit losses from unemployment increases",
            "NII compression from interest rate cuts",
            "Trading revenue volatility from market stress",
            "Fee income pressure from economic downturn"
        ]
    
    def export_ccar_results(self, filepath: str) -> None:
        """Export CCAR results to file."""
        if not self.stress_test_results:
            raise ValueError("No stress test results to export")
        
        report = self.generate_ccar_report()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            # Export summary as CSV
            summary_data = []
            for scenario_name in report['scenario_results'].keys():
                capital_adequacy = report['capital_adequacy'][scenario_name]
                compliance_status = report['compliance_status'][scenario_name]
                
                summary_data.append({
                    'scenario': scenario_name,
                    'min_cet1_ratio': capital_adequacy['minimum_cet1_ratio'],
                    'min_tier1_ratio': capital_adequacy['minimum_tier1_ratio'],
                    'min_leverage_ratio': capital_adequacy['minimum_leverage_ratio'],
                    'compliance_rate': compliance_status['compliance_rate'],
                    'first_breach_quarter': compliance_status['first_breach_quarter']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(filepath, index=False)
        
        self.logger.info(f"CCAR results exported to {filepath}")
    
    def validate_ccar_compliance(self) -> Dict[str, Any]:
        """Validate CCAR compliance requirements."""
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'requirements_met': {},
            'deficiencies': [],
            'overall_status': 'COMPLIANT'
        }
        
        if not self.stress_test_results:
            validation_results['deficiencies'].append("No stress test results available")
            validation_results['overall_status'] = 'NON_COMPLIANT'
            return validation_results
        
        # Check scenario coverage
        required_scenarios = {'baseline', 'adverse', 'severely_adverse'}
        tested_scenarios = set(self.stress_test_results['scenarios'].keys())
        
        if not required_scenarios.issubset(tested_scenarios):
            missing_scenarios = required_scenarios - tested_scenarios
            validation_results['deficiencies'].append(f"Missing required scenarios: {missing_scenarios}")
            validation_results['overall_status'] = 'NON_COMPLIANT'
        
        validation_results['requirements_met']['scenario_coverage'] = required_scenarios.issubset(tested_scenarios)
        
        # Check projection horizon
        expected_horizon = 9  # quarters
        actual_horizon = self.projection_horizon
        
        validation_results['requirements_met']['projection_horizon'] = actual_horizon >= expected_horizon
        if actual_horizon < expected_horizon:
            validation_results['deficiencies'].append(f"Insufficient projection horizon: {actual_horizon} < {expected_horizon}")
            validation_results['overall_status'] = 'NON_COMPLIANT'
        
        # Check capital ratio calculations
        validation_results['requirements_met']['capital_ratios'] = 'regulatory_ratios' in self.stress_test_results
        if 'regulatory_ratios' not in self.stress_test_results:
            validation_results['deficiencies'].append("Missing capital ratio calculations")
            validation_results['overall_status'] = 'NON_COMPLIANT'
        
        return validation_results