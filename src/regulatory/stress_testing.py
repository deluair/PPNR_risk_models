"""
Stress Testing Framework Module

This module provides a comprehensive framework for conducting various types
of stress tests including regulatory, internal, and scenario-based testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Types of stress tests"""
    REGULATORY = "regulatory"
    INTERNAL = "internal"
    REVERSE = "reverse"
    SENSITIVITY = "sensitivity"
    SCENARIO = "scenario"


class RiskFactor(Enum):
    """Risk factors for stress testing"""
    INTEREST_RATE = "interest_rate"
    CREDIT_SPREAD = "credit_spread"
    EQUITY_PRICE = "equity_price"
    FX_RATE = "fx_rate"
    COMMODITY_PRICE = "commodity_price"
    HOUSE_PRICE = "house_price"
    GDP_GROWTH = "gdp_growth"
    UNEMPLOYMENT = "unemployment"
    INFLATION = "inflation"


@dataclass
class StressScenarioDefinition:
    """Definition of a stress scenario"""
    name: str
    description: str
    risk_factors: Dict[RiskFactor, List[float]]
    time_horizon: int  # quarters
    severity: str  # mild, moderate, severe
    probability: Optional[float] = None


@dataclass
class StressTestConfiguration:
    """Configuration for stress testing"""
    test_type: StressTestType
    scenarios: List[StressScenarioDefinition]
    portfolios: List[str]
    risk_factors: List[RiskFactor]
    time_horizon: int
    confidence_level: float = 0.99
    monte_carlo_simulations: int = 1000


@dataclass
class PortfolioStressResult:
    """Stress test results for a portfolio"""
    portfolio_name: str
    scenario_name: str
    base_value: float
    stressed_value: float
    absolute_loss: float
    relative_loss: float
    var_estimate: float
    expected_shortfall: float
    risk_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestReport:
    """Comprehensive stress test report"""
    test_date: datetime
    test_type: StressTestType
    scenarios_tested: List[str]
    portfolio_results: List[PortfolioStressResult]
    aggregate_results: Dict[str, float]
    risk_metrics: Dict[str, float]
    recommendations: List[str]
    pass_fail_status: Optional[bool] = None


class StressTestEngine(ABC):
    """Abstract base class for stress test engines"""
    
    @abstractmethod
    def apply_scenario(self, 
                      portfolio_data: pd.DataFrame,
                      scenario: StressScenarioDefinition) -> PortfolioStressResult:
        """Apply stress scenario to portfolio"""
        pass
    
    @abstractmethod
    def calculate_risk_metrics(self, results: List[PortfolioStressResult]) -> Dict[str, float]:
        """Calculate risk metrics from stress results"""
        pass


class CreditStressEngine(StressTestEngine):
    """Stress test engine for credit portfolios"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_pd_curves = {
            'AAA': [0.001, 0.002, 0.003],
            'AA': [0.002, 0.004, 0.006],
            'A': [0.005, 0.010, 0.015],
            'BBB': [0.015, 0.030, 0.045],
            'BB': [0.050, 0.100, 0.150],
            'B': [0.150, 0.250, 0.350],
            'CCC': [0.300, 0.450, 0.600]
        }
    
    def apply_scenario(self, 
                      portfolio_data: pd.DataFrame,
                      scenario: StressScenarioDefinition) -> PortfolioStressResult:
        """Apply credit stress scenario to portfolio"""
        try:
            base_value = portfolio_data['exposure_amount'].sum()
            total_loss = 0
            risk_contributions = {}
            
            # Apply stress to each exposure
            for _, exposure in portfolio_data.iterrows():
                rating = exposure.get('rating', 'BBB')
                exposure_amount = exposure['exposure_amount']
                lgd = exposure.get('lgd', 0.45)
                
                # Get base PD and apply stress
                base_pd = self._get_base_pd(rating)
                stressed_pd = self._apply_stress_to_pd(base_pd, scenario)
                
                # Calculate expected loss
                expected_loss = exposure_amount * stressed_pd * lgd
                total_loss += expected_loss
                
                # Track risk contributions
                sector = exposure.get('sector', 'Other')
                risk_contributions[sector] = risk_contributions.get(sector, 0) + expected_loss
            
            stressed_value = base_value - total_loss
            relative_loss = total_loss / base_value if base_value > 0 else 0
            
            # Calculate VaR and ES (simplified)
            var_estimate = total_loss * 1.5  # Simplified VaR calculation
            expected_shortfall = total_loss * 2.0  # Simplified ES calculation
            
            return PortfolioStressResult(
                portfolio_name="Credit Portfolio",
                scenario_name=scenario.name,
                base_value=base_value,
                stressed_value=stressed_value,
                absolute_loss=total_loss,
                relative_loss=relative_loss,
                var_estimate=var_estimate,
                expected_shortfall=expected_shortfall,
                risk_contributions=risk_contributions
            )
            
        except Exception as e:
            logger.error(f"Error applying credit stress scenario: {str(e)}")
            raise
    
    def calculate_risk_metrics(self, results: List[PortfolioStressResult]) -> Dict[str, float]:
        """Calculate credit risk metrics"""
        if not results:
            return {}
        
        total_losses = [r.absolute_loss for r in results]
        relative_losses = [r.relative_loss for r in results]
        
        return {
            'max_absolute_loss': max(total_losses),
            'max_relative_loss': max(relative_losses),
            'average_loss': np.mean(total_losses),
            'loss_volatility': np.std(total_losses),
            'var_95': np.percentile(total_losses, 95),
            'var_99': np.percentile(total_losses, 99)
        }
    
    def _get_base_pd(self, rating: str) -> float:
        """Get base probability of default for rating"""
        return self.default_pd_curves.get(rating, [0.05])[0]
    
    def _apply_stress_to_pd(self, base_pd: float, scenario: StressScenarioDefinition) -> float:
        """Apply stress scenario to probability of default"""
        stress_multiplier = 1.0
        
        # GDP impact
        if RiskFactor.GDP_GROWTH in scenario.risk_factors:
            gdp_stress = scenario.risk_factors[RiskFactor.GDP_GROWTH][0]
            if gdp_stress < 0:
                stress_multiplier += abs(gdp_stress) * 2  # 2x multiplier for negative GDP
        
        # Unemployment impact
        if RiskFactor.UNEMPLOYMENT in scenario.risk_factors:
            unemployment_stress = scenario.risk_factors[RiskFactor.UNEMPLOYMENT][0]
            if unemployment_stress > 5.0:  # Above normal unemployment
                stress_multiplier += (unemployment_stress - 5.0) * 0.5
        
        return min(base_pd * stress_multiplier, 1.0)  # Cap at 100%


class MarketStressEngine(StressTestEngine):
    """Stress test engine for market risk portfolios"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.risk_factor_sensitivities = {
            RiskFactor.INTEREST_RATE: {'bonds': -5.0, 'loans': 2.0},
            RiskFactor.EQUITY_PRICE: {'equities': 1.0, 'equity_derivatives': 1.2},
            RiskFactor.FX_RATE: {'fx_positions': 1.0, 'foreign_bonds': 0.3},
            RiskFactor.CREDIT_SPREAD: {'corporate_bonds': -3.0, 'credit_derivatives': -2.5}
        }
    
    def apply_scenario(self, 
                      portfolio_data: pd.DataFrame,
                      scenario: StressScenarioDefinition) -> PortfolioStressResult:
        """Apply market stress scenario to portfolio"""
        try:
            base_value = portfolio_data['market_value'].sum()
            total_pnl = 0
            risk_contributions = {}
            
            # Apply stress to each position
            for _, position in portfolio_data.iterrows():
                instrument_type = position.get('instrument_type', 'other')
                market_value = position['market_value']
                position_pnl = 0
                
                # Apply each risk factor stress
                for risk_factor, stress_values in scenario.risk_factors.items():
                    if risk_factor in self.risk_factor_sensitivities:
                        sensitivity = self._get_sensitivity(instrument_type, risk_factor)
                        if sensitivity != 0:
                            stress_impact = stress_values[0] / 100  # Convert to decimal
                            factor_pnl = market_value * sensitivity * stress_impact
                            position_pnl += factor_pnl
                
                total_pnl += position_pnl
                
                # Track risk contributions by instrument type
                risk_contributions[instrument_type] = (
                    risk_contributions.get(instrument_type, 0) + position_pnl
                )
            
            stressed_value = base_value + total_pnl
            relative_loss = -total_pnl / base_value if base_value > 0 else 0
            
            # Calculate VaR and ES using historical simulation approach
            var_estimate = abs(total_pnl) * 1.3
            expected_shortfall = abs(total_pnl) * 1.8
            
            return PortfolioStressResult(
                portfolio_name="Market Portfolio",
                scenario_name=scenario.name,
                base_value=base_value,
                stressed_value=stressed_value,
                absolute_loss=-total_pnl,  # Negative PnL is loss
                relative_loss=relative_loss,
                var_estimate=var_estimate,
                expected_shortfall=expected_shortfall,
                risk_contributions=risk_contributions
            )
            
        except Exception as e:
            logger.error(f"Error applying market stress scenario: {str(e)}")
            raise
    
    def calculate_risk_metrics(self, results: List[PortfolioStressResult]) -> Dict[str, float]:
        """Calculate market risk metrics"""
        if not results:
            return {}
        
        pnl_values = [r.absolute_loss for r in results]
        
        return {
            'worst_case_loss': max(pnl_values),
            'best_case_gain': min(pnl_values),
            'average_pnl': np.mean(pnl_values),
            'pnl_volatility': np.std(pnl_values),
            'var_95': np.percentile(pnl_values, 95),
            'var_99': np.percentile(pnl_values, 99),
            'expected_shortfall_95': np.mean([x for x in pnl_values if x >= np.percentile(pnl_values, 95)])
        }
    
    def _get_sensitivity(self, instrument_type: str, risk_factor: RiskFactor) -> float:
        """Get sensitivity of instrument to risk factor"""
        if risk_factor in self.risk_factor_sensitivities:
            sensitivities = self.risk_factor_sensitivities[risk_factor]
            return sensitivities.get(instrument_type, 0.0)
        return 0.0


class StressTestFramework:
    """
    Comprehensive stress testing framework
    
    Orchestrates stress testing across different risk types and portfolios
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize stress testing framework
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.engines = {
            'credit': CreditStressEngine(config),
            'market': MarketStressEngine(config)
        }
        
        # Predefined scenarios
        self.standard_scenarios = self._create_standard_scenarios()
    
    def run_stress_test(self,
                       portfolio_data: Dict[str, pd.DataFrame],
                       configuration: StressTestConfiguration) -> StressTestReport:
        """
        Run comprehensive stress test
        
        Args:
            portfolio_data: Dictionary of portfolio DataFrames by type
            configuration: Stress test configuration
            
        Returns:
            Comprehensive stress test report
        """
        try:
            logger.info(f"Running {configuration.test_type.value} stress test")
            
            all_results = []
            
            # Run stress test for each scenario
            for scenario in configuration.scenarios:
                logger.info(f"Processing scenario: {scenario.name}")
                
                # Test each portfolio type
                for portfolio_type, data in portfolio_data.items():
                    if portfolio_type in self.engines:
                        engine = self.engines[portfolio_type]
                        result = engine.apply_scenario(data, scenario)
                        all_results.append(result)
            
            # Calculate aggregate results
            aggregate_results = self._calculate_aggregate_results(all_results)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_comprehensive_risk_metrics(all_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(all_results, configuration)
            
            # Determine pass/fail status for regulatory tests
            pass_fail_status = None
            if configuration.test_type == StressTestType.REGULATORY:
                pass_fail_status = self._evaluate_regulatory_compliance(all_results)
            
            return StressTestReport(
                test_date=datetime.now(),
                test_type=configuration.test_type,
                scenarios_tested=[s.name for s in configuration.scenarios],
                portfolio_results=all_results,
                aggregate_results=aggregate_results,
                risk_metrics=risk_metrics,
                recommendations=recommendations,
                pass_fail_status=pass_fail_status
            )
            
        except Exception as e:
            logger.error(f"Error running stress test: {str(e)}")
            raise
    
    def create_custom_scenario(self,
                             name: str,
                             description: str,
                             risk_factor_shocks: Dict[str, List[float]],
                             severity: str = "moderate") -> StressScenarioDefinition:
        """
        Create custom stress scenario
        
        Args:
            name: Scenario name
            description: Scenario description
            risk_factor_shocks: Dictionary of risk factor shocks
            severity: Severity level
            
        Returns:
            Custom stress scenario definition
        """
        # Convert string keys to RiskFactor enums
        risk_factors = {}
        for factor_name, shocks in risk_factor_shocks.items():
            try:
                risk_factor = RiskFactor(factor_name.lower())
                risk_factors[risk_factor] = shocks
            except ValueError:
                logger.warning(f"Unknown risk factor: {factor_name}")
        
        return StressScenarioDefinition(
            name=name,
            description=description,
            risk_factors=risk_factors,
            time_horizon=len(list(risk_factor_shocks.values())[0]) if risk_factor_shocks else 4,
            severity=severity
        )
    
    def run_reverse_stress_test(self,
                              portfolio_data: Dict[str, pd.DataFrame],
                              target_loss: float,
                              risk_factors: List[RiskFactor]) -> Dict:
        """
        Run reverse stress test to find scenarios that produce target loss
        
        Args:
            portfolio_data: Portfolio data
            target_loss: Target loss amount
            risk_factors: Risk factors to vary
            
        Returns:
            Reverse stress test results
        """
        try:
            logger.info(f"Running reverse stress test for target loss: ${target_loss:,.0f}")
            
            # Use optimization to find scenario parameters
            from scipy.optimize import minimize
            
            def objective_function(params):
                # Create scenario from parameters
                scenario_factors = {}
                for i, factor in enumerate(risk_factors):
                    scenario_factors[factor] = [params[i]]
                
                scenario = StressScenarioDefinition(
                    name="Reverse Test Scenario",
                    description="Generated for reverse stress test",
                    risk_factors=scenario_factors,
                    time_horizon=1,
                    severity="custom"
                )
                
                # Calculate total loss
                total_loss = 0
                for portfolio_type, data in portfolio_data.items():
                    if portfolio_type in self.engines:
                        engine = self.engines[portfolio_type]
                        result = engine.apply_scenario(data, scenario)
                        total_loss += result.absolute_loss
                
                # Return squared difference from target
                return (total_loss - target_loss) ** 2
            
            # Initial guess (moderate stress)
            initial_params = [-10.0] * len(risk_factors)  # -10% shock for each factor
            
            # Optimize
            result = minimize(objective_function, initial_params, method='BFGS')
            
            if result.success:
                # Create final scenario
                final_factors = {}
                for i, factor in enumerate(risk_factors):
                    final_factors[factor] = [result.x[i]]
                
                return {
                    'success': True,
                    'scenario_parameters': {factor.value: result.x[i] for i, factor in enumerate(risk_factors)},
                    'achieved_loss': target_loss,
                    'optimization_error': result.fun
                }
            else:
                return {
                    'success': False,
                    'error': 'Optimization failed to converge'
                }
                
        except Exception as e:
            logger.error(f"Error in reverse stress test: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_standard_scenarios(self) -> Dict[str, StressScenarioDefinition]:
        """Create standard stress scenarios"""
        scenarios = {}
        
        # Mild recession scenario
        scenarios['mild_recession'] = StressScenarioDefinition(
            name="Mild Recession",
            description="Moderate economic downturn",
            risk_factors={
                RiskFactor.GDP_GROWTH: [-1.5, -0.5, 1.0],
                RiskFactor.UNEMPLOYMENT: [6.5, 7.0, 6.0],
                RiskFactor.EQUITY_PRICE: [-15.0, -5.0, 10.0],
                RiskFactor.HOUSE_PRICE: [-5.0, -2.0, 2.0]
            },
            time_horizon=3,
            severity="mild"
        )
        
        # Severe recession scenario
        scenarios['severe_recession'] = StressScenarioDefinition(
            name="Severe Recession",
            description="Severe economic downturn similar to 2008",
            risk_factors={
                RiskFactor.GDP_GROWTH: [-4.0, -2.0, 1.5],
                RiskFactor.UNEMPLOYMENT: [10.0, 9.0, 7.0],
                RiskFactor.EQUITY_PRICE: [-40.0, -20.0, 15.0],
                RiskFactor.HOUSE_PRICE: [-20.0, -10.0, 5.0],
                RiskFactor.CREDIT_SPREAD: [300, 200, 100]  # basis points
            },
            time_horizon=3,
            severity="severe"
        )
        
        # Interest rate shock scenario
        scenarios['rate_shock'] = StressScenarioDefinition(
            name="Interest Rate Shock",
            description="Sudden increase in interest rates",
            risk_factors={
                RiskFactor.INTEREST_RATE: [300, 200, 100],  # basis points
                RiskFactor.CREDIT_SPREAD: [150, 100, 50]
            },
            time_horizon=3,
            severity="moderate"
        )
        
        return scenarios
    
    def _calculate_aggregate_results(self, results: List[PortfolioStressResult]) -> Dict[str, float]:
        """Calculate aggregate results across all portfolios"""
        if not results:
            return {}
        
        total_base_value = sum(r.base_value for r in results)
        total_stressed_value = sum(r.stressed_value for r in results)
        total_loss = sum(r.absolute_loss for r in results)
        
        return {
            'total_base_value': total_base_value,
            'total_stressed_value': total_stressed_value,
            'total_absolute_loss': total_loss,
            'total_relative_loss': total_loss / total_base_value if total_base_value > 0 else 0,
            'number_of_portfolios': len(results)
        }
    
    def _calculate_comprehensive_risk_metrics(self, results: List[PortfolioStressResult]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        if not results:
            return {}
        
        losses = [r.absolute_loss for r in results]
        relative_losses = [r.relative_loss for r in results]
        
        return {
            'maximum_loss': max(losses),
            'average_loss': np.mean(losses),
            'loss_standard_deviation': np.std(losses),
            'maximum_relative_loss': max(relative_losses),
            'loss_concentration': max(losses) / sum(losses) if sum(losses) > 0 else 0,
            'number_of_scenarios': len(set(r.scenario_name for r in results))
        }
    
    def _generate_recommendations(self, 
                                results: List[PortfolioStressResult],
                                configuration: StressTestConfiguration) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        # Analyze results
        max_loss = max(r.absolute_loss for r in results) if results else 0
        max_relative_loss = max(r.relative_loss for r in results) if results else 0
        
        # Loss-based recommendations
        if max_relative_loss > 0.20:  # 20% loss
            recommendations.append("Consider reducing portfolio concentration in high-risk assets")
        
        if max_relative_loss > 0.15:  # 15% loss
            recommendations.append("Review and strengthen risk management frameworks")
        
        # Portfolio-specific recommendations
        portfolio_losses = {}
        for result in results:
            portfolio_losses[result.portfolio_name] = result.relative_loss
        
        worst_portfolio = max(portfolio_losses.items(), key=lambda x: x[1]) if portfolio_losses else None
        if worst_portfolio and worst_portfolio[1] > 0.10:
            recommendations.append(f"Focus risk mitigation efforts on {worst_portfolio[0]}")
        
        # Scenario-specific recommendations
        scenario_losses = {}
        for result in results:
            if result.scenario_name not in scenario_losses:
                scenario_losses[result.scenario_name] = []
            scenario_losses[result.scenario_name].append(result.absolute_loss)
        
        for scenario, losses in scenario_losses.items():
            avg_loss = np.mean(losses)
            if avg_loss > max_loss * 0.8:  # Within 80% of maximum loss
                recommendations.append(f"Develop contingency plans for {scenario} scenario")
        
        if not recommendations:
            recommendations.append("Stress test results are within acceptable risk tolerance")
        
        return recommendations
    
    def _evaluate_regulatory_compliance(self, results: List[PortfolioStressResult]) -> bool:
        """Evaluate regulatory compliance for stress tests"""
        # Simplified compliance check
        max_relative_loss = max(r.relative_loss for r in results) if results else 0
        
        # Assume regulatory threshold of 25% maximum loss
        return max_relative_loss <= 0.25