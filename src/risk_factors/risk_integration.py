"""
Risk Integration Module

Comprehensive risk factor integration for PPNR modeling:
- Multi-risk factor correlation modeling
- Portfolio-level risk aggregation
- Risk factor scenario generation
- Integrated stress testing
- Cross-risk dependencies
- Economic capital allocation
- Risk-adjusted performance metrics
- Regulatory capital integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import json
from scipy import stats
from scipy.linalg import cholesky
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import seaborn as sns

class RiskType(Enum):
    """Risk type categories."""
    CREDIT = "Credit Risk"
    MARKET = "Market Risk"
    OPERATIONAL = "Operational Risk"
    LIQUIDITY = "Liquidity Risk"
    MODEL = "Model Risk"
    CONCENTRATION = "Concentration Risk"

class AggregationMethod(Enum):
    """Risk aggregation methods."""
    SIMPLE_SUM = "Simple Sum"
    CORRELATION_MATRIX = "Correlation Matrix"
    COPULA = "Copula"
    MONTE_CARLO = "Monte Carlo"
    FACTOR_MODEL = "Factor Model"

class StressType(Enum):
    """Stress test types."""
    HISTORICAL = "Historical Scenario"
    HYPOTHETICAL = "Hypothetical Scenario"
    REGULATORY = "Regulatory Scenario"
    REVERSE = "Reverse Stress Test"

@dataclass
class RiskFactor:
    """Individual risk factor definition."""
    factor_id: str
    factor_name: str
    risk_type: RiskType
    current_value: float
    volatility: float
    distribution: str = "normal"
    parameters: Dict[str, float] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class RiskScenario:
    """Risk scenario definition."""
    scenario_id: str
    scenario_name: str
    scenario_type: StressType
    probability: float
    factor_shocks: Dict[str, float]  # factor_id -> shock value
    description: str = ""
    regulatory_scenario: bool = False

@dataclass
class PortfolioRisk:
    """Portfolio risk metrics."""
    portfolio_id: str
    expected_loss: float
    unexpected_loss: float
    var_95: float
    var_99: float
    var_999: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    risk_contributions: Dict[str, float]
    diversification_benefit: float

class RiskIntegrationModel:
    """
    Comprehensive risk integration and aggregation system.
    
    Features:
    - Multi-risk factor correlation modeling
    - Portfolio-level risk aggregation
    - Integrated stress testing across risk types
    - Economic capital allocation
    - Risk-adjusted performance measurement
    - Regulatory capital integration
    - Cross-risk dependencies modeling
    - Dynamic correlation estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk integration model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.integration_config = config.get('risk_integration', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.RiskIntegration")
        
        # Risk factors and data
        self.risk_factors: Dict[str, RiskFactor] = {}
        self.risk_factor_data: pd.DataFrame = pd.DataFrame()
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        
        # Models and components
        self.credit_risk_model = None
        self.market_risk_model = None
        self.operational_risk_model = None
        
        # Scenarios and stress tests
        self.risk_scenarios: List[RiskScenario] = []
        self.stress_test_results: Dict[str, Any] = {}
        
        # Aggregation settings
        self.aggregation_method = AggregationMethod(
            self.integration_config.get('aggregation_method', 'CORRELATION_MATRIX')
        )
        self.confidence_levels = self.integration_config.get('confidence_levels', [0.95, 0.99, 0.999])
        self.time_horizon = self.integration_config.get('time_horizon', 1)  # years
        
        # Results storage
        self.portfolio_risks: Dict[str, PortfolioRisk] = {}
        self.economic_capital: Dict[str, float] = {}
        self.risk_adjusted_metrics: Dict[str, Any] = {}
        
        self.logger.info("Risk integration model initialized")
    
    def register_risk_models(self, credit_model=None, market_model=None, operational_model=None):
        """
        Register individual risk models for integration.
        
        Args:
            credit_model: Credit risk model instance
            market_model: Market risk model instance
            operational_model: Operational risk model instance
        """
        self.logger.info("Registering risk models for integration")
        
        if credit_model:
            self.credit_risk_model = credit_model
            self.logger.info("Credit risk model registered")
        
        if market_model:
            self.market_risk_model = market_model
            self.logger.info("Market risk model registered")
        
        if operational_model:
            self.operational_risk_model = operational_model
            self.logger.info("Operational risk model registered")
    
    def define_risk_factors(self, risk_factors_data: pd.DataFrame) -> None:
        """
        Define risk factors for integration.
        
        Args:
            risk_factors_data: DataFrame with risk factor definitions
        """
        self.logger.info(f"Defining {len(risk_factors_data)} risk factors")
        
        self.risk_factors = {}
        
        for _, row in risk_factors_data.iterrows():
            factor = RiskFactor(
                factor_id=str(row.get('factor_id', '')),
                factor_name=str(row.get('factor_name', '')),
                risk_type=RiskType(row.get('risk_type', 'CREDIT')),
                current_value=float(row.get('current_value', 0.0)),
                volatility=float(row.get('volatility', 0.0)),
                distribution=str(row.get('distribution', 'normal')),
                parameters=json.loads(row.get('parameters', '{}')) if isinstance(row.get('parameters'), str) else row.get('parameters', {})
            )
            
            self.risk_factors[factor.factor_id] = factor
        
        self.logger.info(f"Defined {len(self.risk_factors)} risk factors")
    
    def load_risk_factor_data(self, factor_data: pd.DataFrame) -> None:
        """
        Load historical risk factor data.
        
        Args:
            factor_data: DataFrame with historical risk factor values
        """
        self.logger.info(f"Loading risk factor data with shape {factor_data.shape}")
        
        # Ensure date index
        if 'date' in factor_data.columns:
            factor_data = factor_data.set_index('date')
        
        # Convert to datetime index if needed
        if not isinstance(factor_data.index, pd.DatetimeIndex):
            factor_data.index = pd.to_datetime(factor_data.index)
        
        self.risk_factor_data = factor_data.copy()
        
        # Calculate returns/changes
        self.risk_factor_returns = self.risk_factor_data.pct_change().dropna()
        
        self.logger.info(f"Loaded risk factor data: {len(self.risk_factor_data)} observations, {len(self.risk_factor_data.columns)} factors")
    
    def estimate_correlation_matrix(self, method: str = 'ledoit_wolf', 
                                  lookback_window: Optional[int] = None) -> pd.DataFrame:
        """
        Estimate correlation matrix between risk factors.
        
        Args:
            method: Estimation method ('sample', 'ledoit_wolf', 'exponential_weighted')
            lookback_window: Number of observations to use (None for all)
            
        Returns:
            Correlation matrix
        """
        self.logger.info(f"Estimating correlation matrix using {method} method")
        
        if self.risk_factor_returns.empty:
            self.logger.warning("No risk factor data available for correlation estimation")
            return pd.DataFrame()
        
        # Select data window
        data = self.risk_factor_returns.copy()
        if lookback_window:
            data = data.tail(lookback_window)
        
        # Remove factors with insufficient data
        data = data.dropna(axis=1, thresh=len(data) * 0.5)  # At least 50% non-null
        
        if method == 'sample':
            correlation_matrix = data.corr()
        
        elif method == 'ledoit_wolf':
            # Shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = lw.fit(data.fillna(0)).covariance_
            
            # Convert to correlation
            std_devs = np.sqrt(np.diag(cov_matrix))
            correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
            correlation_matrix = pd.DataFrame(correlation_matrix, 
                                            index=data.columns, 
                                            columns=data.columns)
        
        elif method == 'exponential_weighted':
            # Exponentially weighted correlation
            span = self.integration_config.get('ewm_span', 60)
            correlation_matrix = data.ewm(span=span).corr().iloc[-len(data.columns):]
        
        else:
            self.logger.error(f"Unsupported correlation estimation method: {method}")
            correlation_matrix = pd.DataFrame()
        
        # Ensure positive semi-definite
        correlation_matrix = self._ensure_positive_semidefinite(correlation_matrix)
        
        self.correlation_matrix = correlation_matrix
        self.logger.info(f"Correlation matrix estimated: {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def _ensure_positive_semidefinite(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Ensure correlation matrix is positive semi-definite."""
        try:
            # Eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(matrix.values)
            
            # Set negative eigenvalues to small positive value
            eigenvals = np.maximum(eigenvals, 1e-8)
            
            # Reconstruct matrix
            reconstructed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Normalize to correlation matrix
            diag_sqrt = np.sqrt(np.diag(reconstructed))
            correlation_matrix = reconstructed / np.outer(diag_sqrt, diag_sqrt)
            
            return pd.DataFrame(correlation_matrix, index=matrix.index, columns=matrix.columns)
        
        except:
            self.logger.warning("Failed to ensure positive semi-definite matrix, using original")
            return matrix
    
    def define_risk_scenarios(self, scenarios_data: pd.DataFrame) -> None:
        """
        Define risk scenarios for stress testing.
        
        Args:
            scenarios_data: DataFrame with scenario definitions
        """
        self.logger.info(f"Defining {len(scenarios_data)} risk scenarios")
        
        self.risk_scenarios = []
        
        for _, row in scenarios_data.iterrows():
            # Parse factor shocks
            factor_shocks = {}
            if 'factor_shocks' in row and row['factor_shocks']:
                if isinstance(row['factor_shocks'], str):
                    factor_shocks = json.loads(row['factor_shocks'])
                else:
                    factor_shocks = row['factor_shocks']
            
            scenario = RiskScenario(
                scenario_id=str(row.get('scenario_id', '')),
                scenario_name=str(row.get('scenario_name', '')),
                scenario_type=StressType(row.get('scenario_type', 'HYPOTHETICAL')),
                probability=float(row.get('probability', 0.0)),
                factor_shocks=factor_shocks,
                description=str(row.get('description', '')),
                regulatory_scenario=bool(row.get('regulatory_scenario', False))
            )
            
            self.risk_scenarios.append(scenario)
        
        self.logger.info(f"Defined {len(self.risk_scenarios)} risk scenarios")
    
    def calculate_portfolio_risk(self, portfolio_id: str, 
                               exposures: Dict[str, float],
                               method: str = None) -> PortfolioRisk:
        """
        Calculate integrated portfolio risk metrics.
        
        Args:
            portfolio_id: Portfolio identifier
            exposures: Risk factor exposures {factor_id: exposure}
            method: Aggregation method override
            
        Returns:
            Portfolio risk metrics
        """
        self.logger.info(f"Calculating portfolio risk for {portfolio_id}")
        
        if method is None:
            method = self.aggregation_method.value
        
        # Get risk factor volatilities
        factor_vols = {}
        for factor_id, exposure in exposures.items():
            if factor_id in self.risk_factors:
                factor_vols[factor_id] = self.risk_factors[factor_id].volatility
            else:
                # Estimate from data if available
                if factor_id in self.risk_factor_returns.columns:
                    factor_vols[factor_id] = self.risk_factor_returns[factor_id].std() * np.sqrt(252)  # Annualized
                else:
                    factor_vols[factor_id] = 0.0
        
        # Calculate portfolio risk based on method
        if method == 'SIMPLE_SUM':
            portfolio_risk = self._calculate_simple_sum_risk(exposures, factor_vols)
        elif method == 'CORRELATION_MATRIX':
            portfolio_risk = self._calculate_correlation_matrix_risk(exposures, factor_vols)
        elif method == 'MONTE_CARLO':
            portfolio_risk = self._calculate_monte_carlo_risk(exposures, factor_vols)
        else:
            self.logger.warning(f"Unsupported aggregation method: {method}, using correlation matrix")
            portfolio_risk = self._calculate_correlation_matrix_risk(exposures, factor_vols)
        
        # Store results
        self.portfolio_risks[portfolio_id] = portfolio_risk
        
        self.logger.info(f"Portfolio risk calculated for {portfolio_id}: VaR99 = ${portfolio_risk.var_99:,.2f}")
        return portfolio_risk
    
    def _calculate_simple_sum_risk(self, exposures: Dict[str, float], 
                                 factor_vols: Dict[str, float]) -> PortfolioRisk:
        """Calculate portfolio risk using simple sum (no diversification)."""
        total_risk = 0.0
        risk_contributions = {}
        
        for factor_id, exposure in exposures.items():
            factor_risk = abs(exposure) * factor_vols.get(factor_id, 0.0)
            total_risk += factor_risk
            risk_contributions[factor_id] = factor_risk
        
        # Assume normal distribution for VaR calculations
        var_95 = total_risk * stats.norm.ppf(0.95)
        var_99 = total_risk * stats.norm.ppf(0.99)
        var_999 = total_risk * stats.norm.ppf(0.999)
        
        es_95 = total_risk * stats.norm.expect(lambda x: x, lb=stats.norm.ppf(0.95))
        es_99 = total_risk * stats.norm.expect(lambda x: x, lb=stats.norm.ppf(0.99))
        
        return PortfolioRisk(
            portfolio_id="simple_sum",
            expected_loss=0.0,
            unexpected_loss=total_risk,
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            risk_contributions=risk_contributions,
            diversification_benefit=0.0
        )
    
    def _calculate_correlation_matrix_risk(self, exposures: Dict[str, float], 
                                         factor_vols: Dict[str, float]) -> PortfolioRisk:
        """Calculate portfolio risk using correlation matrix approach."""
        # Filter factors present in correlation matrix
        common_factors = set(exposures.keys()) & set(self.correlation_matrix.columns)
        
        if not common_factors:
            self.logger.warning("No common factors between exposures and correlation matrix")
            return self._calculate_simple_sum_risk(exposures, factor_vols)
        
        # Create exposure and volatility vectors
        factor_list = list(common_factors)
        exposure_vector = np.array([exposures[f] for f in factor_list])
        vol_vector = np.array([factor_vols.get(f, 0.0) for f in factor_list])
        
        # Get correlation submatrix
        corr_matrix = self.correlation_matrix.loc[factor_list, factor_list].values
        
        # Calculate covariance matrix
        cov_matrix = np.outer(vol_vector, vol_vector) * corr_matrix
        
        # Portfolio variance
        portfolio_variance = exposure_vector.T @ cov_matrix @ exposure_vector
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Risk contributions (marginal contributions)
        marginal_contributions = cov_matrix @ exposure_vector
        risk_contributions = {}
        
        for i, factor_id in enumerate(factor_list):
            contribution = exposures[factor_id] * marginal_contributions[i] / portfolio_variance if portfolio_variance > 0 else 0
            risk_contributions[factor_id] = contribution * portfolio_vol
        
        # VaR calculations (assuming normal distribution)
        var_95 = portfolio_vol * stats.norm.ppf(0.95)
        var_99 = portfolio_vol * stats.norm.ppf(0.99)
        var_999 = portfolio_vol * stats.norm.ppf(0.999)
        
        # Expected shortfall
        es_95 = portfolio_vol * stats.norm.expect(lambda x: x, lb=stats.norm.ppf(0.95))
        es_99 = portfolio_vol * stats.norm.expect(lambda x: x, lb=stats.norm.ppf(0.99))
        
        # Diversification benefit
        undiversified_risk = sum(abs(exposures[f]) * factor_vols.get(f, 0.0) for f in factor_list)
        diversification_benefit = max(0, undiversified_risk - portfolio_vol)
        
        return PortfolioRisk(
            portfolio_id="correlation_matrix",
            expected_loss=0.0,
            unexpected_loss=portfolio_vol,
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            risk_contributions=risk_contributions,
            diversification_benefit=diversification_benefit
        )
    
    def _calculate_monte_carlo_risk(self, exposures: Dict[str, float], 
                                  factor_vols: Dict[str, float],
                                  n_simulations: int = 100000) -> PortfolioRisk:
        """Calculate portfolio risk using Monte Carlo simulation."""
        # Filter factors present in correlation matrix
        common_factors = set(exposures.keys()) & set(self.correlation_matrix.columns)
        
        if not common_factors:
            self.logger.warning("No common factors for Monte Carlo simulation")
            return self._calculate_simple_sum_risk(exposures, factor_vols)
        
        factor_list = list(common_factors)
        exposure_vector = np.array([exposures[f] for f in factor_list])
        vol_vector = np.array([factor_vols.get(f, 0.0) for f in factor_list])
        
        # Get correlation matrix
        corr_matrix = self.correlation_matrix.loc[factor_list, factor_list].values
        
        # Generate correlated random shocks
        np.random.seed(42)  # For reproducibility
        
        try:
            # Cholesky decomposition for correlation
            L = cholesky(corr_matrix, lower=True)
            
            # Generate independent normal random variables
            independent_shocks = np.random.normal(0, 1, (n_simulations, len(factor_list)))
            
            # Apply correlation structure
            correlated_shocks = independent_shocks @ L.T
            
            # Scale by volatilities
            factor_shocks = correlated_shocks * vol_vector
            
            # Calculate portfolio P&L for each simulation
            portfolio_pnl = factor_shocks @ exposure_vector
            
        except np.linalg.LinAlgError:
            self.logger.warning("Cholesky decomposition failed, using independent shocks")
            # Fallback to independent shocks
            factor_shocks = np.random.normal(0, vol_vector, (n_simulations, len(factor_list)))
            portfolio_pnl = factor_shocks @ exposure_vector
        
        # Calculate risk metrics
        expected_loss = np.mean(portfolio_pnl)
        portfolio_vol = np.std(portfolio_pnl)
        
        var_95 = np.percentile(portfolio_pnl, 5)  # 5th percentile for losses
        var_99 = np.percentile(portfolio_pnl, 1)
        var_999 = np.percentile(portfolio_pnl, 0.1)
        
        # Expected shortfall
        es_95 = np.mean(portfolio_pnl[portfolio_pnl <= var_95])
        es_99 = np.mean(portfolio_pnl[portfolio_pnl <= var_99])
        
        # Risk contributions (approximate using correlation approach)
        risk_contributions = {}
        for i, factor_id in enumerate(factor_list):
            # Correlation between factor and portfolio
            factor_portfolio_corr = np.corrcoef(factor_shocks[:, i], portfolio_pnl)[0, 1]
            contribution = factor_portfolio_corr * vol_vector[i] * abs(exposures[factor_id]) / portfolio_vol if portfolio_vol > 0 else 0
            risk_contributions[factor_id] = contribution * portfolio_vol
        
        # Diversification benefit
        undiversified_risk = sum(abs(exposures[f]) * factor_vols.get(f, 0.0) for f in factor_list)
        diversification_benefit = max(0, undiversified_risk - portfolio_vol)
        
        return PortfolioRisk(
            portfolio_id="monte_carlo",
            expected_loss=expected_loss,
            unexpected_loss=portfolio_vol,
            var_95=abs(var_95),  # Convert to positive loss amount
            var_99=abs(var_99),
            var_999=abs(var_999),
            expected_shortfall_95=abs(es_95),
            expected_shortfall_99=abs(es_99),
            risk_contributions=risk_contributions,
            diversification_benefit=diversification_benefit
        )
    
    def run_integrated_stress_test(self, scenario_id: str = None) -> Dict[str, Any]:
        """
        Run integrated stress test across all risk types.
        
        Args:
            scenario_id: Specific scenario to run (None for all scenarios)
            
        Returns:
            Integrated stress test results
        """
        self.logger.info(f"Running integrated stress test")
        
        stress_results = {
            'baseline_metrics': {},
            'scenario_results': {},
            'worst_case_analysis': {},
            'risk_type_breakdown': {}
        }
        
        # Calculate baseline metrics
        baseline_metrics = self._calculate_baseline_metrics()
        stress_results['baseline_metrics'] = baseline_metrics
        
        # Run scenarios
        scenarios_to_run = [s for s in self.risk_scenarios if scenario_id is None or s.scenario_id == scenario_id]
        
        for scenario in scenarios_to_run:
            scenario_result = self._run_scenario_stress_test(scenario)
            stress_results['scenario_results'][scenario.scenario_id] = scenario_result
        
        # Analyze worst case
        if stress_results['scenario_results']:
            worst_case = self._identify_worst_case_scenario(stress_results['scenario_results'])
            stress_results['worst_case_analysis'] = worst_case
        
        # Risk type breakdown
        stress_results['risk_type_breakdown'] = self._analyze_risk_type_contributions(stress_results)
        
        self.stress_test_results = stress_results
        self.logger.info("Integrated stress test completed")
        
        return stress_results
    
    def _calculate_baseline_metrics(self) -> Dict[str, Any]:
        """Calculate baseline risk metrics across all risk types."""
        baseline = {
            'credit_risk': {},
            'market_risk': {},
            'operational_risk': {},
            'total_risk': {}
        }
        
        # Credit risk baseline
        if self.credit_risk_model:
            try:
                credit_var = self.credit_risk_model.calculate_portfolio_var()
                baseline['credit_risk'] = {
                    'var_99': credit_var.get('var_99', 0.0),
                    'expected_loss': credit_var.get('expected_loss', 0.0),
                    'unexpected_loss': credit_var.get('unexpected_loss', 0.0)
                }
            except:
                baseline['credit_risk'] = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        # Market risk baseline
        if self.market_risk_model:
            try:
                market_var = self.market_risk_model.calculate_portfolio_var()
                baseline['market_risk'] = {
                    'var_99': market_var.get('var_99', 0.0),
                    'expected_loss': 0.0,  # Market risk typically has zero expected loss
                    'unexpected_loss': market_var.get('var_99', 0.0)
                }
            except:
                baseline['market_risk'] = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        # Operational risk baseline
        if self.operational_risk_model:
            try:
                op_var = self.operational_risk_model.calculate_operational_var()
                baseline['operational_risk'] = {
                    'var_99': op_var.get('var_amount', 0.0),
                    'expected_loss': op_var.get('expected_loss', 0.0),
                    'unexpected_loss': op_var.get('unexpected_loss', 0.0)
                }
            except:
                baseline['operational_risk'] = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        # Total risk (simple aggregation for baseline)
        total_var_99 = (baseline['credit_risk'].get('var_99', 0.0) + 
                       baseline['market_risk'].get('var_99', 0.0) + 
                       baseline['operational_risk'].get('var_99', 0.0))
        
        total_expected_loss = (baseline['credit_risk'].get('expected_loss', 0.0) + 
                             baseline['market_risk'].get('expected_loss', 0.0) + 
                             baseline['operational_risk'].get('expected_loss', 0.0))
        
        baseline['total_risk'] = {
            'var_99': total_var_99,
            'expected_loss': total_expected_loss,
            'unexpected_loss': total_var_99 - total_expected_loss
        }
        
        return baseline
    
    def _run_scenario_stress_test(self, scenario: RiskScenario) -> Dict[str, Any]:
        """Run stress test for a specific scenario."""
        scenario_result = {
            'scenario_info': {
                'scenario_id': scenario.scenario_id,
                'scenario_name': scenario.scenario_name,
                'scenario_type': scenario.scenario_type.value,
                'probability': scenario.probability
            },
            'stressed_metrics': {},
            'impact_analysis': {},
            'risk_type_impacts': {}
        }
        
        # Apply shocks to risk factors
        stressed_factors = self._apply_scenario_shocks(scenario)
        
        # Calculate stressed metrics for each risk type
        stressed_metrics = {
            'credit_risk': {},
            'market_risk': {},
            'operational_risk': {},
            'total_risk': {}
        }
        
        # Credit risk under stress
        if self.credit_risk_model:
            try:
                # This would require implementing scenario application in credit model
                credit_stressed = self._stress_credit_risk(scenario, stressed_factors)
                stressed_metrics['credit_risk'] = credit_stressed
            except:
                stressed_metrics['credit_risk'] = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        # Market risk under stress
        if self.market_risk_model:
            try:
                market_stressed = self._stress_market_risk(scenario, stressed_factors)
                stressed_metrics['market_risk'] = market_stressed
            except:
                stressed_metrics['market_risk'] = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        # Operational risk under stress
        if self.operational_risk_model:
            try:
                op_stressed = self._stress_operational_risk(scenario, stressed_factors)
                stressed_metrics['operational_risk'] = op_stressed
            except:
                stressed_metrics['operational_risk'] = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        # Total stressed risk
        total_stressed_var = (stressed_metrics['credit_risk'].get('var_99', 0.0) + 
                            stressed_metrics['market_risk'].get('var_99', 0.0) + 
                            stressed_metrics['operational_risk'].get('var_99', 0.0))
        
        total_stressed_el = (stressed_metrics['credit_risk'].get('expected_loss', 0.0) + 
                           stressed_metrics['market_risk'].get('expected_loss', 0.0) + 
                           stressed_metrics['operational_risk'].get('expected_loss', 0.0))
        
        stressed_metrics['total_risk'] = {
            'var_99': total_stressed_var,
            'expected_loss': total_stressed_el,
            'unexpected_loss': total_stressed_var - total_stressed_el
        }
        
        scenario_result['stressed_metrics'] = stressed_metrics
        
        # Impact analysis
        baseline = self._calculate_baseline_metrics()
        impact_analysis = {}
        
        for risk_type in ['credit_risk', 'market_risk', 'operational_risk', 'total_risk']:
            baseline_var = baseline.get(risk_type, {}).get('var_99', 0.0)
            stressed_var = stressed_metrics.get(risk_type, {}).get('var_99', 0.0)
            
            impact_analysis[risk_type] = {
                'baseline_var_99': baseline_var,
                'stressed_var_99': stressed_var,
                'absolute_impact': stressed_var - baseline_var,
                'relative_impact': (stressed_var / baseline_var - 1) * 100 if baseline_var > 0 else 0
            }
        
        scenario_result['impact_analysis'] = impact_analysis
        
        return scenario_result
    
    def _apply_scenario_shocks(self, scenario: RiskScenario) -> Dict[str, float]:
        """Apply scenario shocks to risk factors."""
        stressed_factors = {}
        
        for factor_id, shock in scenario.factor_shocks.items():
            if factor_id in self.risk_factors:
                current_value = self.risk_factors[factor_id].current_value
                stressed_value = current_value * (1 + shock)  # Assuming percentage shock
                stressed_factors[factor_id] = stressed_value
            else:
                stressed_factors[factor_id] = shock
        
        return stressed_factors
    
    def _stress_credit_risk(self, scenario: RiskScenario, stressed_factors: Dict[str, float]) -> Dict[str, Any]:
        """Apply stress to credit risk model."""
        # Placeholder implementation
        # In practice, this would involve:
        # 1. Mapping scenario shocks to credit risk factors (PD, LGD, EAD)
        # 2. Recalculating credit VaR under stressed conditions
        # 3. Considering correlation changes under stress
        
        baseline_credit = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        if self.credit_risk_model:
            try:
                baseline_credit = self.credit_risk_model.calculate_portfolio_var()
            except:
                pass
        
        # Apply stress multipliers based on scenario type
        stress_multiplier = 1.0
        if scenario.scenario_type == StressType.REGULATORY:
            stress_multiplier = 2.0  # Regulatory scenarios typically more severe
        elif scenario.scenario_type == StressType.HISTORICAL:
            stress_multiplier = 1.5
        else:
            stress_multiplier = 1.2
        
        return {
            'var_99': baseline_credit.get('var_99', 0.0) * stress_multiplier,
            'expected_loss': baseline_credit.get('expected_loss', 0.0) * stress_multiplier,
            'unexpected_loss': baseline_credit.get('unexpected_loss', 0.0) * stress_multiplier
        }
    
    def _stress_market_risk(self, scenario: RiskScenario, stressed_factors: Dict[str, float]) -> Dict[str, Any]:
        """Apply stress to market risk model."""
        # Placeholder implementation
        baseline_market = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        if self.market_risk_model:
            try:
                baseline_market = self.market_risk_model.calculate_portfolio_var()
            except:
                pass
        
        # Market risk is more directly affected by factor shocks
        shock_magnitude = np.mean([abs(shock) for shock in scenario.factor_shocks.values()]) if scenario.factor_shocks else 0.1
        stress_multiplier = 1.0 + shock_magnitude * 2  # Amplify based on shock size
        
        return {
            'var_99': baseline_market.get('var_99', 0.0) * stress_multiplier,
            'expected_loss': 0.0,  # Market risk typically has zero expected loss
            'unexpected_loss': baseline_market.get('var_99', 0.0) * stress_multiplier
        }
    
    def _stress_operational_risk(self, scenario: RiskScenario, stressed_factors: Dict[str, float]) -> Dict[str, Any]:
        """Apply stress to operational risk model."""
        # Placeholder implementation
        baseline_op = {'var_99': 0.0, 'expected_loss': 0.0, 'unexpected_loss': 0.0}
        
        if self.operational_risk_model:
            try:
                baseline_op = self.operational_risk_model.calculate_operational_var()
                baseline_op = {
                    'var_99': baseline_op.get('var_amount', 0.0),
                    'expected_loss': baseline_op.get('expected_loss', 0.0),
                    'unexpected_loss': baseline_op.get('unexpected_loss', 0.0)
                }
            except:
                pass
        
        # Operational risk may increase during stress due to operational failures
        stress_multiplier = 1.3  # Moderate increase under stress
        
        return {
            'var_99': baseline_op.get('var_99', 0.0) * stress_multiplier,
            'expected_loss': baseline_op.get('expected_loss', 0.0) * stress_multiplier,
            'unexpected_loss': baseline_op.get('unexpected_loss', 0.0) * stress_multiplier
        }
    
    def _identify_worst_case_scenario(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the worst case scenario from stress test results."""
        worst_case = {
            'scenario_id': '',
            'scenario_name': '',
            'total_var_99': 0.0,
            'total_impact': 0.0,
            'risk_breakdown': {}
        }
        
        max_var = 0.0
        
        for scenario_id, result in scenario_results.items():
            total_var = result.get('stressed_metrics', {}).get('total_risk', {}).get('var_99', 0.0)
            
            if total_var > max_var:
                max_var = total_var
                worst_case['scenario_id'] = scenario_id
                worst_case['scenario_name'] = result.get('scenario_info', {}).get('scenario_name', '')
                worst_case['total_var_99'] = total_var
                
                # Calculate total impact
                baseline_total = self._calculate_baseline_metrics().get('total_risk', {}).get('var_99', 0.0)
                worst_case['total_impact'] = total_var - baseline_total
                
                # Risk breakdown
                worst_case['risk_breakdown'] = result.get('impact_analysis', {})
        
        return worst_case
    
    def _analyze_risk_type_contributions(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk type contributions across scenarios."""
        risk_contributions = {
            'average_contributions': {},
            'max_contributions': {},
            'correlation_analysis': {}
        }
        
        # Collect data across scenarios
        scenario_data = {}
        risk_types = ['credit_risk', 'market_risk', 'operational_risk']
        
        for risk_type in risk_types:
            scenario_data[risk_type] = []
        
        for scenario_id, result in stress_results.get('scenario_results', {}).items():
            for risk_type in risk_types:
                var_99 = result.get('stressed_metrics', {}).get(risk_type, {}).get('var_99', 0.0)
                scenario_data[risk_type].append(var_99)
        
        # Calculate statistics
        for risk_type in risk_types:
            if scenario_data[risk_type]:
                risk_contributions['average_contributions'][risk_type] = np.mean(scenario_data[risk_type])
                risk_contributions['max_contributions'][risk_type] = np.max(scenario_data[risk_type])
        
        # Correlation analysis between risk types
        if all(len(scenario_data[rt]) > 1 for rt in risk_types):
            corr_matrix = np.corrcoef([scenario_data[rt] for rt in risk_types])
            risk_contributions['correlation_analysis'] = {
                f'{risk_types[i]}_{risk_types[j]}': corr_matrix[i, j]
                for i in range(len(risk_types))
                for j in range(i+1, len(risk_types))
            }
        
        return risk_contributions
    
    def calculate_economic_capital(self, confidence_level: float = 0.999) -> Dict[str, Any]:
        """
        Calculate economic capital allocation across risk types.
        
        Args:
            confidence_level: Confidence level for economic capital
            
        Returns:
            Economic capital allocation
        """
        self.logger.info(f"Calculating economic capital at {confidence_level*100}% confidence level")
        
        economic_capital = {
            'total_economic_capital': 0.0,
            'risk_type_allocation': {},
            'diversification_benefit': 0.0,
            'capital_ratios': {},
            'confidence_level': confidence_level
        }
        
        # Calculate standalone economic capital by risk type
        standalone_capitals = {}
        
        # Credit risk economic capital
        if self.credit_risk_model:
            try:
                credit_var = self.credit_risk_model.calculate_portfolio_var(confidence_level=confidence_level)
                standalone_capitals['credit_risk'] = credit_var.get('var_99', 0.0)
            except:
                standalone_capitals['credit_risk'] = 0.0
        
        # Market risk economic capital
        if self.market_risk_model:
            try:
                market_var = self.market_risk_model.calculate_portfolio_var(confidence_level=confidence_level)
                standalone_capitals['market_risk'] = market_var.get('var_99', 0.0)
            except:
                standalone_capitals['market_risk'] = 0.0
        
        # Operational risk economic capital
        if self.operational_risk_model:
            try:
                op_var = self.operational_risk_model.calculate_operational_var(confidence_level=confidence_level)
                standalone_capitals['operational_risk'] = op_var.get('var_amount', 0.0)
            except:
                standalone_capitals['operational_risk'] = 0.0
        
        # Calculate diversified economic capital
        # This is a simplified approach - in practice would use copulas or other advanced methods
        total_standalone = sum(standalone_capitals.values())
        
        # Assume some diversification benefit (typically 10-30%)
        diversification_factor = self.integration_config.get('diversification_factor', 0.8)
        diversified_capital = total_standalone * diversification_factor
        
        economic_capital['total_economic_capital'] = diversified_capital
        economic_capital['risk_type_allocation'] = standalone_capitals
        economic_capital['diversification_benefit'] = total_standalone - diversified_capital
        
        # Calculate capital ratios
        if diversified_capital > 0:
            for risk_type, capital in standalone_capitals.items():
                economic_capital['capital_ratios'][risk_type] = capital / diversified_capital
        
        self.economic_capital = economic_capital
        self.logger.info(f"Economic capital calculated: ${diversified_capital:,.2f}")
        
        return economic_capital
    
    def calculate_risk_adjusted_performance(self, business_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            business_metrics: Business performance metrics (revenue, costs, etc.)
            
        Returns:
            Risk-adjusted performance metrics
        """
        self.logger.info("Calculating risk-adjusted performance metrics")
        
        performance_metrics = {
            'raroc': 0.0,  # Risk-Adjusted Return on Capital
            'rorac': 0.0,  # Return on Risk-Adjusted Capital
            'economic_value_added': 0.0,
            'risk_efficiency_ratio': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Get economic capital
        if not self.economic_capital:
            self.calculate_economic_capital()
        
        total_economic_capital = self.economic_capital.get('total_economic_capital', 0.0)
        
        # Extract business metrics
        net_income = business_metrics.get('net_income', 0.0)
        revenue = business_metrics.get('revenue', 0.0)
        costs = business_metrics.get('costs', 0.0)
        cost_of_capital = business_metrics.get('cost_of_capital', 0.10)  # 10% default
        
        # Calculate RAROC (Risk-Adjusted Return on Capital)
        if total_economic_capital > 0:
            performance_metrics['raroc'] = net_income / total_economic_capital
        
        # Calculate RORAC (Return on Risk-Adjusted Capital)
        # Same as RAROC in this simplified implementation
        performance_metrics['rorac'] = performance_metrics['raroc']
        
        # Calculate Economic Value Added (EVA)
        capital_charge = total_economic_capital * cost_of_capital
        performance_metrics['economic_value_added'] = net_income - capital_charge
        
        # Risk Efficiency Ratio (Revenue per unit of risk)
        if total_economic_capital > 0:
            performance_metrics['risk_efficiency_ratio'] = revenue / total_economic_capital
        
        # Sharpe Ratio (simplified - would need return volatility)
        risk_free_rate = business_metrics.get('risk_free_rate', 0.02)  # 2% default
        return_volatility = business_metrics.get('return_volatility', 0.15)  # 15% default
        
        if return_volatility > 0:
            excess_return = performance_metrics['raroc'] - risk_free_rate
            performance_metrics['sharpe_ratio'] = excess_return / return_volatility
        
        self.risk_adjusted_metrics = performance_metrics
        self.logger.info(f"Risk-adjusted performance calculated: RAROC = {performance_metrics['raroc']:.2%}")
        
        return performance_metrics
    
    def generate_risk_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk integration report."""
        self.logger.info("Generating risk integration report")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'report_type': 'Risk Integration Analysis',
                'version': '1.0'
            },
            'executive_summary': {},
            'risk_factor_analysis': {},
            'correlation_analysis': {},
            'portfolio_risk_metrics': {},
            'stress_test_results': {},
            'economic_capital': {},
            'risk_adjusted_performance': {},
            'recommendations': []
        }
        
        # Executive summary
        total_factors = len(self.risk_factors)
        total_portfolios = len(self.portfolio_risks)
        
        report['executive_summary'] = {
            'total_risk_factors': total_factors,
            'total_portfolios_analyzed': total_portfolios,
            'correlation_matrix_size': f"{self.correlation_matrix.shape[0]}x{self.correlation_matrix.shape[1]}" if not self.correlation_matrix.empty else "0x0",
            'stress_scenarios_analyzed': len(self.risk_scenarios),
            'aggregation_method': self.aggregation_method.value
        }
        
        # Risk factor analysis
        if self.risk_factors:
            risk_type_counts = {}
            for factor in self.risk_factors.values():
                risk_type = factor.risk_type.value
                risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
            
            report['risk_factor_analysis'] = {
                'risk_type_distribution': risk_type_counts,
                'average_volatility_by_type': self._calculate_average_volatility_by_type()
            }
        
        # Correlation analysis
        if not self.correlation_matrix.empty:
            report['correlation_analysis'] = {
                'matrix_statistics': {
                    'mean_correlation': self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].mean(),
                    'max_correlation': self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].max(),
                    'min_correlation': self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].min()
                },
                'high_correlations': self._identify_high_correlations()
            }
        
        # Portfolio risk metrics
        report['portfolio_risk_metrics'] = {
            portfolio_id: {
                'var_99': portfolio.var_99,
                'expected_shortfall_99': portfolio.expected_shortfall_99,
                'diversification_benefit': portfolio.diversification_benefit
            }
            for portfolio_id, portfolio in self.portfolio_risks.items()
        }
        
        # Stress test results
        if self.stress_test_results:
            report['stress_test_results'] = self.stress_test_results
        
        # Economic capital
        if self.economic_capital:
            report['economic_capital'] = self.economic_capital
        
        # Risk-adjusted performance
        if self.risk_adjusted_metrics:
            report['risk_adjusted_performance'] = self.risk_adjusted_metrics
        
        # Recommendations
        report['recommendations'] = self._generate_integration_recommendations()
        
        self.logger.info("Risk integration report generated")
        return report
    
    def _calculate_average_volatility_by_type(self) -> Dict[str, float]:
        """Calculate average volatility by risk type."""
        volatility_by_type = {}
        
        for risk_type in RiskType:
            factors_of_type = [f for f in self.risk_factors.values() if f.risk_type == risk_type]
            if factors_of_type:
                avg_vol = np.mean([f.volatility for f in factors_of_type])
                volatility_by_type[risk_type.value] = avg_vol
        
        return volatility_by_type
    
    def _identify_high_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify high correlations in the correlation matrix."""
        high_correlations = []
        
        if self.correlation_matrix.empty:
            return high_correlations
        
        # Get upper triangle indices
        upper_triangle = np.triu_indices_from(self.correlation_matrix.values, k=1)
        
        for i, j in zip(upper_triangle[0], upper_triangle[1]):
            correlation = self.correlation_matrix.iloc[i, j]
            
            if abs(correlation) >= threshold:
                high_correlations.append({
                    'factor_1': self.correlation_matrix.index[i],
                    'factor_2': self.correlation_matrix.columns[j],
                    'correlation': correlation
                })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return high_correlations[:10]  # Top 10 high correlations
    
    def _generate_integration_recommendations(self) -> List[Dict[str, Any]]:
        """Generate risk integration recommendations."""
        recommendations = []
        
        # Data quality recommendations
        if len(self.risk_factors) < 10:
            recommendations.append({
                'category': 'Data Coverage',
                'priority': 'HIGH',
                'recommendation': 'Expand risk factor coverage to improve model comprehensiveness',
                'timeline': '3-6 months'
            })
        
        if self.correlation_matrix.empty:
            recommendations.append({
                'category': 'Correlation Modeling',
                'priority': 'HIGH',
                'recommendation': 'Implement correlation matrix estimation for risk aggregation',
                'timeline': '1-3 months'
            })
        
        # Model development recommendations
        if not self.stress_test_results:
            recommendations.append({
                'category': 'Stress Testing',
                'priority': 'MEDIUM',
                'recommendation': 'Implement integrated stress testing across all risk types',
                'timeline': '6-12 months'
            })
        
        if not self.economic_capital:
            recommendations.append({
                'category': 'Capital Management',
                'priority': 'MEDIUM',
                'recommendation': 'Develop economic capital allocation framework',
                'timeline': '6-12 months'
            })
        
        # Performance measurement recommendations
        if not self.risk_adjusted_metrics:
            recommendations.append({
                'category': 'Performance Measurement',
                'priority': 'MEDIUM',
                'recommendation': 'Implement risk-adjusted performance measurement system',
                'timeline': '3-6 months'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Model Validation',
                'priority': 'HIGH',
                'recommendation': 'Establish regular validation procedures for integrated risk models',
                'timeline': '3-6 months'
            },
            {
                'category': 'Technology Infrastructure',
                'priority': 'MEDIUM',
                'recommendation': 'Enhance technology infrastructure for real-time risk integration',
                'timeline': '12-18 months'
            }
        ])
        
        return recommendations