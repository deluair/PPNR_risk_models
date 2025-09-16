"""
Market Risk Modeling Module

Comprehensive market risk factor modeling for PPNR:
- Interest rate risk modeling
- Equity risk modeling
- Foreign exchange risk
- Credit spread risk
- Commodity risk
- Value at Risk (VaR) calculations
- Expected Shortfall (ES) calculations
- Stress testing and scenario analysis
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
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

class RiskFactorType(Enum):
    """Types of market risk factors."""
    INTEREST_RATE = "Interest Rate"
    EQUITY = "Equity"
    FX = "Foreign Exchange"
    CREDIT_SPREAD = "Credit Spread"
    COMMODITY = "Commodity"
    VOLATILITY = "Volatility"

class VaRMethod(Enum):
    """Value at Risk calculation methods."""
    HISTORICAL = "Historical Simulation"
    PARAMETRIC = "Parametric"
    MONTE_CARLO = "Monte Carlo"
    FILTERED_HISTORICAL = "Filtered Historical Simulation"

@dataclass
class MarketRiskFactor:
    """Individual market risk factor."""
    factor_id: str
    factor_name: str
    factor_type: RiskFactorType
    current_value: float
    currency: str = "USD"
    volatility: float = 0.0
    correlation_group: str = ""
    
    def __post_init__(self):
        if not self.correlation_group:
            self.correlation_group = self.factor_type.value

@dataclass
class PortfolioPosition:
    """Portfolio position for market risk calculation."""
    position_id: str
    instrument_type: str
    notional_amount: float
    market_value: float
    currency: str
    risk_factor_sensitivities: Dict[str, float]
    maturity_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.risk_factor_sensitivities is None:
            self.risk_factor_sensitivities = {}

@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_amount: float
    confidence_level: float
    time_horizon: int
    method: VaRMethod
    expected_shortfall: float = 0.0
    component_vars: Dict[str, float] = None
    
    def __post_init__(self):
        if self.component_vars is None:
            self.component_vars = {}

class MarketRiskModel:
    """
    Comprehensive market risk modeling system.
    
    Features:
    - Multi-factor risk modeling
    - VaR and Expected Shortfall calculations
    - Stress testing and scenario analysis
    - Risk factor correlation modeling
    - Portfolio risk attribution
    - Regulatory capital calculations
    - Model validation and backtesting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market risk model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.market_config = config.get('market_risk', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.MarketRisk")
        
        # Risk factors and data
        self.risk_factors: List[MarketRiskFactor] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Portfolio positions
        self.positions: List[PortfolioPosition] = []
        
        # Model parameters
        self.confidence_levels = self.market_config.get('confidence_levels', [0.95, 0.99])
        self.time_horizons = self.market_config.get('time_horizons', [1, 10])  # days
        self.lookback_period = self.market_config.get('lookback_period', 252)  # trading days
        
        # Calculation results
        self.var_results: Dict[str, VaRResult] = {}
        self.stress_test_results: Dict[str, Any] = {}
        
        # Model validation
        self.backtesting_results: Dict[str, Any] = {}
        
        self.logger.info("Market risk model initialized")
    
    def load_risk_factors(self, risk_factors_data: pd.DataFrame) -> None:
        """
        Load market risk factors.
        
        Args:
            risk_factors_data: DataFrame with risk factor information
        """
        self.logger.info(f"Loading {len(risk_factors_data)} market risk factors")
        
        self.risk_factors = []
        
        for _, row in risk_factors_data.iterrows():
            factor = MarketRiskFactor(
                factor_id=str(row.get('factor_id', '')),
                factor_name=str(row.get('factor_name', '')),
                factor_type=RiskFactorType(row.get('factor_type', 'INTEREST_RATE')),
                current_value=float(row.get('current_value', 0.0)),
                currency=str(row.get('currency', 'USD')),
                volatility=float(row.get('volatility', 0.0)),
                correlation_group=str(row.get('correlation_group', ''))
            )
            self.risk_factors.append(factor)
        
        self.logger.info(f"Loaded {len(self.risk_factors)} risk factors")
    
    def load_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Load historical market data for risk factors.
        
        Args:
            historical_data: Dictionary of historical time series data
        """
        self.logger.info("Loading historical market data")
        
        self.historical_data = historical_data
        
        # Validate data consistency
        for factor_id, data in historical_data.items():
            if 'date' not in data.columns or 'value' not in data.columns:
                self.logger.warning(f"Invalid data format for factor {factor_id}")
                continue
            
            # Ensure data is sorted by date
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
            self.historical_data[factor_id] = data
        
        self.logger.info(f"Historical data loaded for {len(self.historical_data)} factors")
    
    def load_portfolio_positions(self, positions_data: pd.DataFrame) -> None:
        """
        Load portfolio positions for risk calculation.
        
        Args:
            positions_data: DataFrame with position information
        """
        self.logger.info(f"Loading {len(positions_data)} portfolio positions")
        
        self.positions = []
        
        for _, row in positions_data.iterrows():
            # Parse risk factor sensitivities
            sensitivities = {}
            sensitivity_cols = [col for col in row.index if col.startswith('sensitivity_')]
            for col in sensitivity_cols:
                factor_id = col.replace('sensitivity_', '')
                sensitivities[factor_id] = float(row.get(col, 0.0))
            
            position = PortfolioPosition(
                position_id=str(row.get('position_id', '')),
                instrument_type=str(row.get('instrument_type', '')),
                notional_amount=float(row.get('notional_amount', 0.0)),
                market_value=float(row.get('market_value', 0.0)),
                currency=str(row.get('currency', 'USD')),
                risk_factor_sensitivities=sensitivities,
                maturity_date=pd.to_datetime(row.get('maturity_date')) if pd.notna(row.get('maturity_date')) else None
            )
            self.positions.append(position)
        
        self.logger.info(f"Loaded {len(self.positions)} portfolio positions")
    
    def estimate_correlation_matrix(self, method: str = 'ledoit_wolf') -> pd.DataFrame:
        """
        Estimate correlation matrix for risk factors.
        
        Args:
            method: Correlation estimation method ('sample', 'ledoit_wolf', 'shrinkage')
            
        Returns:
            Correlation matrix
        """
        self.logger.info(f"Estimating correlation matrix using {method} method")
        
        if not self.historical_data:
            self.logger.error("No historical data available for correlation estimation")
            return pd.DataFrame()
        
        # Prepare returns data
        returns_data = self._prepare_returns_data()
        
        if returns_data.empty:
            self.logger.error("No valid returns data for correlation estimation")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        if method == 'sample':
            correlation_matrix = returns_data.corr()
        elif method == 'ledoit_wolf':
            # Use Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_data.fillna(0)).covariance_
            
            # Convert covariance to correlation
            std_devs = np.sqrt(np.diag(cov_matrix))
            correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
            correlation_matrix = pd.DataFrame(correlation_matrix, 
                                            index=returns_data.columns, 
                                            columns=returns_data.columns)
        elif method == 'shrinkage':
            # Simple shrinkage towards identity matrix
            sample_corr = returns_data.corr()
            identity = np.eye(len(sample_corr))
            shrinkage_factor = 0.1  # Could be optimized
            correlation_matrix = (1 - shrinkage_factor) * sample_corr + shrinkage_factor * identity
        else:
            correlation_matrix = returns_data.corr()
        
        self.correlation_matrix = correlation_matrix
        self.logger.info(f"Correlation matrix estimated for {len(correlation_matrix)} factors")
        
        return correlation_matrix
    
    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data from historical prices."""
        returns_dict = {}
        
        for factor_id, data in self.historical_data.items():
            if len(data) < 2:
                continue
            
            # Calculate returns
            data = data.sort_values('date')
            returns = data['value'].pct_change().dropna()
            
            if len(returns) > 0:
                returns_dict[factor_id] = returns.values
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Align all series to same length
        min_length = min(len(series) for series in returns_dict.values())
        aligned_returns = {}
        
        for factor_id, returns in returns_dict.items():
            aligned_returns[factor_id] = returns[-min_length:]
        
        return pd.DataFrame(aligned_returns)
    
    def calculate_var(self, confidence_level: float = 0.95, 
                     time_horizon: int = 1, method: VaRMethod = VaRMethod.HISTORICAL) -> VaRResult:
        """
        Calculate Value at Risk for the portfolio.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon: Time horizon in days
            method: VaR calculation method
            
        Returns:
            VaR calculation result
        """
        self.logger.info(f"Calculating {confidence_level*100}% VaR with {method.value} method")
        
        if not self.positions:
            self.logger.error("No portfolio positions loaded")
            return VaRResult(0.0, confidence_level, time_horizon, method)
        
        if method == VaRMethod.HISTORICAL:
            var_result = self._calculate_historical_var(confidence_level, time_horizon)
        elif method == VaRMethod.PARAMETRIC:
            var_result = self._calculate_parametric_var(confidence_level, time_horizon)
        elif method == VaRMethod.MONTE_CARLO:
            var_result = self._calculate_monte_carlo_var(confidence_level, time_horizon)
        else:
            self.logger.error(f"Unsupported VaR method: {method}")
            return VaRResult(0.0, confidence_level, time_horizon, method)
        
        # Store result
        result_key = f"{method.value}_{confidence_level}_{time_horizon}d"
        self.var_results[result_key] = var_result
        
        self.logger.info(f"VaR calculated: ${var_result.var_amount:,.2f}")
        return var_result
    
    def _calculate_historical_var(self, confidence_level: float, time_horizon: int) -> VaRResult:
        """Calculate VaR using historical simulation method."""
        # Get historical returns
        returns_data = self._prepare_returns_data()
        
        if returns_data.empty:
            return VaRResult(0.0, confidence_level, time_horizon, VaRMethod.HISTORICAL)
        
        # Calculate portfolio P&L for each historical scenario
        portfolio_pnl = []
        
        for i in range(len(returns_data)):
            pnl = 0.0
            
            for position in self.positions:
                position_pnl = 0.0
                
                # Calculate position P&L based on risk factor sensitivities
                for factor_id, sensitivity in position.risk_factor_sensitivities.items():
                    if factor_id in returns_data.columns:
                        factor_return = returns_data.iloc[i][factor_id]
                        position_pnl += sensitivity * factor_return
                
                pnl += position_pnl
            
            portfolio_pnl.append(pnl)
        
        portfolio_pnl = np.array(portfolio_pnl)
        
        # Scale for time horizon
        if time_horizon > 1:
            portfolio_pnl = portfolio_pnl * np.sqrt(time_horizon)
        
        # Calculate VaR and Expected Shortfall
        var_percentile = (1 - confidence_level) * 100
        var_amount = -np.percentile(portfolio_pnl, var_percentile)
        
        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = portfolio_pnl[portfolio_pnl <= -var_amount]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_amount
        
        return VaRResult(
            var_amount=var_amount,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.HISTORICAL,
            expected_shortfall=expected_shortfall
        )
    
    def _calculate_parametric_var(self, confidence_level: float, time_horizon: int) -> VaRResult:
        """Calculate VaR using parametric (variance-covariance) method."""
        if self.correlation_matrix is None:
            self.estimate_correlation_matrix()
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return VaRResult(0.0, confidence_level, time_horizon, VaRMethod.PARAMETRIC)
        
        # Calculate portfolio volatility
        portfolio_variance = self._calculate_portfolio_variance()
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Scale for time horizon
        if time_horizon > 1:
            portfolio_volatility = portfolio_volatility * np.sqrt(time_horizon)
        
        # Calculate VaR using normal distribution assumption
        z_score = stats.norm.ppf(confidence_level)
        var_amount = z_score * portfolio_volatility
        
        # Expected Shortfall for normal distribution
        expected_shortfall = portfolio_volatility * stats.norm.pdf(z_score) / (1 - confidence_level)
        
        return VaRResult(
            var_amount=var_amount,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.PARAMETRIC,
            expected_shortfall=expected_shortfall
        )
    
    def _calculate_portfolio_variance(self) -> float:
        """Calculate portfolio variance using risk factor sensitivities."""
        if self.correlation_matrix is None or not self.positions:
            return 0.0
        
        # Get risk factor volatilities
        factor_volatilities = {}
        for factor in self.risk_factors:
            if factor.volatility > 0:
                factor_volatilities[factor.factor_id] = factor.volatility
            else:
                # Estimate from historical data if available
                if factor.factor_id in self.historical_data:
                    returns_data = self._prepare_returns_data()
                    if factor.factor_id in returns_data.columns:
                        factor_volatilities[factor.factor_id] = returns_data[factor.factor_id].std()
        
        # Calculate portfolio sensitivities
        portfolio_sensitivities = {}
        for factor_id in factor_volatilities.keys():
            total_sensitivity = sum(
                position.risk_factor_sensitivities.get(factor_id, 0.0) 
                for position in self.positions
            )
            portfolio_sensitivities[factor_id] = total_sensitivity
        
        # Calculate portfolio variance
        portfolio_variance = 0.0
        
        for factor1_id, sensitivity1 in portfolio_sensitivities.items():
            for factor2_id, sensitivity2 in portfolio_sensitivities.items():
                vol1 = factor_volatilities.get(factor1_id, 0.0)
                vol2 = factor_volatilities.get(factor2_id, 0.0)
                
                if factor1_id in self.correlation_matrix.index and factor2_id in self.correlation_matrix.columns:
                    correlation = self.correlation_matrix.loc[factor1_id, factor2_id]
                else:
                    correlation = 1.0 if factor1_id == factor2_id else 0.0
                
                portfolio_variance += sensitivity1 * sensitivity2 * vol1 * vol2 * correlation
        
        return portfolio_variance
    
    def _calculate_monte_carlo_var(self, confidence_level: float, time_horizon: int, 
                                  n_simulations: int = 10000) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation."""
        if self.correlation_matrix is None:
            self.estimate_correlation_matrix()
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return VaRResult(0.0, confidence_level, time_horizon, VaRMethod.MONTE_CARLO)
        
        # Get risk factor volatilities
        factor_volatilities = {}
        factor_ids = []
        
        for factor in self.risk_factors:
            if factor.factor_id in self.correlation_matrix.index:
                if factor.volatility > 0:
                    factor_volatilities[factor.factor_id] = factor.volatility
                else:
                    # Estimate from historical data
                    returns_data = self._prepare_returns_data()
                    if factor.factor_id in returns_data.columns:
                        factor_volatilities[factor.factor_id] = returns_data[factor.factor_id].std()
                
                if factor.factor_id in factor_volatilities:
                    factor_ids.append(factor.factor_id)
        
        if not factor_ids:
            return VaRResult(0.0, confidence_level, time_horizon, VaRMethod.MONTE_CARLO)
        
        # Prepare correlation matrix and volatilities
        corr_matrix = self.correlation_matrix.loc[factor_ids, factor_ids]
        volatilities = np.array([factor_volatilities[factor_id] for factor_id in factor_ids])
        
        # Generate correlated random shocks
        np.random.seed(42)  # For reproducibility
        random_shocks = np.random.multivariate_normal(
            mean=np.zeros(len(factor_ids)),
            cov=corr_matrix.values,
            size=n_simulations
        )
        
        # Scale by volatilities and time horizon
        if time_horizon > 1:
            time_scaling = np.sqrt(time_horizon)
        else:
            time_scaling = 1.0
        
        factor_returns = random_shocks * volatilities * time_scaling
        
        # Calculate portfolio P&L for each simulation
        portfolio_pnl = []
        
        for sim in range(n_simulations):
            pnl = 0.0
            
            for position in self.positions:
                position_pnl = 0.0
                
                for i, factor_id in enumerate(factor_ids):
                    sensitivity = position.risk_factor_sensitivities.get(factor_id, 0.0)
                    factor_return = factor_returns[sim, i]
                    position_pnl += sensitivity * factor_return
                
                pnl += position_pnl
            
            portfolio_pnl.append(pnl)
        
        portfolio_pnl = np.array(portfolio_pnl)
        
        # Calculate VaR and Expected Shortfall
        var_percentile = (1 - confidence_level) * 100
        var_amount = -np.percentile(portfolio_pnl, var_percentile)
        
        # Expected Shortfall
        tail_losses = portfolio_pnl[portfolio_pnl <= -var_amount]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_amount
        
        return VaRResult(
            var_amount=var_amount,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=VaRMethod.MONTE_CARLO,
            expected_shortfall=expected_shortfall
        )
    
    def calculate_component_var(self, confidence_level: float = 0.95, 
                               time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate component VaR for risk attribution.
        
        Args:
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            
        Returns:
            Component VaR by position or risk factor
        """
        self.logger.info("Calculating component VaR for risk attribution")
        
        component_vars = {}
        
        if not self.positions:
            return component_vars
        
        # Calculate total portfolio VaR
        total_var = self.calculate_var(confidence_level, time_horizon, VaRMethod.PARAMETRIC)
        
        if total_var.var_amount == 0:
            return component_vars
        
        # Calculate marginal VaR for each position
        for position in self.positions:
            # Create temporary portfolio without this position
            original_positions = self.positions.copy()
            self.positions = [pos for pos in self.positions if pos.position_id != position.position_id]
            
            # Calculate VaR without this position
            var_without_position = self.calculate_var(confidence_level, time_horizon, VaRMethod.PARAMETRIC)
            
            # Component VaR is the difference
            component_var = total_var.var_amount - var_without_position.var_amount
            component_vars[position.position_id] = component_var
            
            # Restore original positions
            self.positions = original_positions
        
        self.logger.info(f"Component VaR calculated for {len(component_vars)} positions")
        return component_vars
    
    def stress_test_portfolio(self, stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Perform stress testing on the portfolio.
        
        Args:
            stress_scenarios: List of stress scenarios (factor_id -> shock)
            
        Returns:
            Stress test results
        """
        self.logger.info(f"Running stress tests with {len(stress_scenarios)} scenarios")
        
        stress_results = {
            'scenario_results': {},
            'worst_case_loss': 0.0,
            'best_case_gain': 0.0,
            'scenario_summary': {}
        }
        
        scenario_pnls = []
        
        for i, scenario in enumerate(stress_scenarios):
            scenario_name = f"Scenario_{i+1}"
            scenario_pnl = 0.0
            
            # Calculate P&L for each position under this scenario
            for position in self.positions:
                position_pnl = 0.0
                
                for factor_id, shock in scenario.items():
                    sensitivity = position.risk_factor_sensitivities.get(factor_id, 0.0)
                    position_pnl += sensitivity * shock
                
                scenario_pnl += position_pnl
            
            stress_results['scenario_results'][scenario_name] = {
                'total_pnl': scenario_pnl,
                'scenario_shocks': scenario,
                'position_breakdown': self._calculate_position_pnl_breakdown(scenario)
            }
            
            scenario_pnls.append(scenario_pnl)
        
        # Summary statistics
        if scenario_pnls:
            stress_results['worst_case_loss'] = min(scenario_pnls)
            stress_results['best_case_gain'] = max(scenario_pnls)
            stress_results['scenario_summary'] = {
                'mean_pnl': np.mean(scenario_pnls),
                'std_pnl': np.std(scenario_pnls),
                'min_pnl': min(scenario_pnls),
                'max_pnl': max(scenario_pnls),
                'percentile_5': np.percentile(scenario_pnls, 5),
                'percentile_95': np.percentile(scenario_pnls, 95)
            }
        
        self.stress_test_results = stress_results
        self.logger.info(f"Stress testing completed. Worst case loss: ${stress_results['worst_case_loss']:,.2f}")
        
        return stress_results
    
    def _calculate_position_pnl_breakdown(self, scenario: Dict[str, float]) -> Dict[str, float]:
        """Calculate P&L breakdown by position for a stress scenario."""
        position_pnls = {}
        
        for position in self.positions:
            position_pnl = 0.0
            
            for factor_id, shock in scenario.items():
                sensitivity = position.risk_factor_sensitivities.get(factor_id, 0.0)
                position_pnl += sensitivity * shock
            
            position_pnls[position.position_id] = position_pnl
        
        return position_pnls
    
    def backtest_var_model(self, start_date: datetime, end_date: datetime,
                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Backtest VaR model performance.
        
        Args:
            start_date: Backtesting start date
            end_date: Backtesting end date
            confidence_level: VaR confidence level
            
        Returns:
            Backtesting results
        """
        self.logger.info(f"Backtesting VaR model from {start_date} to {end_date}")
        
        backtest_results = {
            'test_period': {'start': start_date, 'end': end_date},
            'confidence_level': confidence_level,
            'total_observations': 0,
            'var_breaches': 0,
            'breach_rate': 0.0,
            'expected_breach_rate': 1 - confidence_level,
            'kupiec_test': {},
            'christoffersen_test': {},
            'breach_dates': [],
            'daily_results': []
        }
        
        # This would require actual historical P&L data
        # For now, return a placeholder structure
        
        # In a real implementation, you would:
        # 1. Calculate daily VaR predictions
        # 2. Compare with actual P&L
        # 3. Count breaches
        # 4. Perform statistical tests
        
        self.logger.info("VaR backtesting completed (placeholder implementation)")
        return backtest_results
    
    def calculate_regulatory_capital(self) -> Dict[str, float]:
        """
        Calculate regulatory capital requirements for market risk.
        
        Returns:
            Regulatory capital amounts by approach
        """
        self.logger.info("Calculating regulatory market risk capital")
        
        capital_requirements = {
            'standardized_approach': 0.0,
            'internal_models_approach': 0.0,
            'stressed_var': 0.0,
            'incremental_risk_charge': 0.0,
            'comprehensive_risk_measure': 0.0,
            'total_capital_requirement': 0.0
        }
        
        # Calculate VaR-based capital (Internal Models Approach)
        var_99_10d = self.calculate_var(confidence_level=0.99, time_horizon=10, method=VaRMethod.HISTORICAL)
        
        # Regulatory multiplier (typically 3-4)
        multiplier = self.market_config.get('regulatory_multiplier', 3.0)
        capital_requirements['internal_models_approach'] = var_99_10d.var_amount * multiplier
        
        # Stressed VaR (using stressed historical scenarios)
        # This would require implementation of stressed scenarios
        capital_requirements['stressed_var'] = capital_requirements['internal_models_approach'] * 1.5
        
        # Total capital requirement
        capital_requirements['total_capital_requirement'] = max(
            capital_requirements['internal_models_approach'],
            capital_requirements['stressed_var']
        )
        
        self.logger.info(f"Regulatory capital calculated: ${capital_requirements['total_capital_requirement']:,.2f}")
        return capital_requirements
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive market risk report."""
        self.logger.info("Generating market risk report")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'report_type': 'Market Risk Analysis',
                'version': '1.0'
            },
            'portfolio_summary': {},
            'var_summary': {},
            'stress_test_summary': {},
            'risk_factor_analysis': {},
            'regulatory_capital': {},
            'recommendations': []
        }
        
        # Portfolio summary
        if self.positions:
            total_market_value = sum(pos.market_value for pos in self.positions)
            total_notional = sum(pos.notional_amount for pos in self.positions)
            
            instrument_breakdown = {}
            for position in self.positions:
                instrument_type = position.instrument_type
                if instrument_type not in instrument_breakdown:
                    instrument_breakdown[instrument_type] = {'count': 0, 'market_value': 0.0, 'notional': 0.0}
                
                instrument_breakdown[instrument_type]['count'] += 1
                instrument_breakdown[instrument_type]['market_value'] += position.market_value
                instrument_breakdown[instrument_type]['notional'] += position.notional_amount
            
            report['portfolio_summary'] = {
                'total_positions': len(self.positions),
                'total_market_value': total_market_value,
                'total_notional_amount': total_notional,
                'instrument_breakdown': instrument_breakdown
            }
        
        # VaR summary
        if self.var_results:
            var_summary = {}
            for key, var_result in self.var_results.items():
                var_summary[key] = {
                    'var_amount': var_result.var_amount,
                    'expected_shortfall': var_result.expected_shortfall,
                    'confidence_level': var_result.confidence_level,
                    'time_horizon': var_result.time_horizon,
                    'method': var_result.method.value
                }
            report['var_summary'] = var_summary
        
        # Stress test summary
        if self.stress_test_results:
            report['stress_test_summary'] = self.stress_test_results.get('scenario_summary', {})
        
        # Risk factor analysis
        if self.risk_factors:
            factor_analysis = {}
            for factor in self.risk_factors:
                factor_analysis[factor.factor_id] = {
                    'factor_name': factor.factor_name,
                    'factor_type': factor.factor_type.value,
                    'current_value': factor.current_value,
                    'volatility': factor.volatility,
                    'correlation_group': factor.correlation_group
                }
            report['risk_factor_analysis'] = factor_analysis
        
        # Regulatory capital
        report['regulatory_capital'] = self.calculate_regulatory_capital()
        
        # Recommendations
        report['recommendations'] = self._generate_market_risk_recommendations()
        
        self.logger.info("Market risk report generated")
        return report
    
    def _generate_market_risk_recommendations(self) -> List[Dict[str, Any]]:
        """Generate market risk management recommendations."""
        recommendations = []
        
        # VaR model recommendations
        if not self.var_results:
            recommendations.append({
                'category': 'Risk Measurement',
                'priority': 'HIGH',
                'recommendation': 'Implement comprehensive VaR calculation framework',
                'timeline': '1-3 months'
            })
        
        # Correlation modeling
        if self.correlation_matrix is None:
            recommendations.append({
                'category': 'Risk Modeling',
                'priority': 'HIGH',
                'recommendation': 'Develop robust correlation matrix estimation',
                'timeline': '2-4 months'
            })
        
        # Stress testing
        if not self.stress_test_results:
            recommendations.append({
                'category': 'Stress Testing',
                'priority': 'MEDIUM',
                'recommendation': 'Implement comprehensive stress testing framework',
                'timeline': '3-6 months'
            })
        
        # Model validation
        if not self.backtesting_results:
            recommendations.append({
                'category': 'Model Validation',
                'priority': 'MEDIUM',
                'recommendation': 'Implement VaR model backtesting and validation',
                'timeline': '2-4 months'
            })
        
        # Portfolio diversification
        if self.positions:
            # Check for concentration risk
            total_value = sum(pos.market_value for pos in self.positions)
            instrument_concentrations = {}
            
            for position in self.positions:
                instrument_type = position.instrument_type
                if instrument_type not in instrument_concentrations:
                    instrument_concentrations[instrument_type] = 0.0
                instrument_concentrations[instrument_type] += position.market_value
            
            max_concentration = max(conc / total_value for conc in instrument_concentrations.values()) if total_value > 0 else 0
            
            if max_concentration > 0.5:  # 50% concentration threshold
                recommendations.append({
                    'category': 'Portfolio Management',
                    'priority': 'MEDIUM',
                    'recommendation': 'Reduce portfolio concentration risk through diversification',
                    'timeline': '6-12 months'
                })
        
        return recommendations
    
    def export_results(self, filepath: str) -> None:
        """
        Export market risk results to file.
        
        Args:
            filepath: Output file path
        """
        results = {
            'var_results': {key: {
                'var_amount': result.var_amount,
                'confidence_level': result.confidence_level,
                'time_horizon': result.time_horizon,
                'method': result.method.value,
                'expected_shortfall': result.expected_shortfall
            } for key, result in self.var_results.items()},
            'stress_test_results': self.stress_test_results,
            'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else {},
            'portfolio_summary': {
                'total_positions': len(self.positions),
                'total_market_value': sum(pos.market_value for pos in self.positions),
                'risk_factors_count': len(self.risk_factors)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Market risk results exported to {filepath}")


class MarketRiskFactors:
    """
    Market Risk Factors Collection and Management
    
    This class provides a collection of market risk factors and utilities
    for managing and analyzing them in the context of PPNR modeling.
    """
    
    def __init__(self):
        """Initialize market risk factors collection"""
        self.factors = {}
        self.factor_types = {
            'interest_rate': RiskFactorType.INTEREST_RATE,
            'equity': RiskFactorType.EQUITY,
            'fx': RiskFactorType.FX,
            'credit_spread': RiskFactorType.CREDIT_SPREAD,
            'commodity': RiskFactorType.COMMODITY,
            'volatility': RiskFactorType.VOLATILITY
        }
    
    def add_factor(self, name: str, factor_type: str, data: pd.Series) -> None:
        """
        Add a market risk factor
        
        Args:
            name: Factor name
            factor_type: Type of risk factor
            data: Time series data for the factor
        """
        if factor_type in self.factor_types:
            self.factors[name] = MarketRiskFactor(
                name=name,
                factor_type=self.factor_types[factor_type],
                current_value=data.iloc[-1] if not data.empty else 0.0,
                historical_data=data,
                volatility=data.std() if not data.empty else 0.0
            )
    
    def get_factor(self, name: str) -> Optional[MarketRiskFactor]:
        """Get a specific risk factor"""
        return self.factors.get(name)
    
    def list_factors(self) -> List[str]:
        """List all available risk factors"""
        return list(self.factors.keys())
    
    def get_factors_by_type(self, factor_type: str) -> Dict[str, MarketRiskFactor]:
        """Get all factors of a specific type"""
        if factor_type not in self.factor_types:
            return {}
        
        target_type = self.factor_types[factor_type]
        return {name: factor for name, factor in self.factors.items() 
                if factor.factor_type == target_type}