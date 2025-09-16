"""
Scenario Generator for PPNR Stress Testing

Generates economic scenarios for stress testing including:
- Baseline, adverse, and severely adverse scenarios
- Monte Carlo scenario generation
- Historical scenario replication
- Custom scenario creation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.decomposition import PCA
import warnings

class ScenarioGenerator:
    """
    Economic scenario generator for PPNR stress testing.
    
    Features:
    - Regulatory scenario compliance (CCAR/DFAST)
    - Monte Carlo simulation
    - Historical scenario replication
    - Correlation-aware scenario generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scenario generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stress_config = config.get('stress_testing', {})
        self.scenarios_config = self.stress_config.get('scenarios', {})
        self.monte_carlo_config = self.stress_config.get('monte_carlo', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.ScenarioGenerator")
        
        # Scenario parameters
        self.n_simulations = self.monte_carlo_config.get('simulations', 10000)
        self.confidence_intervals = self.monte_carlo_config.get('confidence_intervals', [0.05, 0.95])
        
        # Historical data for calibration
        self.historical_data = None
        self.correlation_matrix = None
        
    def load_historical_data(self, data: pd.DataFrame) -> None:
        """
        Load historical economic data for scenario calibration.
        
        Args:
            data: Historical economic indicators
        """
        self.historical_data = data.copy()
        
        # Calculate correlation matrix
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.correlation_matrix = data[numeric_cols].corr()
        
        self.logger.info(f"Loaded historical data with {len(data)} observations")
    
    def generate_regulatory_scenarios(self, horizon: int = 36) -> Dict[str, pd.DataFrame]:
        """
        Generate regulatory stress scenarios (CCAR/DFAST compliant).
        
        Args:
            horizon: Forecast horizon in months
            
        Returns:
            Dictionary of scenario DataFrames
        """
        scenarios = {}
        
        # Generate time index
        time_index = pd.date_range(
            start=pd.Timestamp.now(),
            periods=horizon,
            freq='M'
        )
        
        # Baseline scenario
        baseline = self._create_baseline_scenario(time_index)
        scenarios['baseline'] = baseline
        
        # Adverse scenario
        adverse = self._create_adverse_scenario(time_index)
        scenarios['adverse'] = adverse
        
        # Severely adverse scenario
        severely_adverse = self._create_severely_adverse_scenario(time_index)
        scenarios['severely_adverse'] = severely_adverse
        
        self.logger.info(f"Generated {len(scenarios)} regulatory scenarios")
        return scenarios
    
    def _create_baseline_scenario(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create baseline economic scenario."""
        baseline_params = self.scenarios_config.get('baseline', {})
        
        scenario = pd.DataFrame(index=time_index)
        
        # GDP growth path
        gdp_path = baseline_params.get('gdp_growth', [2.1, 2.3, 2.5])
        scenario['gdp_growth'] = self._interpolate_path(gdp_path, len(time_index))
        
        # Unemployment rate path
        unemployment_path = baseline_params.get('unemployment', [3.8, 3.9, 4.0])
        scenario['unemployment_rate'] = self._interpolate_path(unemployment_path, len(time_index))
        
        # Federal funds rate path
        fed_funds_path = baseline_params.get('fed_funds_rate', [5.25, 5.50, 5.75])
        scenario['fed_funds_rate'] = self._interpolate_path(fed_funds_path, len(time_index))
        
        # Derived variables
        scenario['treasury_2y'] = scenario['fed_funds_rate'] + 0.3
        scenario['treasury_10y'] = scenario['fed_funds_rate'] + 0.8
        scenario['mortgage_rate'] = scenario['treasury_10y'] + 1.5
        scenario['credit_spreads'] = 150 + np.random.normal(0, 20, len(time_index))  # 150 bps base
        
        # Market variables
        scenario['vix'] = 18 + np.random.normal(0, 3, len(time_index))
        scenario['sp500_return'] = np.random.normal(0.08/12, 0.15/np.sqrt(12), len(time_index))  # Monthly returns
        
        return scenario
    
    def _create_adverse_scenario(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create adverse economic scenario."""
        adverse_params = self.scenarios_config.get('adverse', {})
        
        scenario = pd.DataFrame(index=time_index)
        
        # GDP growth path (recession)
        gdp_path = adverse_params.get('gdp_growth', [-1.5, -2.0, 1.0])
        scenario['gdp_growth'] = self._interpolate_path(gdp_path, len(time_index))
        
        # Unemployment rate path (rising)
        unemployment_path = adverse_params.get('unemployment', [6.5, 8.0, 7.5])
        scenario['unemployment_rate'] = self._interpolate_path(unemployment_path, len(time_index))
        
        # Federal funds rate path (cutting)
        fed_funds_path = adverse_params.get('fed_funds_rate', [3.0, 2.0, 2.5])
        scenario['fed_funds_rate'] = self._interpolate_path(fed_funds_path, len(time_index))
        
        # Derived variables with stress
        scenario['treasury_2y'] = scenario['fed_funds_rate'] + 0.2
        scenario['treasury_10y'] = scenario['fed_funds_rate'] + 0.6
        scenario['mortgage_rate'] = scenario['treasury_10y'] + 2.0  # Higher spread
        scenario['credit_spreads'] = 300 + np.random.normal(0, 50, len(time_index))  # 300 bps stressed
        
        # Market stress
        scenario['vix'] = 35 + np.random.normal(0, 8, len(time_index))
        scenario['sp500_return'] = np.random.normal(-0.15/12, 0.25/np.sqrt(12), len(time_index))  # Negative returns
        
        return scenario
    
    def _create_severely_adverse_scenario(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create severely adverse economic scenario."""
        severely_adverse_params = self.scenarios_config.get('severely_adverse', {})
        
        scenario = pd.DataFrame(index=time_index)
        
        # GDP growth path (severe recession)
        gdp_path = severely_adverse_params.get('gdp_growth', [-4.0, -3.5, 0.5])
        scenario['gdp_growth'] = self._interpolate_path(gdp_path, len(time_index))
        
        # Unemployment rate path (severe rise)
        unemployment_path = severely_adverse_params.get('unemployment', [10.0, 12.5, 11.0])
        scenario['unemployment_rate'] = self._interpolate_path(unemployment_path, len(time_index))
        
        # Federal funds rate path (zero bound)
        fed_funds_path = severely_adverse_params.get('fed_funds_rate', [0.5, 0.25, 1.0])
        scenario['fed_funds_rate'] = self._interpolate_path(fed_funds_path, len(time_index))
        
        # Derived variables with severe stress
        scenario['treasury_2y'] = scenario['fed_funds_rate'] + 0.1
        scenario['treasury_10y'] = scenario['fed_funds_rate'] + 0.4
        scenario['mortgage_rate'] = scenario['treasury_10y'] + 2.5  # Very high spread
        scenario['credit_spreads'] = 600 + np.random.normal(0, 100, len(time_index))  # 600 bps severely stressed
        
        # Severe market stress
        scenario['vix'] = 50 + np.random.normal(0, 15, len(time_index))
        scenario['sp500_return'] = np.random.normal(-0.30/12, 0.35/np.sqrt(12), len(time_index))  # Severe negative returns
        
        return scenario
    
    def _interpolate_path(self, path_points: List[float], n_periods: int) -> np.ndarray:
        """
        Interpolate a smooth path between key points.
        
        Args:
            path_points: Key points to interpolate between
            n_periods: Number of periods to generate
            
        Returns:
            Interpolated path
        """
        if len(path_points) == 1:
            return np.full(n_periods, path_points[0])
        
        # Create key time points
        key_times = np.linspace(0, n_periods - 1, len(path_points))
        
        # Interpolate
        time_points = np.arange(n_periods)
        interpolated = np.interp(time_points, key_times, path_points)
        
        return interpolated
    
    def generate_monte_carlo_scenarios(self, n_scenarios: int = None, 
                                     horizon: int = 36) -> pd.DataFrame:
        """
        Generate Monte Carlo scenarios using historical calibration.
        
        Args:
            n_scenarios: Number of scenarios to generate
            horizon: Forecast horizon in months
            
        Returns:
            DataFrame with Monte Carlo scenarios
        """
        if n_scenarios is None:
            n_scenarios = self.n_simulations
        
        if self.historical_data is None:
            raise ValueError("Historical data must be loaded first")
        
        # Calculate historical statistics
        numeric_cols = self.historical_data.select_dtypes(include=[np.number]).columns
        means = self.historical_data[numeric_cols].mean()
        cov_matrix = self.historical_data[numeric_cols].cov()
        
        # Generate correlated random scenarios
        scenarios_list = []
        
        for i in range(n_scenarios):
            # Generate correlated random variables
            random_vars = np.random.multivariate_normal(
                mean=means.values,
                cov=cov_matrix.values,
                size=horizon
            )
            
            # Create scenario DataFrame
            scenario_df = pd.DataFrame(
                random_vars,
                columns=numeric_cols,
                index=pd.date_range(
                    start=pd.Timestamp.now(),
                    periods=horizon,
                    freq='M'
                )
            )
            
            scenario_df['scenario_id'] = i
            scenarios_list.append(scenario_df)
        
        # Combine all scenarios
        all_scenarios = pd.concat(scenarios_list, ignore_index=False)
        
        self.logger.info(f"Generated {n_scenarios} Monte Carlo scenarios")
        return all_scenarios
    
    def generate_historical_scenario(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate scenario based on historical period.
        
        Args:
            start_date: Start date for historical period
            end_date: End date for historical period
            
        Returns:
            Historical scenario DataFrame
        """
        if self.historical_data is None:
            raise ValueError("Historical data must be loaded first")
        
        # Filter historical data
        mask = (self.historical_data.index >= start_date) & (self.historical_data.index <= end_date)
        historical_scenario = self.historical_data[mask].copy()
        
        if len(historical_scenario) == 0:
            raise ValueError(f"No data found for period {start_date} to {end_date}")
        
        self.logger.info(f"Generated historical scenario for {start_date} to {end_date}")
        return historical_scenario
    
    def create_custom_scenario(self, scenario_params: Dict[str, Any], 
                             horizon: int = 36) -> pd.DataFrame:
        """
        Create custom scenario with user-defined parameters.
        
        Args:
            scenario_params: Dictionary of scenario parameters
            horizon: Forecast horizon in months
            
        Returns:
            Custom scenario DataFrame
        """
        time_index = pd.date_range(
            start=pd.Timestamp.now(),
            periods=horizon,
            freq='M'
        )
        
        scenario = pd.DataFrame(index=time_index)
        
        # Apply custom parameters
        for variable, params in scenario_params.items():
            if isinstance(params, list):
                # Path specification
                scenario[variable] = self._interpolate_path(params, horizon)
            elif isinstance(params, dict):
                # Statistical specification
                if 'distribution' in params:
                    if params['distribution'] == 'normal':
                        scenario[variable] = np.random.normal(
                            params.get('mean', 0),
                            params.get('std', 1),
                            horizon
                        )
                    elif params['distribution'] == 'uniform':
                        scenario[variable] = np.random.uniform(
                            params.get('low', 0),
                            params.get('high', 1),
                            horizon
                        )
                else:
                    # Constant value
                    scenario[variable] = params.get('value', 0)
            else:
                # Constant value
                scenario[variable] = params
        
        self.logger.info("Generated custom scenario")
        return scenario
    
    def calculate_scenario_statistics(self, scenarios: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate statistics across Monte Carlo scenarios.
        
        Args:
            scenarios: DataFrame with multiple scenarios
            
        Returns:
            Dictionary of statistical summaries
        """
        if 'scenario_id' not in scenarios.columns:
            raise ValueError("Scenarios must include 'scenario_id' column")
        
        # Group by time period
        grouped = scenarios.groupby(scenarios.index)
        
        # Calculate statistics
        stats_dict = {
            'mean': grouped.mean().drop('scenario_id', axis=1),
            'std': grouped.std().drop('scenario_id', axis=1),
            'min': grouped.min().drop('scenario_id', axis=1),
            'max': grouped.max().drop('scenario_id', axis=1)
        }
        
        # Calculate percentiles
        for percentile in [5, 25, 50, 75, 95]:
            stats_dict[f'p{percentile}'] = grouped.quantile(percentile/100).drop('scenario_id', axis=1)
        
        return stats_dict