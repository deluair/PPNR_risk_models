"""
Stress Tester for PPNR Models

Coordinates stress testing across all PPNR model components:
- Applies scenarios to models
- Aggregates results
- Calculates capital impact
- Generates stress test reports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings

from ..models.base_model import BaseModel
from .scenario_generator import ScenarioGenerator

class StressTester:
    """
    Main stress testing coordinator for PPNR models.
    
    Features:
    - Multi-model stress testing
    - Scenario application
    - Results aggregation
    - Capital impact calculation
    - Regulatory compliance reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stress tester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stress_config = config.get('stress_testing', {})
        self.regulatory_config = config.get('regulatory', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.StressTester")
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator(config)
        self.models = {}
        self.stress_results = {}
        
    def register_model(self, model_name: str, model: BaseModel) -> None:
        """
        Register a PPNR model for stress testing.
        
        Args:
            model_name: Name of the model
            model: Fitted PPNR model instance
        """
        if not model.is_fitted:
            raise ValueError(f"Model {model_name} must be fitted before registration")
        
        self.models[model_name] = model
        self.logger.info(f"Registered model: {model_name}")
    
    def run_regulatory_stress_test(self, historical_data: pd.DataFrame,
                                 horizon: int = 36) -> Dict[str, pd.DataFrame]:
        """
        Run regulatory stress test (CCAR/DFAST compliant).
        
        Args:
            historical_data: Historical data for scenario generation
            horizon: Stress test horizon in months
            
        Returns:
            Dictionary of stress test results by scenario
        """
        self.logger.info("Starting regulatory stress test...")
        
        # Load historical data for scenario generation
        self.scenario_generator.load_historical_data(historical_data)
        
        # Generate regulatory scenarios
        scenarios = self.scenario_generator.generate_regulatory_scenarios(horizon)
        
        # Run stress test for each scenario
        stress_results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            self.logger.info(f"Running stress test for scenario: {scenario_name}")
            
            scenario_results = self._run_scenario_stress_test(scenario_data, scenario_name)
            stress_results[scenario_name] = scenario_results
        
        # Store results
        self.stress_results = stress_results
        
        self.logger.info("Regulatory stress test completed")
        return stress_results
    
    def _run_scenario_stress_test(self, scenario_data: pd.DataFrame, 
                                scenario_name: str) -> pd.DataFrame:
        """
        Run stress test for a single scenario.
        
        Args:
            scenario_data: Economic scenario data
            scenario_name: Name of the scenario
            
        Returns:
            Aggregated stress test results
        """
        model_results = {}
        
        # Apply scenario to each registered model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    # Standard prediction
                    predictions = model.predict(scenario_data)
                    model_results[f'{model_name}_forecast'] = predictions
                
                # Model-specific stress testing
                if hasattr(model, 'stress_test_trading_revenue') and 'trading' in model_name.lower():
                    # Trading revenue stress test
                    stress_scenarios = self._get_trading_stress_scenarios(scenario_name)
                    trading_stress = model.stress_test_trading_revenue(scenario_data, stress_scenarios)
                    for col in trading_stress.columns:
                        model_results[f'{model_name}_{col}'] = trading_stress[col]
                
                elif hasattr(model, 'forecast_stress_scenario') and 'fee' in model_name.lower():
                    # Fee income stress test
                    stress_factors = self._get_fee_stress_factors(scenario_name)
                    fee_stress = model.forecast_stress_scenario(scenario_data, stress_factors)
                    for col in fee_stress.columns:
                        model_results[f'{model_name}_{col}'] = fee_stress[col]
                
                elif hasattr(model, 'calculate_asset_sensitivity') and 'nii' in model_name.lower():
                    # NII sensitivity analysis
                    sensitivity = model.calculate_asset_sensitivity(scenario_data)
                    for key, value in sensitivity.items():
                        model_results[f'{model_name}_{key}'] = [value] * len(scenario_data)
                
            except Exception as e:
                self.logger.error(f"Error running stress test for {model_name}: {str(e)}")
                continue
        
        # Combine results into DataFrame
        results_df = pd.DataFrame(model_results, index=scenario_data.index)
        
        # Add scenario information
        results_df['scenario'] = scenario_name
        results_df = pd.concat([results_df, scenario_data], axis=1)
        
        return results_df
    
    def _get_trading_stress_scenarios(self, scenario_name: str) -> Dict[str, Dict[str, float]]:
        """Get trading-specific stress scenarios."""
        if scenario_name == 'baseline':
            return {
                'market_stress': {
                    'sp500_return': -0.05,
                    'vix': 0.3,
                    'credit_spreads': 50
                }
            }
        elif scenario_name == 'adverse':
            return {
                'market_stress': {
                    'sp500_return': -0.15,
                    'vix': 0.8,
                    'credit_spreads': 200
                }
            }
        else:  # severely_adverse
            return {
                'market_stress': {
                    'sp500_return': -0.30,
                    'vix': 1.5,
                    'credit_spreads': 400
                }
            }
    
    def _get_fee_stress_factors(self, scenario_name: str) -> Dict[str, float]:
        """Get fee income stress factors."""
        if scenario_name == 'baseline':
            return {
                'vix': 0.2,
                'sp500_return': -0.05,
                'gdp_growth': -0.01
            }
        elif scenario_name == 'adverse':
            return {
                'vix': 0.6,
                'sp500_return': -0.15,
                'gdp_growth': -0.03,
                'unemployment_rate': 0.02
            }
        else:  # severely_adverse
            return {
                'vix': 1.2,
                'sp500_return': -0.30,
                'gdp_growth': -0.05,
                'unemployment_rate': 0.05
            }
    
    def calculate_ppnr_aggregation(self, stress_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate PPNR components across scenarios.
        
        Args:
            stress_results: Dictionary of scenario results
            
        Returns:
            Aggregated PPNR results
        """
        aggregated_results = []
        
        for scenario_name, results in stress_results.items():
            scenario_agg = pd.DataFrame(index=results.index)
            scenario_agg['scenario'] = scenario_name
            
            # Aggregate NII components
            nii_cols = [col for col in results.columns if 'nii' in col.lower() and 'forecast' in col]
            if nii_cols:
                scenario_agg['total_nii'] = results[nii_cols].sum(axis=1)
            
            # Aggregate fee income components
            fee_cols = [col for col in results.columns if 'fee' in col.lower() and 'forecast' in col]
            if fee_cols:
                scenario_agg['total_fee_income'] = results[fee_cols].sum(axis=1)
            
            # Aggregate trading revenue components
            trading_cols = [col for col in results.columns if 'trading' in col.lower() and 'forecast' in col]
            if trading_cols:
                scenario_agg['total_trading_revenue'] = results[trading_cols].sum(axis=1)
            
            # Calculate total PPNR
            ppnr_components = ['total_nii', 'total_fee_income', 'total_trading_revenue']
            available_components = [col for col in ppnr_components if col in scenario_agg.columns]
            
            if available_components:
                scenario_agg['total_ppnr'] = scenario_agg[available_components].sum(axis=1)
            
            # Add economic variables
            econ_vars = ['gdp_growth', 'unemployment_rate', 'fed_funds_rate']
            for var in econ_vars:
                if var in results.columns:
                    scenario_agg[var] = results[var]
            
            aggregated_results.append(scenario_agg)
        
        # Combine all scenarios
        final_results = pd.concat(aggregated_results, ignore_index=False)
        
        return final_results
    
    def calculate_capital_impact(self, ppnr_results: pd.DataFrame, 
                               initial_capital: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate capital impact from PPNR stress test results.
        
        Args:
            ppnr_results: Aggregated PPNR results
            initial_capital: Initial capital ratios
            
        Returns:
            Capital impact analysis
        """
        capital_results = []
        
        # Get regulatory requirements
        ccar_config = self.regulatory_config.get('ccar', {})
        capital_ratios = ccar_config.get('capital_ratios', [])
        minimum_thresholds = ccar_config.get('minimum_thresholds', {})
        
        for scenario in ppnr_results['scenario'].unique():
            scenario_data = ppnr_results[ppnr_results['scenario'] == scenario].copy()
            
            # Calculate cumulative PPNR impact
            if 'total_ppnr' in scenario_data.columns:
                cumulative_ppnr = scenario_data['total_ppnr'].cumsum()
                scenario_data['cumulative_ppnr_impact'] = cumulative_ppnr
            
            # Calculate capital ratio impacts
            for ratio_name in capital_ratios:
                if ratio_name in initial_capital:
                    initial_ratio = initial_capital[ratio_name]
                    
                    # Simplified capital impact calculation
                    # In practice, this would be more complex with RWA changes
                    if 'cumulative_ppnr_impact' in scenario_data.columns:
                        # Assume PPNR flows to capital (simplified)
                        capital_impact = scenario_data['cumulative_ppnr_impact'] * 0.001  # Convert to ratio impact
                        stressed_ratio = initial_ratio + capital_impact
                        scenario_data[f'{ratio_name}_stressed'] = stressed_ratio
                        
                        # Check minimum requirements
                        minimum = minimum_thresholds.get(ratio_name, 0)
                        scenario_data[f'{ratio_name}_breach'] = (stressed_ratio < minimum).astype(int)
            
            capital_results.append(scenario_data)
        
        return pd.concat(capital_results, ignore_index=False)
    
    def generate_stress_test_report(self, stress_results: Dict[str, pd.DataFrame],
                                  output_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive stress test report.
        
        Args:
            stress_results: Dictionary of stress test results
            output_path: Path to save report (optional)
            
        Returns:
            Dictionary containing report data
        """
        report = {
            'metadata': {
                'report_date': datetime.now().isoformat(),
                'scenarios_tested': list(stress_results.keys()),
                'models_included': list(self.models.keys()),
                'stress_test_horizon': len(next(iter(stress_results.values())))
            }
        }
        
        # Aggregate PPNR results
        ppnr_aggregated = self.calculate_ppnr_aggregation(stress_results)
        report['ppnr_results'] = ppnr_aggregated
        
        # Summary statistics by scenario
        scenario_summaries = {}
        for scenario_name in ppnr_aggregated['scenario'].unique():
            scenario_data = ppnr_aggregated[ppnr_aggregated['scenario'] == scenario_name]
            
            summary = {}
            if 'total_ppnr' in scenario_data.columns:
                summary['total_ppnr_sum'] = scenario_data['total_ppnr'].sum()
                summary['total_ppnr_mean'] = scenario_data['total_ppnr'].mean()
                summary['total_ppnr_min'] = scenario_data['total_ppnr'].min()
                summary['total_ppnr_max'] = scenario_data['total_ppnr'].max()
            
            # Component breakdowns
            for component in ['total_nii', 'total_fee_income', 'total_trading_revenue']:
                if component in scenario_data.columns:
                    summary[f'{component}_sum'] = scenario_data[component].sum()
                    summary[f'{component}_contribution_pct'] = (
                        scenario_data[component].sum() / scenario_data['total_ppnr'].sum() * 100
                        if 'total_ppnr' in scenario_data.columns and scenario_data['total_ppnr'].sum() != 0
                        else 0
                    )
            
            scenario_summaries[scenario_name] = summary
        
        report['scenario_summaries'] = scenario_summaries
        
        # Model performance metrics
        model_performance = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                if importance is not None:
                    model_performance[f'{model_name}_top_features'] = importance.head(10).to_dict()
        
        report['model_performance'] = model_performance
        
        # Save report if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                # Convert pandas objects to serializable format
                serializable_report = self._make_serializable(report)
                json.dump(serializable_report, f, indent=2)
            
            self.logger.info(f"Stress test report saved to {output_path}")
        
        return report
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert pandas objects to JSON serializable format."""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_monte_carlo_stress_test(self, historical_data: pd.DataFrame,
                                  n_scenarios: int = 1000,
                                  horizon: int = 36) -> Dict[str, Any]:
        """
        Run Monte Carlo stress test.
        
        Args:
            historical_data: Historical data for calibration
            n_scenarios: Number of Monte Carlo scenarios
            horizon: Forecast horizon in months
            
        Returns:
            Monte Carlo stress test results
        """
        self.logger.info(f"Starting Monte Carlo stress test with {n_scenarios} scenarios...")
        
        # Load historical data
        self.scenario_generator.load_historical_data(historical_data)
        
        # Generate Monte Carlo scenarios
        mc_scenarios = self.scenario_generator.generate_monte_carlo_scenarios(n_scenarios, horizon)
        
        # Run stress test for each scenario
        all_results = []
        
        for scenario_id in range(n_scenarios):
            if scenario_id % 100 == 0:
                self.logger.info(f"Processing scenario {scenario_id}/{n_scenarios}")
            
            scenario_data = mc_scenarios[mc_scenarios['scenario_id'] == scenario_id].drop('scenario_id', axis=1)
            
            if len(scenario_data) > 0:
                scenario_results = self._run_scenario_stress_test(scenario_data, f'mc_{scenario_id}')
                scenario_results['scenario_id'] = scenario_id
                all_results.append(scenario_results)
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=False)
        
        # Calculate statistics
        scenario_stats = self.scenario_generator.calculate_scenario_statistics(combined_results)
        
        mc_results = {
            'raw_results': combined_results,
            'statistics': scenario_stats,
            'n_scenarios': n_scenarios,
            'horizon': horizon
        }
        
        self.logger.info("Monte Carlo stress test completed")
        return mc_results