"""
Operational Risk Modeling Module

Comprehensive operational risk modeling for PPNR:
- Loss data collection and analysis
- Frequency and severity modeling
- Scenario analysis and stress testing
- Key Risk Indicators (KRI) monitoring
- Business Environment and Internal Control Factors (BEICF)
- Advanced Measurement Approach (AMA) implementation
- Regulatory capital calculations
- Risk and control self-assessment (RCSA)
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class EventType(Enum):
    """Basel II operational risk event types."""
    INTERNAL_FRAUD = "Internal Fraud"
    EXTERNAL_FRAUD = "External Fraud"
    EMPLOYMENT_PRACTICES = "Employment Practices and Workplace Safety"
    CLIENTS_PRODUCTS = "Clients, Products & Business Practices"
    DAMAGE_ASSETS = "Damage to Physical Assets"
    BUSINESS_DISRUPTION = "Business Disruption and System Failures"
    EXECUTION_DELIVERY = "Execution, Delivery & Process Management"

class BusinessLine(Enum):
    """Basel II business lines."""
    CORPORATE_FINANCE = "Corporate Finance"
    TRADING_SALES = "Trading & Sales"
    RETAIL_BANKING = "Retail Banking"
    COMMERCIAL_BANKING = "Commercial Banking"
    PAYMENT_SETTLEMENT = "Payment & Settlement"
    AGENCY_SERVICES = "Agency Services"
    ASSET_MANAGEMENT = "Asset Management"
    RETAIL_BROKERAGE = "Retail Brokerage"

class RiskLevel(Enum):
    """Risk level categories."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class OperationalLoss:
    """Individual operational loss event."""
    loss_id: str
    event_date: datetime
    discovery_date: datetime
    event_type: EventType
    business_line: BusinessLine
    gross_loss: float
    recoveries: float = 0.0
    net_loss: float = 0.0
    description: str = ""
    root_cause: str = ""
    
    def __post_init__(self):
        if self.net_loss == 0.0:
            self.net_loss = self.gross_loss - self.recoveries

@dataclass
class KeyRiskIndicator:
    """Key Risk Indicator (KRI) definition."""
    kri_id: str
    kri_name: str
    business_line: BusinessLine
    event_type: EventType
    measurement_frequency: str  # Daily, Weekly, Monthly, Quarterly
    threshold_green: float
    threshold_amber: float
    threshold_red: float
    current_value: float = 0.0
    trend: str = "Stable"  # Improving, Stable, Deteriorating
    
    @property
    def risk_level(self) -> RiskLevel:
        """Determine current risk level based on thresholds."""
        if self.current_value <= self.threshold_green:
            return RiskLevel.LOW
        elif self.current_value <= self.threshold_amber:
            return RiskLevel.MEDIUM
        elif self.current_value <= self.threshold_red:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

@dataclass
class ScenarioAnalysis:
    """Operational risk scenario analysis."""
    scenario_id: str
    scenario_name: str
    event_type: EventType
    business_line: BusinessLine
    frequency_estimate: float  # Expected frequency per year
    severity_estimate: float  # Expected severity (mean loss)
    severity_volatility: float  # Severity standard deviation
    confidence_level: float = 0.95
    expert_judgment: bool = True
    
    @property
    def expected_annual_loss(self) -> float:
        """Calculate expected annual loss."""
        return self.frequency_estimate * self.severity_estimate

class OperationalRiskModel:
    """
    Comprehensive operational risk modeling system.
    
    Features:
    - Loss Data Analysis (LDA)
    - Frequency and severity distribution modeling
    - Scenario analysis and stress testing
    - Key Risk Indicators monitoring
    - Business Environment and Internal Control Factors
    - Advanced Measurement Approach implementation
    - Regulatory capital calculations
    - Risk and Control Self-Assessment integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize operational risk model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.operational_config = config.get('operational_risk', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.OperationalRisk")
        
        # Loss data and analysis
        self.loss_events: List[OperationalLoss] = []
        self.frequency_models: Dict[str, Any] = {}
        self.severity_models: Dict[str, Any] = {}
        
        # Key Risk Indicators
        self.kris: List[KeyRiskIndicator] = []
        self.kri_history: Dict[str, pd.DataFrame] = {}
        
        # Scenario analysis
        self.scenarios: List[ScenarioAnalysis] = []
        
        # Model parameters
        self.confidence_levels = self.operational_config.get('confidence_levels', [0.95, 0.99, 0.999])
        self.time_horizon = self.operational_config.get('time_horizon', 1)  # years
        self.minimum_loss_threshold = self.operational_config.get('minimum_loss_threshold', 10000)
        
        # Results storage
        self.capital_calculations: Dict[str, Any] = {}
        self.stress_test_results: Dict[str, Any] = {}
        
        self.logger.info("Operational risk model initialized")
    
    def load_loss_data(self, loss_data: pd.DataFrame) -> None:
        """
        Load historical operational loss data.
        
        Args:
            loss_data: DataFrame with loss event information
        """
        self.logger.info(f"Loading {len(loss_data)} operational loss events")
        
        self.loss_events = []
        
        for _, row in loss_data.iterrows():
            loss_event = OperationalLoss(
                loss_id=str(row.get('loss_id', '')),
                event_date=pd.to_datetime(row.get('event_date')),
                discovery_date=pd.to_datetime(row.get('discovery_date')),
                event_type=EventType(row.get('event_type', 'EXECUTION_DELIVERY')),
                business_line=BusinessLine(row.get('business_line', 'RETAIL_BANKING')),
                gross_loss=float(row.get('gross_loss', 0.0)),
                recoveries=float(row.get('recoveries', 0.0)),
                net_loss=float(row.get('net_loss', 0.0)),
                description=str(row.get('description', '')),
                root_cause=str(row.get('root_cause', ''))
            )
            
            # Filter by minimum threshold
            if loss_event.net_loss >= self.minimum_loss_threshold:
                self.loss_events.append(loss_event)
        
        self.logger.info(f"Loaded {len(self.loss_events)} loss events above threshold ${self.minimum_loss_threshold:,.0f}")
    
    def load_kri_data(self, kri_definitions: pd.DataFrame, kri_history: Dict[str, pd.DataFrame]) -> None:
        """
        Load Key Risk Indicators data.
        
        Args:
            kri_definitions: DataFrame with KRI definitions
            kri_history: Dictionary of historical KRI values
        """
        self.logger.info(f"Loading {len(kri_definitions)} Key Risk Indicators")
        
        self.kris = []
        
        for _, row in kri_definitions.iterrows():
            kri = KeyRiskIndicator(
                kri_id=str(row.get('kri_id', '')),
                kri_name=str(row.get('kri_name', '')),
                business_line=BusinessLine(row.get('business_line', 'RETAIL_BANKING')),
                event_type=EventType(row.get('event_type', 'EXECUTION_DELIVERY')),
                measurement_frequency=str(row.get('measurement_frequency', 'Monthly')),
                threshold_green=float(row.get('threshold_green', 0.0)),
                threshold_amber=float(row.get('threshold_amber', 0.0)),
                threshold_red=float(row.get('threshold_red', 0.0)),
                current_value=float(row.get('current_value', 0.0)),
                trend=str(row.get('trend', 'Stable'))
            )
            self.kris.append(kri)
        
        self.kri_history = kri_history
        self.logger.info(f"Loaded {len(self.kris)} KRIs with historical data for {len(kri_history)} indicators")
    
    def load_scenarios(self, scenarios_data: pd.DataFrame) -> None:
        """
        Load scenario analysis data.
        
        Args:
            scenarios_data: DataFrame with scenario definitions
        """
        self.logger.info(f"Loading {len(scenarios_data)} operational risk scenarios")
        
        self.scenarios = []
        
        for _, row in scenarios_data.iterrows():
            scenario = ScenarioAnalysis(
                scenario_id=str(row.get('scenario_id', '')),
                scenario_name=str(row.get('scenario_name', '')),
                event_type=EventType(row.get('event_type', 'EXECUTION_DELIVERY')),
                business_line=BusinessLine(row.get('business_line', 'RETAIL_BANKING')),
                frequency_estimate=float(row.get('frequency_estimate', 0.0)),
                severity_estimate=float(row.get('severity_estimate', 0.0)),
                severity_volatility=float(row.get('severity_volatility', 0.0)),
                confidence_level=float(row.get('confidence_level', 0.95)),
                expert_judgment=bool(row.get('expert_judgment', True))
            )
            self.scenarios.append(scenario)
        
        self.logger.info(f"Loaded {len(self.scenarios)} operational risk scenarios")
    
    def analyze_loss_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive loss data analysis.
        
        Returns:
            Loss data analysis results
        """
        self.logger.info("Performing loss data analysis")
        
        analysis_results = {
            'summary_statistics': {},
            'event_type_analysis': {},
            'business_line_analysis': {},
            'temporal_analysis': {},
            'severity_analysis': {},
            'frequency_analysis': {}
        }
        
        if not self.loss_events:
            self.logger.warning("No loss events available for analysis")
            return analysis_results
        
        # Convert to DataFrame for analysis
        loss_df = self._convert_losses_to_dataframe()
        
        # Summary statistics
        analysis_results['summary_statistics'] = {
            'total_events': len(self.loss_events),
            'total_gross_loss': loss_df['gross_loss'].sum(),
            'total_net_loss': loss_df['net_loss'].sum(),
            'total_recoveries': loss_df['recoveries'].sum(),
            'average_loss': loss_df['net_loss'].mean(),
            'median_loss': loss_df['net_loss'].median(),
            'max_loss': loss_df['net_loss'].max(),
            'min_loss': loss_df['net_loss'].min(),
            'loss_std': loss_df['net_loss'].std()
        }
        
        # Event type analysis
        event_type_stats = loss_df.groupby('event_type').agg({
            'net_loss': ['count', 'sum', 'mean', 'std'],
            'gross_loss': 'sum',
            'recoveries': 'sum'
        }).round(2)
        
        analysis_results['event_type_analysis'] = event_type_stats.to_dict()
        
        # Business line analysis
        business_line_stats = loss_df.groupby('business_line').agg({
            'net_loss': ['count', 'sum', 'mean', 'std'],
            'gross_loss': 'sum',
            'recoveries': 'sum'
        }).round(2)
        
        analysis_results['business_line_analysis'] = business_line_stats.to_dict()
        
        # Temporal analysis
        loss_df['year'] = loss_df['event_date'].dt.year
        loss_df['month'] = loss_df['event_date'].dt.month
        
        yearly_stats = loss_df.groupby('year').agg({
            'net_loss': ['count', 'sum', 'mean']
        }).round(2)
        
        analysis_results['temporal_analysis'] = {
            'yearly_statistics': yearly_stats.to_dict(),
            'trend_analysis': self._analyze_loss_trends(loss_df)
        }
        
        # Severity analysis
        analysis_results['severity_analysis'] = self._analyze_severity_distribution(loss_df)
        
        # Frequency analysis
        analysis_results['frequency_analysis'] = self._analyze_frequency_distribution(loss_df)
        
        self.logger.info("Loss data analysis completed")
        return analysis_results
    
    def _convert_losses_to_dataframe(self) -> pd.DataFrame:
        """Convert loss events to DataFrame for analysis."""
        data = []
        for loss in self.loss_events:
            data.append({
                'loss_id': loss.loss_id,
                'event_date': loss.event_date,
                'discovery_date': loss.discovery_date,
                'event_type': loss.event_type.value,
                'business_line': loss.business_line.value,
                'gross_loss': loss.gross_loss,
                'recoveries': loss.recoveries,
                'net_loss': loss.net_loss,
                'description': loss.description,
                'root_cause': loss.root_cause
            })
        
        return pd.DataFrame(data)
    
    def _analyze_loss_trends(self, loss_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal trends in loss data."""
        trends = {}
        
        # Annual frequency trend
        annual_counts = loss_df.groupby('year').size()
        if len(annual_counts) > 1:
            # Simple linear trend
            years = annual_counts.index.values
            counts = annual_counts.values
            trend_coef = np.polyfit(years, counts, 1)[0]
            trends['frequency_trend'] = 'Increasing' if trend_coef > 0 else 'Decreasing' if trend_coef < 0 else 'Stable'
        else:
            trends['frequency_trend'] = 'Insufficient data'
        
        # Annual severity trend
        annual_severity = loss_df.groupby('year')['net_loss'].sum()
        if len(annual_severity) > 1:
            years = annual_severity.index.values
            severity = annual_severity.values
            trend_coef = np.polyfit(years, severity, 1)[0]
            trends['severity_trend'] = 'Increasing' if trend_coef > 0 else 'Decreasing' if trend_coef < 0 else 'Stable'
        else:
            trends['severity_trend'] = 'Insufficient data'
        
        return trends
    
    def _analyze_severity_distribution(self, loss_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze severity distribution of losses."""
        severity_analysis = {}
        
        losses = loss_df['net_loss'].values
        
        # Distribution fitting
        distributions = ['lognorm', 'gamma', 'weibull_min', 'pareto']
        best_fit = None
        best_aic = float('inf')
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(losses)
                
                # Calculate AIC
                log_likelihood = np.sum(dist.logpdf(losses, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                
                if aic < best_aic:
                    best_aic = aic
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'aic': aic
                    }
            except:
                continue
        
        severity_analysis['best_fit_distribution'] = best_fit
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99, 99.9]
        severity_analysis['percentiles'] = {
            f'p{p}': np.percentile(losses, p) for p in percentiles
        }
        
        # Tail analysis
        tail_threshold = np.percentile(losses, 95)
        tail_losses = losses[losses >= tail_threshold]
        
        severity_analysis['tail_analysis'] = {
            'tail_threshold': tail_threshold,
            'tail_count': len(tail_losses),
            'tail_percentage': len(tail_losses) / len(losses) * 100,
            'tail_mean': np.mean(tail_losses),
            'tail_std': np.std(tail_losses)
        }
        
        return severity_analysis
    
    def _analyze_frequency_distribution(self, loss_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze frequency distribution of losses."""
        frequency_analysis = {}
        
        # Annual frequency
        annual_counts = loss_df.groupby('year').size()
        
        if len(annual_counts) > 0:
            frequency_analysis['annual_statistics'] = {
                'mean_frequency': annual_counts.mean(),
                'std_frequency': annual_counts.std(),
                'min_frequency': annual_counts.min(),
                'max_frequency': annual_counts.max()
            }
            
            # Fit Poisson distribution
            mean_freq = annual_counts.mean()
            frequency_analysis['poisson_fit'] = {
                'lambda': mean_freq,
                'goodness_of_fit': self._test_poisson_fit(annual_counts.values, mean_freq)
            }
        
        return frequency_analysis
    
    def _test_poisson_fit(self, observed_frequencies: np.ndarray, lambda_param: float) -> Dict[str, float]:
        """Test goodness of fit for Poisson distribution."""
        try:
            # Chi-square goodness of fit test
            max_freq = int(observed_frequencies.max())
            expected_probs = [stats.poisson.pmf(k, lambda_param) for k in range(max_freq + 1)]
            
            # Combine low probability bins
            min_expected = 5
            combined_expected = []
            combined_observed = []
            
            current_expected = 0
            current_observed = 0
            
            for k in range(max_freq + 1):
                freq_count = np.sum(observed_frequencies == k)
                current_expected += expected_probs[k] * len(observed_frequencies)
                current_observed += freq_count
                
                if current_expected >= min_expected or k == max_freq:
                    combined_expected.append(current_expected)
                    combined_observed.append(current_observed)
                    current_expected = 0
                    current_observed = 0
            
            if len(combined_expected) > 1:
                chi2_stat = np.sum((np.array(combined_observed) - np.array(combined_expected))**2 / np.array(combined_expected))
                p_value = 1 - stats.chi2.cdf(chi2_stat, len(combined_expected) - 2)  # -1 for estimated parameter, -1 for df
                
                return {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(combined_expected) - 2
                }
        except:
            pass
        
        return {'chi2_statistic': 0.0, 'p_value': 1.0, 'degrees_of_freedom': 0}
    
    def fit_frequency_models(self) -> Dict[str, Any]:
        """
        Fit frequency models for different event types and business lines.
        
        Returns:
            Frequency model fitting results
        """
        self.logger.info("Fitting frequency models")
        
        fitting_results = {
            'event_type_models': {},
            'business_line_models': {},
            'combined_model': {}
        }
        
        if not self.loss_events:
            self.logger.warning("No loss events available for frequency modeling")
            return fitting_results
        
        loss_df = self._convert_losses_to_dataframe()
        
        # Fit models by event type
        for event_type in EventType:
            event_losses = loss_df[loss_df['event_type'] == event_type.value]
            if len(event_losses) > 0:
                annual_counts = event_losses.groupby('year').size()
                if len(annual_counts) > 0:
                    lambda_param = annual_counts.mean()
                    self.frequency_models[f'event_type_{event_type.value}'] = {
                        'distribution': 'poisson',
                        'lambda': lambda_param,
                        'sample_size': len(annual_counts)
                    }
                    fitting_results['event_type_models'][event_type.value] = {
                        'lambda': lambda_param,
                        'sample_size': len(annual_counts)
                    }
        
        # Fit models by business line
        for business_line in BusinessLine:
            bl_losses = loss_df[loss_df['business_line'] == business_line.value]
            if len(bl_losses) > 0:
                annual_counts = bl_losses.groupby('year').size()
                if len(annual_counts) > 0:
                    lambda_param = annual_counts.mean()
                    self.frequency_models[f'business_line_{business_line.value}'] = {
                        'distribution': 'poisson',
                        'lambda': lambda_param,
                        'sample_size': len(annual_counts)
                    }
                    fitting_results['business_line_models'][business_line.value] = {
                        'lambda': lambda_param,
                        'sample_size': len(annual_counts)
                    }
        
        # Combined model
        annual_counts = loss_df.groupby('year').size()
        if len(annual_counts) > 0:
            lambda_param = annual_counts.mean()
            self.frequency_models['combined'] = {
                'distribution': 'poisson',
                'lambda': lambda_param,
                'sample_size': len(annual_counts)
            }
            fitting_results['combined_model'] = {
                'lambda': lambda_param,
                'sample_size': len(annual_counts)
            }
        
        self.logger.info(f"Frequency models fitted for {len(self.frequency_models)} categories")
        return fitting_results
    
    def fit_severity_models(self) -> Dict[str, Any]:
        """
        Fit severity models for different event types and business lines.
        
        Returns:
            Severity model fitting results
        """
        self.logger.info("Fitting severity models")
        
        fitting_results = {
            'event_type_models': {},
            'business_line_models': {},
            'combined_model': {}
        }
        
        if not self.loss_events:
            self.logger.warning("No loss events available for severity modeling")
            return fitting_results
        
        loss_df = self._convert_losses_to_dataframe()
        
        # Fit models by event type
        for event_type in EventType:
            event_losses = loss_df[loss_df['event_type'] == event_type.value]
            if len(event_losses) >= 10:  # Minimum sample size for fitting
                severity_data = event_losses['net_loss'].values
                model_result = self._fit_severity_distribution(severity_data)
                
                if model_result:
                    self.severity_models[f'event_type_{event_type.value}'] = model_result
                    fitting_results['event_type_models'][event_type.value] = {
                        'distribution': model_result['distribution'],
                        'parameters': model_result['parameters'],
                        'aic': model_result['aic'],
                        'sample_size': len(severity_data)
                    }
        
        # Fit models by business line
        for business_line in BusinessLine:
            bl_losses = loss_df[loss_df['business_line'] == business_line.value]
            if len(bl_losses) >= 10:
                severity_data = bl_losses['net_loss'].values
                model_result = self._fit_severity_distribution(severity_data)
                
                if model_result:
                    self.severity_models[f'business_line_{business_line.value}'] = model_result
                    fitting_results['business_line_models'][business_line.value] = {
                        'distribution': model_result['distribution'],
                        'parameters': model_result['parameters'],
                        'aic': model_result['aic'],
                        'sample_size': len(severity_data)
                    }
        
        # Combined model
        all_losses = loss_df['net_loss'].values
        if len(all_losses) >= 10:
            model_result = self._fit_severity_distribution(all_losses)
            
            if model_result:
                self.severity_models['combined'] = model_result
                fitting_results['combined_model'] = {
                    'distribution': model_result['distribution'],
                    'parameters': model_result['parameters'],
                    'aic': model_result['aic'],
                    'sample_size': len(all_losses)
                }
        
        self.logger.info(f"Severity models fitted for {len(self.severity_models)} categories")
        return fitting_results
    
    def _fit_severity_distribution(self, severity_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fit severity distribution to loss data."""
        distributions = ['lognorm', 'gamma', 'weibull_min', 'pareto', 'genpareto']
        best_fit = None
        best_aic = float('inf')
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                
                # Fit distribution
                if dist_name == 'genpareto':
                    # For GPD, use threshold approach
                    threshold = np.percentile(severity_data, 90)
                    exceedances = severity_data[severity_data > threshold] - threshold
                    if len(exceedances) < 10:
                        continue
                    params = dist.fit(exceedances)
                    # Adjust for threshold
                    log_likelihood = np.sum(dist.logpdf(exceedances, *params))
                else:
                    params = dist.fit(severity_data)
                    log_likelihood = np.sum(dist.logpdf(severity_data, *params))
                
                # Calculate AIC
                aic = 2 * len(params) - 2 * log_likelihood
                
                if aic < best_aic and not np.isnan(aic) and not np.isinf(aic):
                    best_aic = aic
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'aic': aic,
                        'log_likelihood': log_likelihood
                    }
                    
                    if dist_name == 'genpareto':
                        best_fit['threshold'] = threshold
                        
            except Exception as e:
                self.logger.debug(f"Failed to fit {dist_name}: {str(e)}")
                continue
        
        return best_fit
    
    def calculate_operational_var(self, confidence_level: float = 0.99, 
                                 time_horizon: int = 1, method: str = 'monte_carlo',
                                 n_simulations: int = 100000) -> Dict[str, Any]:
        """
        Calculate Operational Value at Risk.
        
        Args:
            confidence_level: Confidence level (e.g., 0.99 for 99% VaR)
            time_horizon: Time horizon in years
            method: Calculation method ('monte_carlo', 'analytical')
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Operational VaR results
        """
        self.logger.info(f"Calculating Operational VaR at {confidence_level*100}% confidence level")
        
        var_results = {
            'var_amount': 0.0,
            'expected_loss': 0.0,
            'unexpected_loss': 0.0,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'method': method,
            'component_breakdown': {}
        }
        
        if method == 'monte_carlo':
            var_results = self._calculate_monte_carlo_var(confidence_level, time_horizon, n_simulations)
        elif method == 'analytical':
            var_results = self._calculate_analytical_var(confidence_level, time_horizon)
        else:
            self.logger.error(f"Unsupported VaR calculation method: {method}")
        
        return var_results
    
    def _calculate_monte_carlo_var(self, confidence_level: float, time_horizon: int, 
                                  n_simulations: int) -> Dict[str, Any]:
        """Calculate Operational VaR using Monte Carlo simulation."""
        simulated_losses = []
        
        # Combine loss data analysis and scenario analysis
        loss_sources = []
        
        # Add fitted models
        for model_key, freq_model in self.frequency_models.items():
            if model_key in self.severity_models:
                severity_model = self.severity_models[model_key]
                loss_sources.append({
                    'name': model_key,
                    'frequency_model': freq_model,
                    'severity_model': severity_model,
                    'type': 'fitted'
                })
        
        # Add scenarios
        for scenario in self.scenarios:
            loss_sources.append({
                'name': scenario.scenario_id,
                'scenario': scenario,
                'type': 'scenario'
            })
        
        if not loss_sources:
            self.logger.warning("No loss sources available for VaR calculation")
            return {
                'var_amount': 0.0,
                'expected_loss': 0.0,
                'unexpected_loss': 0.0,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'method': 'monte_carlo',
                'component_breakdown': {}
            }
        
        # Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        
        for sim in range(n_simulations):
            total_loss = 0.0
            
            for source in loss_sources:
                if source['type'] == 'fitted':
                    # Simulate from fitted models
                    freq_model = source['frequency_model']
                    severity_model = source['severity_model']
                    
                    # Simulate frequency
                    if freq_model['distribution'] == 'poisson':
                        n_events = np.random.poisson(freq_model['lambda'] * time_horizon)
                    else:
                        n_events = 0
                    
                    # Simulate severities
                    if n_events > 0:
                        dist_name = severity_model['distribution']
                        params = severity_model['parameters']
                        
                        try:
                            dist = getattr(stats, dist_name)
                            if dist_name == 'genpareto' and 'threshold' in severity_model:
                                # GPD with threshold
                                threshold = severity_model['threshold']
                                exceedances = dist.rvs(*params, size=n_events)
                                severities = exceedances + threshold
                            else:
                                severities = dist.rvs(*params, size=n_events)
                            
                            total_loss += np.sum(severities)
                        except:
                            # Fallback to empirical distribution
                            pass
                
                elif source['type'] == 'scenario':
                    # Simulate from scenario
                    scenario = source['scenario']
                    
                    # Simulate frequency
                    n_events = np.random.poisson(scenario.frequency_estimate * time_horizon)
                    
                    # Simulate severities (assuming lognormal)
                    if n_events > 0:
                        # Convert to lognormal parameters
                        mu = np.log(scenario.severity_estimate)
                        sigma = scenario.severity_volatility / scenario.severity_estimate
                        
                        severities = np.random.lognormal(mu, sigma, n_events)
                        total_loss += np.sum(severities)
            
            simulated_losses.append(total_loss)
        
        simulated_losses = np.array(simulated_losses)
        
        # Calculate VaR and other metrics
        expected_loss = np.mean(simulated_losses)
        var_amount = np.percentile(simulated_losses, confidence_level * 100)
        unexpected_loss = var_amount - expected_loss
        
        # Component breakdown
        component_breakdown = {}
        for source in loss_sources:
            # This would require separate simulations for each component
            # For now, provide proportional allocation
            if source['type'] == 'scenario':
                scenario = source['scenario']
                component_el = scenario.expected_annual_loss * time_horizon
                component_breakdown[source['name']] = {
                    'expected_loss': component_el,
                    'contribution_pct': (component_el / expected_loss * 100) if expected_loss > 0 else 0
                }
        
        return {
            'var_amount': var_amount,
            'expected_loss': expected_loss,
            'unexpected_loss': unexpected_loss,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'method': 'monte_carlo',
            'component_breakdown': component_breakdown,
            'simulation_statistics': {
                'n_simulations': n_simulations,
                'mean': expected_loss,
                'std': np.std(simulated_losses),
                'min': np.min(simulated_losses),
                'max': np.max(simulated_losses),
                'percentiles': {
                    'p95': np.percentile(simulated_losses, 95),
                    'p99': np.percentile(simulated_losses, 99),
                    'p99.9': np.percentile(simulated_losses, 99.9)
                }
            }
        }
    
    def _calculate_analytical_var(self, confidence_level: float, time_horizon: int) -> Dict[str, Any]:
        """Calculate Operational VaR using analytical approach."""
        # Simplified analytical approach
        # In practice, this would use more sophisticated methods like Panjer recursion
        
        total_expected_loss = 0.0
        total_variance = 0.0
        
        # Calculate from scenarios
        for scenario in self.scenarios:
            annual_el = scenario.expected_annual_loss
            annual_variance = scenario.frequency_estimate * (scenario.severity_estimate**2 + scenario.severity_volatility**2)
            
            total_expected_loss += annual_el * time_horizon
            total_variance += annual_variance * time_horizon
        
        # Assume normal approximation for large portfolios
        if total_variance > 0:
            total_std = np.sqrt(total_variance)
            z_score = stats.norm.ppf(confidence_level)
            var_amount = total_expected_loss + z_score * total_std
        else:
            var_amount = total_expected_loss
        
        return {
            'var_amount': var_amount,
            'expected_loss': total_expected_loss,
            'unexpected_loss': var_amount - total_expected_loss,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'method': 'analytical',
            'component_breakdown': {}
        }
    
    def calculate_regulatory_capital(self) -> Dict[str, Any]:
        """
        Calculate regulatory capital for operational risk.
        
        Returns:
            Regulatory capital calculations
        """
        self.logger.info("Calculating operational risk regulatory capital")
        
        capital_results = {
            'basic_indicator_approach': 0.0,
            'standardized_approach': 0.0,
            'advanced_measurement_approach': 0.0,
            'selected_approach': 'AMA',
            'capital_requirement': 0.0
        }
        
        # Advanced Measurement Approach (AMA)
        # Use 99.9% VaR over 1-year horizon
        var_result = self.calculate_operational_var(confidence_level=0.999, time_horizon=1)
        capital_results['advanced_measurement_approach'] = var_result['var_amount']
        
        # Basic Indicator Approach (15% of gross income)
        # This would require gross income data
        gross_income = self.operational_config.get('gross_income', 0.0)
        capital_results['basic_indicator_approach'] = gross_income * 0.15
        
        # Standardized Approach (beta factors by business line)
        # This would require business line income data
        capital_results['standardized_approach'] = self._calculate_standardized_approach()
        
        # Select approach (typically AMA for large banks)
        capital_results['selected_approach'] = 'AMA'
        capital_results['capital_requirement'] = capital_results['advanced_measurement_approach']
        
        self.capital_calculations = capital_results
        self.logger.info(f"Operational risk capital calculated: ${capital_results['capital_requirement']:,.2f}")
        
        return capital_results
    
    def _calculate_standardized_approach(self) -> float:
        """Calculate capital using Standardized Approach."""
        # Beta factors by business line (Basel II)
        beta_factors = {
            BusinessLine.CORPORATE_FINANCE: 0.18,
            BusinessLine.TRADING_SALES: 0.18,
            BusinessLine.RETAIL_BANKING: 0.12,
            BusinessLine.COMMERCIAL_BANKING: 0.15,
            BusinessLine.PAYMENT_SETTLEMENT: 0.18,
            BusinessLine.AGENCY_SERVICES: 0.15,
            BusinessLine.ASSET_MANAGEMENT: 0.12,
            BusinessLine.RETAIL_BROKERAGE: 0.12
        }
        
        total_capital = 0.0
        
        # This would require business line income data
        # For now, return placeholder
        business_line_income = self.operational_config.get('business_line_income', {})
        
        for business_line, beta in beta_factors.items():
            income = business_line_income.get(business_line.value, 0.0)
            total_capital += beta * income
        
        return total_capital
    
    def analyze_kri_performance(self) -> Dict[str, Any]:
        """
        Analyze Key Risk Indicator performance.
        
        Returns:
            KRI analysis results
        """
        self.logger.info("Analyzing KRI performance")
        
        kri_analysis = {
            'current_status': {},
            'trend_analysis': {},
            'risk_level_distribution': {},
            'alerts': []
        }
        
        if not self.kris:
            self.logger.warning("No KRIs available for analysis")
            return kri_analysis
        
        # Current status
        risk_level_counts = {level.value: 0 for level in RiskLevel}
        
        for kri in self.kris:
            risk_level = kri.risk_level
            risk_level_counts[risk_level.value] += 1
            
            kri_analysis['current_status'][kri.kri_id] = {
                'kri_name': kri.kri_name,
                'current_value': kri.current_value,
                'risk_level': risk_level.value,
                'trend': kri.trend,
                'business_line': kri.business_line.value,
                'event_type': kri.event_type.value
            }
            
            # Generate alerts for high-risk KRIs
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                kri_analysis['alerts'].append({
                    'kri_id': kri.kri_id,
                    'kri_name': kri.kri_name,
                    'risk_level': risk_level.value,
                    'current_value': kri.current_value,
                    'threshold_exceeded': kri.threshold_red if risk_level == RiskLevel.CRITICAL else kri.threshold_amber,
                    'business_line': kri.business_line.value
                })
        
        kri_analysis['risk_level_distribution'] = risk_level_counts
        
        # Trend analysis
        trend_counts = {'Improving': 0, 'Stable': 0, 'Deteriorating': 0}
        for kri in self.kris:
            trend_counts[kri.trend] += 1
        
        kri_analysis['trend_analysis'] = trend_counts
        
        self.logger.info(f"KRI analysis completed for {len(self.kris)} indicators")
        return kri_analysis
    
    def stress_test_operational_risk(self, stress_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform operational risk stress testing.
        
        Args:
            stress_scenarios: List of stress scenario definitions
            
        Returns:
            Stress test results
        """
        self.logger.info(f"Running operational risk stress tests with {len(stress_scenarios)} scenarios")
        
        stress_results = {
            'baseline_capital': 0.0,
            'scenario_results': {},
            'worst_case_scenario': {},
            'stress_impact': {}
        }
        
        # Calculate baseline capital
        baseline_capital = self.calculate_regulatory_capital()
        stress_results['baseline_capital'] = baseline_capital['capital_requirement']
        
        scenario_capitals = []
        
        for i, scenario in enumerate(stress_scenarios):
            scenario_name = f"Stress_Scenario_{i+1}"
            
            # Apply stress to scenarios and KRIs
            stressed_scenarios = self._apply_stress_to_scenarios(scenario)
            
            # Temporarily replace scenarios
            original_scenarios = self.scenarios.copy()
            self.scenarios = stressed_scenarios
            
            # Calculate stressed capital
            stressed_capital = self.calculate_regulatory_capital()
            scenario_capital = stressed_capital['capital_requirement']
            scenario_capitals.append(scenario_capital)
            
            stress_results['scenario_results'][scenario_name] = {
                'capital_requirement': scenario_capital,
                'capital_increase': scenario_capital - stress_results['baseline_capital'],
                'capital_multiple': scenario_capital / stress_results['baseline_capital'] if stress_results['baseline_capital'] > 0 else 1.0,
                'scenario_description': scenario.get('description', ''),
                'stress_factors': scenario
            }
            
            # Restore original scenarios
            self.scenarios = original_scenarios
        
        # Identify worst case scenario
        if scenario_capitals:
            max_capital = max(scenario_capitals)
            worst_case_idx = scenario_capitals.index(max_capital)
            
            stress_results['worst_case_scenario'] = {
                'scenario_name': f"Stress_Scenario_{worst_case_idx+1}",
                'capital_requirement': max_capital,
                'capital_increase': max_capital - stress_results['baseline_capital'],
                'scenario_factors': stress_scenarios[worst_case_idx]
            }
            
            stress_results['stress_impact'] = {
                'maximum_capital': max_capital,
                'maximum_increase': max_capital - stress_results['baseline_capital'],
                'maximum_multiple': max_capital / stress_results['baseline_capital'] if stress_results['baseline_capital'] > 0 else 1.0,
                'average_capital': np.mean(scenario_capitals),
                'capital_volatility': np.std(scenario_capitals)
            }
        
        self.stress_test_results = stress_results
        self.logger.info(f"Operational risk stress testing completed")
        
        return stress_results
    
    def _apply_stress_to_scenarios(self, stress_scenario: Dict[str, Any]) -> List[ScenarioAnalysis]:
        """Apply stress factors to operational risk scenarios."""
        stressed_scenarios = []
        
        for scenario in self.scenarios:
            # Create stressed version of scenario
            frequency_multiplier = stress_scenario.get('frequency_multiplier', 1.0)
            severity_multiplier = stress_scenario.get('severity_multiplier', 1.0)
            volatility_multiplier = stress_scenario.get('volatility_multiplier', 1.0)
            
            # Apply event type specific stress
            event_type_stress = stress_scenario.get('event_type_stress', {})
            if scenario.event_type.value in event_type_stress:
                event_stress = event_type_stress[scenario.event_type.value]
                frequency_multiplier *= event_stress.get('frequency_multiplier', 1.0)
                severity_multiplier *= event_stress.get('severity_multiplier', 1.0)
            
            stressed_scenario = ScenarioAnalysis(
                scenario_id=f"stressed_{scenario.scenario_id}",
                scenario_name=f"Stressed {scenario.scenario_name}",
                event_type=scenario.event_type,
                business_line=scenario.business_line,
                frequency_estimate=scenario.frequency_estimate * frequency_multiplier,
                severity_estimate=scenario.severity_estimate * severity_multiplier,
                severity_volatility=scenario.severity_volatility * volatility_multiplier,
                confidence_level=scenario.confidence_level,
                expert_judgment=scenario.expert_judgment
            )
            
            stressed_scenarios.append(stressed_scenario)
        
        return stressed_scenarios
    
    def generate_operational_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive operational risk report."""
        self.logger.info("Generating operational risk report")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'report_type': 'Operational Risk Analysis',
                'version': '1.0'
            },
            'executive_summary': {},
            'loss_data_analysis': {},
            'model_results': {},
            'kri_analysis': {},
            'capital_calculations': {},
            'stress_test_results': {},
            'recommendations': []
        }
        
        # Executive summary
        if self.loss_events:
            total_losses = sum(loss.net_loss for loss in self.loss_events)
            report['executive_summary'] = {
                'total_loss_events': len(self.loss_events),
                'total_net_losses': total_losses,
                'average_loss': total_losses / len(self.loss_events),
                'largest_loss': max(loss.net_loss for loss in self.loss_events),
                'assessment_period': f"{min(loss.event_date for loss in self.loss_events)} to {max(loss.event_date for loss in self.loss_events)}"
            }
        
        # Loss data analysis
        report['loss_data_analysis'] = self.analyze_loss_data()
        
        # Model results
        report['model_results'] = {
            'frequency_models': len(self.frequency_models),
            'severity_models': len(self.severity_models),
            'var_calculations': self.calculate_operational_var()
        }
        
        # KRI analysis
        report['kri_analysis'] = self.analyze_kri_performance()
        
        # Capital calculations
        report['capital_calculations'] = self.capital_calculations if self.capital_calculations else self.calculate_regulatory_capital()
        
        # Stress test results
        if self.stress_test_results:
            report['stress_test_results'] = self.stress_test_results
        
        # Recommendations
        report['recommendations'] = self._generate_operational_risk_recommendations()
        
        self.logger.info("Operational risk report generated")
        return report
    
    def _generate_operational_risk_recommendations(self) -> List[Dict[str, Any]]:
        """Generate operational risk management recommendations."""
        recommendations = []
        
        # Loss data recommendations
        if len(self.loss_events) < 100:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'HIGH',
                'recommendation': 'Enhance loss data collection to improve model reliability',
                'timeline': '3-6 months'
            })
        
        # Model development recommendations
        if len(self.frequency_models) < 3:
            recommendations.append({
                'category': 'Model Development',
                'priority': 'MEDIUM',
                'recommendation': 'Develop frequency models for all major event types and business lines',
                'timeline': '6-12 months'
            })
        
        if len(self.severity_models) < 3:
            recommendations.append({
                'category': 'Model Development',
                'priority': 'MEDIUM',
                'recommendation': 'Develop severity models for all major event types and business lines',
                'timeline': '6-12 months'
            })
        
        # KRI recommendations
        high_risk_kris = [kri for kri in self.kris if kri.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk_kris:
            recommendations.append({
                'category': 'Risk Monitoring',
                'priority': 'HIGH',
                'recommendation': f'Address {len(high_risk_kris)} high-risk KRIs requiring immediate attention',
                'timeline': '1-3 months'
            })
        
        # Scenario analysis recommendations
        if len(self.scenarios) < 5:
            recommendations.append({
                'category': 'Scenario Analysis',
                'priority': 'MEDIUM',
                'recommendation': 'Expand scenario analysis coverage across all event types and business lines',
                'timeline': '6-12 months'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Model Validation',
                'priority': 'MEDIUM',
                'recommendation': 'Implement regular model validation and backtesting procedures',
                'timeline': '3-6 months'
            },
            {
                'category': 'Risk Culture',
                'priority': 'MEDIUM',
                'recommendation': 'Enhance operational risk awareness and reporting culture',
                'timeline': '6-12 months'
            }
        ])
        
        return recommendations


class OperationalRiskFactors:
    """
    Container class for operational risk factors and indicators.
    
    Manages collections of operational risk factors including:
    - Key Risk Indicators (KRIs)
    - Loss event data
    - Risk scenarios
    - Business environment factors
    """
    
    def __init__(self):
        """Initialize operational risk factors container."""
        self.kris: List[KeyRiskIndicator] = []
        self.loss_events: List[OperationalLoss] = []
        self.scenarios: List[ScenarioAnalysis] = []
        self.business_factors: Dict[str, Any] = {}
        
    def add_kri(self, kri: KeyRiskIndicator):
        """Add a Key Risk Indicator."""
        self.kris.append(kri)
        
    def add_loss_event(self, loss: OperationalLoss):
        """Add a loss event."""
        self.loss_events.append(loss)
        
    def add_scenario(self, scenario: ScenarioAnalysis):
        """Add a risk scenario."""
        self.scenarios.append(scenario)
        
    def get_high_risk_kris(self) -> List[KeyRiskIndicator]:
        """Get KRIs with high or critical risk levels."""
        return [kri for kri in self.kris if kri.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
    def get_losses_by_event_type(self, event_type: EventType) -> List[OperationalLoss]:
        """Get losses filtered by event type."""
        return [loss for loss in self.loss_events if loss.event_type == event_type]
        
    def get_total_losses(self) -> float:
        """Calculate total operational losses."""
        return sum(loss.loss_amount for loss in self.loss_events)
        
    def get_factors_summary(self) -> Dict[str, Any]:
        """Get summary of operational risk factors."""
        return {
            'total_kris': len(self.kris),
            'high_risk_kris': len(self.get_high_risk_kris()),
            'total_loss_events': len(self.loss_events),
            'total_losses': self.get_total_losses(),
            'scenarios_count': len(self.scenarios),
            'business_factors_count': len(self.business_factors)
        }