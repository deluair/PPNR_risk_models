"""
Bank Metrics Processor for PPNR Risk Models

Specialized processing for bank-specific metrics including:
- Balance sheet data processing
- Income statement analysis
- Asset quality metrics
- Capital and liquidity ratios
- Business line performance
- Risk-weighted assets calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class BankMetricsProcessor:
    """
    Comprehensive bank metrics processor for PPNR risk modeling.
    
    Features:
    - Balance sheet and income statement processing
    - Asset quality and credit metrics
    - Capital and liquidity ratio calculations
    - Business line performance analysis
    - Regulatory metrics computation
    - Peer benchmarking capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bank metrics processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bank_config = config.get('bank_metrics', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.BankMetricsProcessor")
        
        # Processing parameters
        self.reporting_frequency = self.bank_config.get('reporting_frequency', 'quarterly')
        self.peer_group = self.bank_config.get('peer_group', 'large_banks')
        self.regulatory_framework = self.bank_config.get('regulatory_framework', 'basel_iii')
        
        # Processed data storage
        self.processed_metrics = {}
        self.calculated_ratios = {}
        self.business_line_metrics = {}
        
        # Define metric categories
        self.metric_categories = {
            'balance_sheet': [
                'total_assets', 'total_loans', 'total_deposits', 'securities',
                'cash_and_equivalents', 'total_liabilities', 'shareholders_equity'
            ],
            'income_statement': [
                'net_interest_income', 'noninterest_income', 'noninterest_expense',
                'provision_for_credit_losses', 'net_income', 'trading_revenue'
            ],
            'asset_quality': [
                'npl_ratio', 'charge_off_rate', 'recovery_rate', 'allowance_ratio',
                'criticized_assets', 'past_due_loans'
            ],
            'capital': [
                'tier1_capital', 'tier2_capital', 'total_capital', 'tangible_common_equity',
                'risk_weighted_assets', 'leverage_ratio'
            ],
            'liquidity': [
                'lcr', 'nsfr', 'deposits_to_loans', 'liquid_assets',
                'funding_cost', 'deposit_beta'
            ],
            'profitability': [
                'roe', 'roa', 'nim', 'efficiency_ratio', 'fee_income_ratio',
                'cost_of_funds', 'yield_on_assets'
            ]
        }
    
    def process_bank_metrics(self, bank_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw bank metrics into analysis-ready format.
        
        Args:
            bank_data: Raw bank metrics DataFrame
            
        Returns:
            Processed bank metrics
        """
        self.logger.info("Processing bank metrics...")
        
        # Ensure data is properly formatted
        processed_data = self._format_bank_data(bank_data)
        
        # Calculate derived metrics
        processed_data = self._calculate_derived_metrics(processed_data)
        
        # Calculate financial ratios
        processed_data = self._calculate_financial_ratios(processed_data)
        
        # Add growth rates and trends
        processed_data = self._calculate_growth_rates(processed_data)
        
        # Calculate regulatory metrics
        processed_data = self._calculate_regulatory_metrics(processed_data)
        
        # Store processed data
        self.processed_metrics = processed_data
        
        self.logger.info(f"Bank metrics processing completed: {processed_data.shape}")
        return processed_data
    
    def _format_bank_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format and validate bank data structure."""
        formatted_data = data.copy()
        
        # Ensure date column is datetime
        if 'date' in formatted_data.columns:
            formatted_data['date'] = pd.to_datetime(formatted_data['date'])
            formatted_data = formatted_data.sort_values('date')
            formatted_data = formatted_data.set_index('date')
        
        # Handle missing values
        formatted_data = self._handle_missing_bank_data(formatted_data)
        
        # Ensure quarterly frequency for bank data
        if self.reporting_frequency == 'quarterly':
            formatted_data = self._ensure_quarterly_frequency(formatted_data)
        
        # Convert to appropriate units (millions)
        balance_sheet_items = self.metric_categories['balance_sheet'] + self.metric_categories['income_statement']
        for item in balance_sheet_items:
            if item in formatted_data.columns:
                # Assume data is in thousands, convert to millions
                if formatted_data[item].max() > 1000000:  # Likely in thousands
                    formatted_data[item] = formatted_data[item] / 1000
        
        return formatted_data
    
    def _handle_missing_bank_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in bank data."""
        # Forward fill for most bank metrics (quarterly reporting)
        data = data.fillna(method='ffill', limit=1)
        
        # Interpolate for smooth series like assets and deposits
        smooth_metrics = ['total_assets', 'total_loans', 'total_deposits']
        for metric in smooth_metrics:
            if metric in data.columns:
                data[metric] = data[metric].interpolate(method='linear')
        
        # Log missing data
        missing_summary = data.isnull().sum()
        if missing_summary.sum() > 0:
            self.logger.warning(f"Missing bank data summary:\n{missing_summary[missing_summary > 0]}")
        
        return data
    
    def _ensure_quarterly_frequency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has consistent quarterly frequency."""
        # Create quarterly date range
        start_date = data.index.min()
        end_date = data.index.max()
        quarterly_index = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        # Reindex to quarterly frequency
        data = data.reindex(quarterly_index)
        
        return data
    
    def _calculate_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived bank metrics."""
        result_data = data.copy()
        
        # Net interest margin components
        if 'net_interest_income' in data.columns and 'total_assets' in data.columns:
            result_data['nim'] = (data['net_interest_income'] * 4) / data['total_assets'] * 100  # Annualized
        
        # Efficiency ratio
        if 'noninterest_expense' in data.columns and 'net_interest_income' in data.columns and 'noninterest_income' in data.columns:
            result_data['efficiency_ratio'] = (
                data['noninterest_expense'] / (data['net_interest_income'] + data['noninterest_income'])
            ) * 100
        
        # Pre-provision net revenue (PPNR)
        if 'net_interest_income' in data.columns and 'noninterest_income' in data.columns and 'noninterest_expense' in data.columns:
            result_data['ppnr'] = (
                data['net_interest_income'] + data['noninterest_income'] - data['noninterest_expense']
            )
        
        # Net charge-off rate
        if 'charge_offs' in data.columns and 'total_loans' in data.columns:
            result_data['charge_off_rate'] = (data['charge_offs'] * 4) / data['total_loans'] * 100  # Annualized
        
        # Provision rate
        if 'provision_for_credit_losses' in data.columns and 'total_loans' in data.columns:
            result_data['provision_rate'] = (data['provision_for_credit_losses'] * 4) / data['total_loans'] * 100
        
        # Loan loss allowance ratio
        if 'allowance_for_credit_losses' in data.columns and 'total_loans' in data.columns:
            result_data['allowance_ratio'] = data['allowance_for_credit_losses'] / data['total_loans'] * 100
        
        # Tangible book value per share
        if 'shareholders_equity' in data.columns and 'goodwill' in data.columns and 'shares_outstanding' in data.columns:
            result_data['tangible_book_value_per_share'] = (
                (data['shareholders_equity'] - data['goodwill']) / data['shares_outstanding']
            )
        
        return result_data
    
    def _calculate_financial_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate key financial ratios."""
        result_data = data.copy()
        
        # Profitability ratios
        if 'net_income' in data.columns and 'shareholders_equity' in data.columns:
            result_data['roe'] = (data['net_income'] * 4) / data['shareholders_equity'] * 100  # Annualized
        
        if 'net_income' in data.columns and 'total_assets' in data.columns:
            result_data['roa'] = (data['net_income'] * 4) / data['total_assets'] * 100  # Annualized
        
        # Asset quality ratios
        if 'nonperforming_loans' in data.columns and 'total_loans' in data.columns:
            result_data['npl_ratio'] = data['nonperforming_loans'] / data['total_loans'] * 100
        
        # Capital ratios
        if 'tier1_capital' in data.columns and 'risk_weighted_assets' in data.columns:
            result_data['tier1_capital_ratio'] = data['tier1_capital'] / data['risk_weighted_assets'] * 100
        
        if 'total_capital' in data.columns and 'risk_weighted_assets' in data.columns:
            result_data['total_capital_ratio'] = data['total_capital'] / data['risk_weighted_assets'] * 100
        
        if 'tier1_capital' in data.columns and 'total_assets' in data.columns:
            result_data['leverage_ratio'] = data['tier1_capital'] / data['total_assets'] * 100
        
        # Liquidity ratios
        if 'total_deposits' in data.columns and 'total_loans' in data.columns:
            result_data['deposits_to_loans'] = data['total_deposits'] / data['total_loans'] * 100
        
        if 'liquid_assets' in data.columns and 'total_assets' in data.columns:
            result_data['liquid_assets_ratio'] = data['liquid_assets'] / data['total_assets'] * 100
        
        # Fee income ratio
        if 'noninterest_income' in data.columns and 'total_revenue' in data.columns:
            result_data['fee_income_ratio'] = data['noninterest_income'] / data['total_revenue'] * 100
        elif 'noninterest_income' in data.columns and 'net_interest_income' in data.columns:
            total_revenue = data['net_interest_income'] + data['noninterest_income']
            result_data['fee_income_ratio'] = data['noninterest_income'] / total_revenue * 100
        
        return result_data
    
    def _calculate_growth_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate growth rates for key metrics."""
        result_data = data.copy()
        
        # Year-over-year growth rates
        growth_metrics = [
            'total_assets', 'total_loans', 'total_deposits', 'net_interest_income',
            'noninterest_income', 'ppnr', 'net_income'
        ]
        
        for metric in growth_metrics:
            if metric in data.columns:
                result_data[f'{metric}_yoy_growth'] = data[metric].pct_change(periods=4) * 100  # 4 quarters
        
        # Quarter-over-quarter growth rates (annualized)
        for metric in growth_metrics:
            if metric in data.columns:
                result_data[f'{metric}_qoq_growth_ann'] = (
                    (data[metric] / data[metric].shift(1)) ** 4 - 1
                ) * 100
        
        # Rolling averages for smoothing
        for metric in growth_metrics:
            if metric in data.columns:
                result_data[f'{metric}_4q_avg'] = data[metric].rolling(4).mean()
        
        return result_data
    
    def _calculate_regulatory_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regulatory metrics."""
        result_data = data.copy()
        
        # Basel III metrics
        if self.regulatory_framework == 'basel_iii':
            # Common Equity Tier 1 ratio
            if 'cet1_capital' in data.columns and 'risk_weighted_assets' in data.columns:
                result_data['cet1_ratio'] = data['cet1_capital'] / data['risk_weighted_assets'] * 100
            
            # Supplementary leverage ratio
            if 'tier1_capital' in data.columns and 'total_leverage_exposure' in data.columns:
                result_data['supplementary_leverage_ratio'] = (
                    data['tier1_capital'] / data['total_leverage_exposure'] * 100
                )
            
            # Liquidity Coverage Ratio
            if 'high_quality_liquid_assets' in data.columns and 'net_cash_outflows' in data.columns:
                result_data['lcr'] = data['high_quality_liquid_assets'] / data['net_cash_outflows'] * 100
            
            # Net Stable Funding Ratio
            if 'available_stable_funding' in data.columns and 'required_stable_funding' in data.columns:
                result_data['nsfr'] = data['available_stable_funding'] / data['required_stable_funding'] * 100
        
        # CCAR/DFAST metrics
        if 'stressed_losses' in data.columns and 'pre_provision_net_revenue' in data.columns:
            result_data['stress_loss_rate'] = data['stressed_losses'] / data['total_loans'] * 100
            result_data['ppnr_coverage_ratio'] = data['pre_provision_net_revenue'] / data['stressed_losses']
        
        return result_data
    
    def analyze_business_lines(self, business_line_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze business line performance.
        
        Args:
            business_line_data: Business line specific data
            
        Returns:
            Processed business line metrics
        """
        self.logger.info("Analyzing business line performance...")
        
        processed_bl_data = business_line_data.copy()
        
        # Ensure proper formatting
        if 'date' in processed_bl_data.columns and 'business_line' in processed_bl_data.columns:
            processed_bl_data['date'] = pd.to_datetime(processed_bl_data['date'])
            processed_bl_data = processed_bl_data.set_index(['date', 'business_line'])
        
        # Calculate business line specific metrics
        processed_bl_data = self._calculate_business_line_metrics(processed_bl_data)
        
        # Calculate business line contributions
        processed_bl_data = self._calculate_business_line_contributions(processed_bl_data)
        
        # Store business line metrics
        self.business_line_metrics = processed_bl_data
        
        return processed_bl_data
    
    def _calculate_business_line_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate business line specific metrics."""
        result_data = data.copy()
        
        # Revenue per employee
        if 'revenue' in data.columns and 'employees' in data.columns:
            result_data['revenue_per_employee'] = data['revenue'] / data['employees']
        
        # Cost-to-income ratio
        if 'expenses' in data.columns and 'revenue' in data.columns:
            result_data['cost_to_income'] = data['expenses'] / data['revenue'] * 100
        
        # Return on allocated capital
        if 'net_income' in data.columns and 'allocated_capital' in data.columns:
            result_data['roac'] = (data['net_income'] * 4) / data['allocated_capital'] * 100
        
        # Risk-adjusted return
        if 'net_income' in data.columns and 'risk_weighted_assets' in data.columns:
            result_data['risk_adjusted_return'] = (data['net_income'] * 4) / data['risk_weighted_assets'] * 100
        
        return result_data
    
    def _calculate_business_line_contributions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate business line contributions to total bank metrics."""
        result_data = data.copy()
        
        # Calculate total bank metrics for each date
        if isinstance(data.index, pd.MultiIndex):
            date_totals = data.groupby(level=0).sum()
            
            # Calculate contributions as percentages
            for metric in ['revenue', 'net_income', 'assets', 'risk_weighted_assets']:
                if metric in data.columns:
                    # Calculate contribution percentage
                    total_metric = date_totals[metric]
                    
                    # Broadcast total back to business line level
                    contribution_col = f'{metric}_contribution_pct'
                    for date in data.index.get_level_values(0).unique():
                        mask = data.index.get_level_values(0) == date
                        if total_metric.loc[date] != 0:
                            result_data.loc[mask, contribution_col] = (
                                data.loc[mask, metric] / total_metric.loc[date] * 100
                            )
        
        return result_data
    
    def calculate_peer_benchmarks(self, peer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate peer benchmarks for key metrics.
        
        Args:
            peer_data: Peer bank data
            
        Returns:
            Peer benchmark statistics
        """
        self.logger.info("Calculating peer benchmarks...")
        
        # Key metrics for benchmarking
        benchmark_metrics = [
            'roe', 'roa', 'nim', 'efficiency_ratio', 'tier1_capital_ratio',
            'npl_ratio', 'charge_off_rate', 'fee_income_ratio'
        ]
        
        # Calculate peer statistics
        peer_stats = {}
        
        for metric in benchmark_metrics:
            if metric in peer_data.columns:
                peer_stats[f'{metric}_peer_median'] = peer_data[metric].median()
                peer_stats[f'{metric}_peer_mean'] = peer_data[metric].mean()
                peer_stats[f'{metric}_peer_25th'] = peer_data[metric].quantile(0.25)
                peer_stats[f'{metric}_peer_75th'] = peer_data[metric].quantile(0.75)
                peer_stats[f'{metric}_peer_std'] = peer_data[metric].std()
        
        # Convert to DataFrame
        peer_benchmarks = pd.DataFrame([peer_stats])
        
        return peer_benchmarks
    
    def identify_outliers(self, data: pd.DataFrame,
                         method: str = 'iqr',
                         threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify outliers in bank metrics.
        
        Args:
            data: Bank metrics data
            method: Outlier detection method
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        result = data.copy()
        
        # Key metrics to check for outliers
        outlier_metrics = [
            'roe', 'roa', 'nim', 'efficiency_ratio', 'npl_ratio',
            'charge_off_rate', 'provision_rate'
        ]
        
        for metric in outlier_metrics:
            if metric in data.columns:
                if method == 'iqr':
                    Q1 = data[metric].quantile(0.25)
                    Q3 = data[metric].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    result[f'{metric}_outlier'] = (
                        (data[metric] < lower_bound) | (data[metric] > upper_bound)
                    ).astype(int)
                
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[metric].dropna()))
                    outlier_mask = z_scores > threshold
                    
                    # Align with original data
                    outlier_series = pd.Series(0, index=data.index)
                    outlier_series.loc[data[metric].dropna().index[outlier_mask]] = 1
                    result[f'{metric}_outlier'] = outlier_series
        
        return result
    
    def calculate_stress_impact(self, baseline_data: pd.DataFrame,
                              stress_scenarios: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate impact of stress scenarios on bank metrics.
        
        Args:
            baseline_data: Baseline bank metrics
            stress_scenarios: Stress scenario parameters
            
        Returns:
            Stressed bank metrics
        """
        self.logger.info("Calculating stress impact on bank metrics...")
        
        stressed_data = baseline_data.copy()
        
        # Apply stress to key metrics
        for scenario_name, stress_factor in stress_scenarios.items():
            if scenario_name == 'credit_losses':
                # Increase provision and charge-offs
                if 'provision_for_credit_losses' in stressed_data.columns:
                    stressed_data['provision_for_credit_losses_stressed'] = (
                        stressed_data['provision_for_credit_losses'] * (1 + stress_factor)
                    )
                
                if 'charge_offs' in stressed_data.columns:
                    stressed_data['charge_offs_stressed'] = (
                        stressed_data['charge_offs'] * (1 + stress_factor)
                    )
            
            elif scenario_name == 'net_interest_income':
                # Apply stress to NII
                if 'net_interest_income' in stressed_data.columns:
                    stressed_data['net_interest_income_stressed'] = (
                        stressed_data['net_interest_income'] * (1 + stress_factor)
                    )
            
            elif scenario_name == 'trading_revenue':
                # Apply stress to trading revenue
                if 'trading_revenue' in stressed_data.columns:
                    stressed_data['trading_revenue_stressed'] = (
                        stressed_data['trading_revenue'] * (1 + stress_factor)
                    )
        
        # Recalculate derived metrics under stress
        stressed_data = self._recalculate_stressed_metrics(stressed_data)
        
        return stressed_data
    
    def _recalculate_stressed_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Recalculate key metrics under stress scenarios."""
        result_data = data.copy()
        
        # Recalculate PPNR under stress
        nii_col = 'net_interest_income_stressed' if 'net_interest_income_stressed' in data.columns else 'net_interest_income'
        trading_col = 'trading_revenue_stressed' if 'trading_revenue_stressed' in data.columns else 'trading_revenue'
        
        if nii_col in data.columns and 'noninterest_income' in data.columns and 'noninterest_expense' in data.columns:
            noninterest_income_adj = data['noninterest_income'].copy()
            if trading_col in data.columns:
                # Adjust noninterest income for stressed trading revenue
                trading_diff = data[trading_col] - data.get('trading_revenue', 0)
                noninterest_income_adj += trading_diff
            
            result_data['ppnr_stressed'] = (
                data[nii_col] + noninterest_income_adj - data['noninterest_expense']
            )
        
        # Recalculate net income under stress
        provision_col = 'provision_for_credit_losses_stressed' if 'provision_for_credit_losses_stressed' in data.columns else 'provision_for_credit_losses'
        
        if 'ppnr_stressed' in result_data.columns and provision_col in data.columns:
            result_data['net_income_stressed'] = result_data['ppnr_stressed'] - data[provision_col]
        
        # Recalculate profitability ratios under stress
        if 'net_income_stressed' in result_data.columns:
            if 'shareholders_equity' in data.columns:
                result_data['roe_stressed'] = (result_data['net_income_stressed'] * 4) / data['shareholders_equity'] * 100
            
            if 'total_assets' in data.columns:
                result_data['roa_stressed'] = (result_data['net_income_stressed'] * 4) / data['total_assets'] * 100
        
        return result_data
    
    def generate_performance_report(self, processed_data: pd.DataFrame,
                                  peer_benchmarks: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            processed_data: Processed bank metrics
            peer_benchmarks: Peer benchmark data
            
        Returns:
            Performance report dictionary
        """
        self.logger.info("Generating performance report...")
        
        # Get latest metrics
        latest_data = processed_data.iloc[-1]
        
        # Calculate trends (last 4 quarters)
        trend_data = processed_data.tail(4)
        
        report = {
            'reporting_date': str(processed_data.index[-1]),
            'key_metrics': {
                'profitability': {
                    'roe': float(latest_data.get('roe', 0)),
                    'roa': float(latest_data.get('roa', 0)),
                    'nim': float(latest_data.get('nim', 0)),
                    'efficiency_ratio': float(latest_data.get('efficiency_ratio', 0))
                },
                'asset_quality': {
                    'npl_ratio': float(latest_data.get('npl_ratio', 0)),
                    'charge_off_rate': float(latest_data.get('charge_off_rate', 0)),
                    'provision_rate': float(latest_data.get('provision_rate', 0)),
                    'allowance_ratio': float(latest_data.get('allowance_ratio', 0))
                },
                'capital': {
                    'tier1_capital_ratio': float(latest_data.get('tier1_capital_ratio', 0)),
                    'total_capital_ratio': float(latest_data.get('total_capital_ratio', 0)),
                    'leverage_ratio': float(latest_data.get('leverage_ratio', 0))
                },
                'liquidity': {
                    'lcr': float(latest_data.get('lcr', 0)),
                    'nsfr': float(latest_data.get('nsfr', 0)),
                    'deposits_to_loans': float(latest_data.get('deposits_to_loans', 0))
                }
            },
            'trends': {
                'ppnr_growth': float(trend_data['ppnr'].pct_change().mean() * 100) if 'ppnr' in trend_data.columns else 0,
                'asset_growth': float(trend_data['total_assets'].pct_change().mean() * 100) if 'total_assets' in trend_data.columns else 0,
                'loan_growth': float(trend_data['total_loans'].pct_change().mean() * 100) if 'total_loans' in trend_data.columns else 0
            }
        }
        
        # Add peer comparisons if available
        if peer_benchmarks is not None and not peer_benchmarks.empty:
            peer_data = peer_benchmarks.iloc[0]
            report['peer_comparison'] = {}
            
            for metric in ['roe', 'roa', 'nim', 'efficiency_ratio']:
                if metric in latest_data.index and f'{metric}_peer_median' in peer_data.index:
                    report['peer_comparison'][metric] = {
                        'bank_value': float(latest_data[metric]),
                        'peer_median': float(peer_data[f'{metric}_peer_median']),
                        'percentile_rank': self._calculate_percentile_rank(
                            latest_data[metric], peer_data, metric
                        )
                    }
        
        return report
    
    def _calculate_percentile_rank(self, value: float, peer_data: pd.Series, metric: str) -> float:
        """Calculate percentile rank vs peers."""
        peer_mean = peer_data.get(f'{metric}_peer_mean', value)
        peer_std = peer_data.get(f'{metric}_peer_std', 1)
        
        # Simple z-score based percentile approximation
        z_score = (value - peer_mean) / peer_std
        percentile = stats.norm.cdf(z_score) * 100
        
        return float(np.clip(percentile, 0, 100))
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processed bank metrics."""
        summary = {
            'processed_metrics_shape': self.processed_metrics.shape if hasattr(self, 'processed_metrics') else None,
            'business_line_metrics_shape': self.business_line_metrics.shape if hasattr(self, 'business_line_metrics') else None,
            'processing_parameters': {
                'reporting_frequency': self.reporting_frequency,
                'peer_group': self.peer_group,
                'regulatory_framework': self.regulatory_framework
            },
            'metric_categories': self.metric_categories
        }
        
        if hasattr(self, 'processed_metrics') and not self.processed_metrics.empty:
            summary['processed_columns'] = list(self.processed_metrics.columns)
            summary['date_range'] = {
                'start': str(self.processed_metrics.index.min()),
                'end': str(self.processed_metrics.index.max())
            }
        
        return summary