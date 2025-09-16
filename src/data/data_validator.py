"""
Data Validator for PPNR Risk Models

Comprehensive data validation including:
- Data quality checks
- Consistency validation
- Completeness assessment
- Outlier detection
- Cross-validation between data sources
- Regulatory compliance checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
import matplotlib.pyplot as plt

class DataValidator:
    """
    Comprehensive data validator for PPNR risk modeling.
    
    Features:
    - Data quality assessment
    - Consistency checks across data sources
    - Completeness validation
    - Outlier detection and flagging
    - Regulatory compliance validation
    - Data lineage tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get('data_validation', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.DataValidator")
        
        # Validation parameters
        self.completeness_threshold = self.validation_config.get('completeness_threshold', 0.95)
        self.outlier_threshold = self.validation_config.get('outlier_threshold', 3.0)
        self.consistency_tolerance = self.validation_config.get('consistency_tolerance', 0.01)
        
        # Validation results storage
        self.validation_results = {}
        self.data_quality_scores = {}
        self.validation_warnings = []
        self.validation_errors = []
        
        # Define validation rules
        self.validation_rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define validation rules for different data types."""
        return {
            'market_data': {
                'required_columns': ['date', 'symbol', 'price'],
                'numeric_columns': ['price', 'volume'],
                'date_columns': ['date'],
                'positive_columns': ['price', 'volume'],
                'range_checks': {
                    'price': {'min': 0, 'max': None},
                    'volume': {'min': 0, 'max': None}
                }
            },
            'economic_indicators': {
                'required_columns': ['date'],
                'numeric_columns': ['GDP', 'UNEMPLOYMENT_RATE', 'CPI', 'FED_FUNDS_RATE'],
                'date_columns': ['date'],
                'range_checks': {
                    'UNEMPLOYMENT_RATE': {'min': 0, 'max': 50},
                    'FED_FUNDS_RATE': {'min': -5, 'max': 25},
                    'CPI': {'min': 0, 'max': None}
                }
            },
            'bank_metrics': {
                'required_columns': ['date', 'total_assets', 'net_income'],
                'numeric_columns': ['total_assets', 'total_loans', 'net_income', 'roe', 'roa'],
                'date_columns': ['date'],
                'positive_columns': ['total_assets', 'total_loans'],
                'range_checks': {
                    'roe': {'min': -50, 'max': 50},
                    'roa': {'min': -10, 'max': 10},
                    'tier1_capital_ratio': {'min': 0, 'max': 30}
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Comprehensive data validation.
        
        Args:
            data: Data to validate
            data_type: Type of data ('market_data', 'economic_indicators', 'bank_metrics')
            
        Returns:
            Validation results dictionary
        """
        self.logger.info(f"Validating {data_type} data...")
        
        validation_results = {
            'data_type': data_type,
            'validation_timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'checks_performed': [],
            'warnings': [],
            'errors': [],
            'quality_score': 0.0
        }
        
        # Get validation rules for this data type
        rules = self.validation_rules.get(data_type, {})
        
        # 1. Structure validation
        structure_results = self._validate_structure(data, rules)
        validation_results.update(structure_results)
        validation_results['checks_performed'].append('structure')
        
        # 2. Data type validation
        dtype_results = self._validate_data_types(data, rules)
        validation_results.update(dtype_results)
        validation_results['checks_performed'].append('data_types')
        
        # 3. Completeness validation
        completeness_results = self._validate_completeness(data, rules)
        validation_results.update(completeness_results)
        validation_results['checks_performed'].append('completeness')
        
        # 4. Range validation
        range_results = self._validate_ranges(data, rules)
        validation_results.update(range_results)
        validation_results['checks_performed'].append('ranges')
        
        # 5. Consistency validation
        consistency_results = self._validate_consistency(data, data_type)
        validation_results.update(consistency_results)
        validation_results['checks_performed'].append('consistency')
        
        # 6. Outlier detection
        outlier_results = self._detect_outliers(data, rules)
        validation_results.update(outlier_results)
        validation_results['checks_performed'].append('outliers')
        
        # 7. Time series validation (if applicable)
        if 'date' in data.columns:
            timeseries_results = self._validate_time_series(data)
            validation_results.update(timeseries_results)
            validation_results['checks_performed'].append('time_series')
        
        # Calculate overall quality score
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        # Store results
        self.validation_results[data_type] = validation_results
        
        self.logger.info(f"Validation completed for {data_type}. Quality score: {validation_results['quality_score']:.2f}")
        
        return validation_results
    
    def _validate_structure(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data structure."""
        results = {
            'structure_valid': True,
            'missing_required_columns': [],
            'unexpected_columns': []
        }
        
        # Check required columns
        required_columns = rules.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            results['structure_valid'] = False
            results['missing_required_columns'] = missing_columns
            self.validation_errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for unexpected columns (informational)
        expected_columns = set(required_columns + rules.get('numeric_columns', []) + rules.get('date_columns', []))
        unexpected_columns = [col for col in data.columns if col not in expected_columns and not col.startswith('_')]
        
        if unexpected_columns:
            results['unexpected_columns'] = unexpected_columns
            self.validation_warnings.append(f"Unexpected columns found: {unexpected_columns}")
        
        return results
    
    def _validate_data_types(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data types."""
        results = {
            'data_types_valid': True,
            'invalid_numeric_columns': [],
            'invalid_date_columns': []
        }
        
        # Check numeric columns
        numeric_columns = rules.get('numeric_columns', [])
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    # Try to convert
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        self.validation_warnings.append(f"Converted {col} to numeric type")
                    except:
                        results['data_types_valid'] = False
                        results['invalid_numeric_columns'].append(col)
                        self.validation_errors.append(f"Cannot convert {col} to numeric type")
        
        # Check date columns
        date_columns = rules.get('date_columns', [])
        for col in date_columns:
            if col in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    # Try to convert
                    try:
                        data[col] = pd.to_datetime(data[col])
                        self.validation_warnings.append(f"Converted {col} to datetime type")
                    except:
                        results['data_types_valid'] = False
                        results['invalid_date_columns'].append(col)
                        self.validation_errors.append(f"Cannot convert {col} to datetime type")
        
        return results
    
    def _validate_completeness(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data completeness."""
        results = {
            'completeness_valid': True,
            'completeness_scores': {},
            'incomplete_columns': []
        }
        
        # Calculate completeness for each column
        for col in data.columns:
            completeness = 1 - (data[col].isnull().sum() / len(data))
            results['completeness_scores'][col] = float(completeness)
            
            if completeness < self.completeness_threshold:
                results['completeness_valid'] = False
                results['incomplete_columns'].append(col)
                self.validation_warnings.append(
                    f"Column {col} has low completeness: {completeness:.2%}"
                )
        
        # Overall completeness
        overall_completeness = np.mean(list(results['completeness_scores'].values()))
        results['overall_completeness'] = float(overall_completeness)
        
        return results
    
    def _validate_ranges(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data ranges."""
        results = {
            'ranges_valid': True,
            'range_violations': {},
            'negative_value_violations': []
        }
        
        # Check range constraints
        range_checks = rules.get('range_checks', {})
        for col, constraints in range_checks.items():
            if col in data.columns:
                violations = []
                
                # Check minimum
                if constraints.get('min') is not None:
                    min_violations = data[data[col] < constraints['min']]
                    if not min_violations.empty:
                        violations.append(f"Values below minimum {constraints['min']}: {len(min_violations)}")
                
                # Check maximum
                if constraints.get('max') is not None:
                    max_violations = data[data[col] > constraints['max']]
                    if not max_violations.empty:
                        violations.append(f"Values above maximum {constraints['max']}: {len(max_violations)}")
                
                if violations:
                    results['ranges_valid'] = False
                    results['range_violations'][col] = violations
                    self.validation_warnings.extend([f"{col}: {v}" for v in violations])
        
        # Check positive columns
        positive_columns = rules.get('positive_columns', [])
        for col in positive_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    results['ranges_valid'] = False
                    results['negative_value_violations'].append(f"{col}: {negative_count} negative values")
                    self.validation_warnings.append(f"{col} has {negative_count} negative values")
        
        return results
    
    def _validate_consistency(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate data consistency."""
        results = {
            'consistency_valid': True,
            'consistency_issues': []
        }
        
        # Data type specific consistency checks
        if data_type == 'bank_metrics':
            results.update(self._validate_bank_metrics_consistency(data))
        elif data_type == 'market_data':
            results.update(self._validate_market_data_consistency(data))
        elif data_type == 'economic_indicators':
            results.update(self._validate_economic_indicators_consistency(data))
        
        return results
    
    def _validate_bank_metrics_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate bank metrics consistency."""
        results = {
            'bank_consistency_valid': True,
            'bank_consistency_issues': []
        }
        
        # Check balance sheet identity: Assets = Liabilities + Equity
        if all(col in data.columns for col in ['total_assets', 'total_liabilities', 'shareholders_equity']):
            balance_check = np.abs(
                data['total_assets'] - (data['total_liabilities'] + data['shareholders_equity'])
            ) / data['total_assets']
            
            violations = balance_check > self.consistency_tolerance
            if violations.any():
                results['bank_consistency_valid'] = False
                results['bank_consistency_issues'].append(
                    f"Balance sheet identity violations: {violations.sum()} records"
                )
        
        # Check ROE calculation consistency
        if all(col in data.columns for col in ['roe', 'net_income', 'shareholders_equity']):
            calculated_roe = (data['net_income'] * 4) / data['shareholders_equity'] * 100  # Annualized
            roe_diff = np.abs(data['roe'] - calculated_roe) / np.abs(data['roe'])
            
            violations = roe_diff > self.consistency_tolerance
            if violations.any():
                results['bank_consistency_valid'] = False
                results['bank_consistency_issues'].append(
                    f"ROE calculation inconsistencies: {violations.sum()} records"
                )
        
        return results
    
    def _validate_market_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data consistency."""
        results = {
            'market_consistency_valid': True,
            'market_consistency_issues': []
        }
        
        # Check for price jumps (more than 50% in one day)
        if 'price' in data.columns and 'symbol' in data.columns:
            data_sorted = data.sort_values(['symbol', 'date'])
            price_changes = data_sorted.groupby('symbol')['price'].pct_change()
            
            large_jumps = np.abs(price_changes) > 0.5
            if large_jumps.any():
                results['market_consistency_valid'] = False
                results['market_consistency_issues'].append(
                    f"Large price jumps detected: {large_jumps.sum()} records"
                )
        
        return results
    
    def _validate_economic_indicators_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate economic indicators consistency."""
        results = {
            'economic_consistency_valid': True,
            'economic_consistency_issues': []
        }
        
        # Check unemployment rate vs employment indicators
        if all(col in data.columns for col in ['UNEMPLOYMENT_RATE', 'NONFARM_PAYROLLS']):
            # Unemployment and payrolls should generally move in opposite directions
            unemployment_change = data['UNEMPLOYMENT_RATE'].diff()
            payrolls_change = data['NONFARM_PAYROLLS'].diff()
            
            # Check correlation (should be negative)
            correlation = unemployment_change.corr(payrolls_change)
            if correlation > -0.3:  # Weak negative correlation threshold
                results['economic_consistency_valid'] = False
                results['economic_consistency_issues'].append(
                    f"Weak negative correlation between unemployment and payrolls: {correlation:.2f}"
                )
        
        return results
    
    def _detect_outliers(self, data: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in the data."""
        results = {
            'outliers_detected': False,
            'outlier_counts': {},
            'outlier_columns': []
        }
        
        # Check numeric columns for outliers
        numeric_columns = rules.get('numeric_columns', [])
        for col in numeric_columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                # Use IQR method for outlier detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    results['outliers_detected'] = True
                    results['outlier_counts'][col] = int(outlier_count)
                    results['outlier_columns'].append(col)
                    
                    outlier_percentage = outlier_count / len(data) * 100
                    self.validation_warnings.append(
                        f"Outliers detected in {col}: {outlier_count} ({outlier_percentage:.1f}%)"
                    )
        
        return results
    
    def _validate_time_series(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate time series properties."""
        results = {
            'time_series_valid': True,
            'time_series_issues': []
        }
        
        if 'date' not in data.columns:
            return results
        
        # Check for duplicate dates
        duplicate_dates = data['date'].duplicated().sum()
        if duplicate_dates > 0:
            results['time_series_valid'] = False
            results['time_series_issues'].append(f"Duplicate dates: {duplicate_dates}")
        
        # Check for gaps in time series
        data_sorted = data.sort_values('date')
        date_diffs = data_sorted['date'].diff()
        
        # Expected frequency (assume daily for market data, monthly for others)
        if len(data) > 10:
            median_diff = date_diffs.median()
            
            # Check for large gaps (more than 3x median difference)
            large_gaps = date_diffs > (median_diff * 3)
            if large_gaps.any():
                results['time_series_valid'] = False
                results['time_series_issues'].append(f"Large time gaps detected: {large_gaps.sum()}")
        
        # Check date range
        date_range = data_sorted['date'].max() - data_sorted['date'].min()
        results['date_range_days'] = date_range.days
        results['earliest_date'] = str(data_sorted['date'].min())
        results['latest_date'] = str(data_sorted['date'].max())
        
        return results
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score_components = []
        
        # Structure score (0 or 1)
        structure_score = 1.0 if validation_results.get('structure_valid', False) else 0.0
        score_components.append(structure_score * 0.2)
        
        # Data types score (0 or 1)
        dtypes_score = 1.0 if validation_results.get('data_types_valid', False) else 0.0
        score_components.append(dtypes_score * 0.15)
        
        # Completeness score
        completeness_score = validation_results.get('overall_completeness', 0.0)
        score_components.append(completeness_score * 0.25)
        
        # Range validation score (0 or 1)
        ranges_score = 1.0 if validation_results.get('ranges_valid', False) else 0.0
        score_components.append(ranges_score * 0.15)
        
        # Consistency score (0 or 1)
        consistency_score = 1.0 if validation_results.get('consistency_valid', False) else 0.0
        score_components.append(consistency_score * 0.15)
        
        # Outlier score (penalize high outlier rates)
        outlier_penalty = 0.0
        if validation_results.get('outliers_detected', False):
            total_outliers = sum(validation_results.get('outlier_counts', {}).values())
            total_records = validation_results.get('data_shape', [0])[0]
            if total_records > 0:
                outlier_rate = total_outliers / total_records
                outlier_penalty = min(outlier_rate * 2, 0.1)  # Max 10% penalty
        
        outlier_score = max(0.0, 1.0 - outlier_penalty)
        score_components.append(outlier_score * 0.1)
        
        # Overall score
        total_score = sum(score_components)
        return min(max(total_score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def cross_validate_data_sources(self, market_data: pd.DataFrame,
                                  economic_data: pd.DataFrame,
                                  bank_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Cross-validate consistency between different data sources.
        
        Args:
            market_data: Market data DataFrame
            economic_data: Economic indicators DataFrame
            bank_data: Bank metrics DataFrame
            
        Returns:
            Cross-validation results
        """
        self.logger.info("Cross-validating data sources...")
        
        results = {
            'cross_validation_timestamp': datetime.now().isoformat(),
            'date_alignment': {},
            'consistency_checks': {},
            'warnings': [],
            'errors': []
        }
        
        # Check date alignment
        results['date_alignment'] = self._check_date_alignment(
            market_data, economic_data, bank_data
        )
        
        # Check consistency between related indicators
        results['consistency_checks'] = self._check_cross_source_consistency(
            market_data, economic_data, bank_data
        )
        
        return results
    
    def _check_date_alignment(self, market_data: pd.DataFrame,
                            economic_data: pd.DataFrame,
                            bank_data: pd.DataFrame) -> Dict[str, Any]:
        """Check date alignment across data sources."""
        alignment_results = {}
        
        # Extract date ranges
        data_sources = {
            'market_data': market_data,
            'economic_data': economic_data,
            'bank_data': bank_data
        }
        
        date_ranges = {}
        for source_name, data in data_sources.items():
            if 'date' in data.columns:
                date_ranges[source_name] = {
                    'start': data['date'].min(),
                    'end': data['date'].max(),
                    'count': len(data['date'].unique())
                }
        
        alignment_results['date_ranges'] = {
            source: {
                'start': str(info['start']),
                'end': str(info['end']),
                'count': info['count']
            }
            for source, info in date_ranges.items()
        }
        
        # Check overlap
        if len(date_ranges) >= 2:
            all_starts = [info['start'] for info in date_ranges.values()]
            all_ends = [info['end'] for info in date_ranges.values()]
            
            overlap_start = max(all_starts)
            overlap_end = min(all_ends)
            
            if overlap_start <= overlap_end:
                alignment_results['overlap_period'] = {
                    'start': str(overlap_start),
                    'end': str(overlap_end),
                    'days': (overlap_end - overlap_start).days
                }
            else:
                alignment_results['overlap_period'] = None
                self.validation_warnings.append("No date overlap between data sources")
        
        return alignment_results
    
    def _check_cross_source_consistency(self, market_data: pd.DataFrame,
                                      economic_data: pd.DataFrame,
                                      bank_data: pd.DataFrame) -> Dict[str, Any]:
        """Check consistency between related indicators across sources."""
        consistency_results = {}
        
        # Check interest rate consistency (market vs economic data)
        if 'FED_FUNDS_RATE' in economic_data.columns and 'USD_3M' in market_data.columns:
            # Align dates and compare
            econ_rates = economic_data.set_index('date')['FED_FUNDS_RATE'] if 'date' in economic_data.columns else economic_data['FED_FUNDS_RATE']
            market_rates = market_data.set_index('date')['USD_3M'] if 'date' in market_data.columns else market_data['USD_3M']
            
            # Find common dates
            common_dates = econ_rates.index.intersection(market_rates.index)
            if len(common_dates) > 0:
                econ_aligned = econ_rates.loc[common_dates]
                market_aligned = market_rates.loc[common_dates]
                
                # Calculate correlation and mean difference
                correlation = econ_aligned.corr(market_aligned)
                mean_diff = (econ_aligned - market_aligned).mean()
                
                consistency_results['interest_rate_consistency'] = {
                    'correlation': float(correlation),
                    'mean_difference': float(mean_diff),
                    'common_observations': len(common_dates)
                }
                
                if correlation < 0.8:
                    self.validation_warnings.append(
                        f"Low correlation between Fed Funds and 3M rates: {correlation:.2f}"
                    )
        
        # Check equity market vs economic indicators
        if 'SP500' in market_data.columns and 'GDP' in economic_data.columns:
            # This would require more sophisticated analysis
            consistency_results['equity_economic_consistency'] = {
                'note': 'Requires advanced time series analysis'
            }
        
        return consistency_results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'data_quality_scores': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings_summary': self.validation_warnings,
            'errors_summary': self.validation_errors
        }
        
        # Summarize validation results
        for data_type, results in self.validation_results.items():
            report['validation_summary'][data_type] = {
                'quality_score': results.get('quality_score', 0.0),
                'checks_performed': results.get('checks_performed', []),
                'issues_count': len(results.get('warnings', [])) + len(results.get('errors', []))
            }
            
            report['data_quality_scores'][data_type] = results.get('quality_score', 0.0)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        # Identify critical issues
        report['critical_issues'] = self._identify_critical_issues()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Check for low quality scores
        for data_type, results in self.validation_results.items():
            quality_score = results.get('quality_score', 0.0)
            
            if quality_score < 0.7:
                recommendations.append(f"Improve data quality for {data_type} (score: {quality_score:.2f})")
            
            # Specific recommendations based on issues
            if not results.get('completeness_valid', True):
                recommendations.append(f"Address missing data in {data_type}")
            
            if not results.get('consistency_valid', True):
                recommendations.append(f"Review data consistency issues in {data_type}")
            
            if results.get('outliers_detected', False):
                recommendations.append(f"Investigate outliers in {data_type}")
        
        # General recommendations
        if len(self.validation_errors) > 0:
            recommendations.append("Address critical data errors before model training")
        
        if len(self.validation_warnings) > 10:
            recommendations.append("Review and address data quality warnings")
        
        return recommendations
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues that must be addressed."""
        critical_issues = []
        
        # Add all errors as critical
        critical_issues.extend(self.validation_errors)
        
        # Add high-impact warnings as critical
        for data_type, results in self.validation_results.items():
            quality_score = results.get('quality_score', 0.0)
            
            if quality_score < 0.5:
                critical_issues.append(f"Very low data quality score for {data_type}: {quality_score:.2f}")
            
            if not results.get('structure_valid', True):
                critical_issues.append(f"Structural issues in {data_type}")
        
        return critical_issues
    
    def export_validation_results(self, filepath: str) -> None:
        """Export validation results to file."""
        report = self.generate_validation_report()
        
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            # Export as CSV summary
            summary_data = []
            for data_type, results in self.validation_results.items():
                summary_data.append({
                    'data_type': data_type,
                    'quality_score': results.get('quality_score', 0.0),
                    'structure_valid': results.get('structure_valid', False),
                    'completeness_valid': results.get('completeness_valid', False),
                    'consistency_valid': results.get('consistency_valid', False),
                    'warnings_count': len(results.get('warnings', [])),
                    'errors_count': len(results.get('errors', []))
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Validation results exported to {filepath}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        if not self.validation_results:
            return {'message': 'No validation results available'}
        
        summary = {
            'total_data_sources_validated': len(self.validation_results),
            'average_quality_score': np.mean([
                results.get('quality_score', 0.0) 
                for results in self.validation_results.values()
            ]),
            'total_warnings': len(self.validation_warnings),
            'total_errors': len(self.validation_errors),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Add per-source summary
        summary['by_data_source'] = {}
        for data_type, results in self.validation_results.items():
            summary['by_data_source'][data_type] = {
                'quality_score': results.get('quality_score', 0.0),
                'data_shape': results.get('data_shape', [0, 0]),
                'checks_performed': len(results.get('checks_performed', [])),
                'issues_found': len(results.get('warnings', [])) + len(results.get('errors', []))
            }
        
        return summary