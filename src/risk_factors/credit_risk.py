"""
Credit Risk Modeling Module

Comprehensive credit risk factor modeling for PPNR:
- Credit loss modeling and provisioning
- Probability of Default (PD) modeling
- Loss Given Default (LGD) modeling
- Exposure at Default (EAD) modeling
- Credit migration matrices
- Portfolio-level credit risk assessment
- Stress testing integration
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
import lightgbm as lgb

class CreditRating(Enum):
    """Credit rating categories."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"

class PortfolioSegment(Enum):
    """Portfolio segments for credit risk modeling."""
    RETAIL = "Retail"
    COMMERCIAL = "Commercial"
    CORPORATE = "Corporate"
    REAL_ESTATE = "Real Estate"
    CONSUMER = "Consumer"
    CREDIT_CARD = "Credit Card"

@dataclass
class CreditExposure:
    """Individual credit exposure."""
    exposure_id: str
    borrower_id: str
    portfolio_segment: PortfolioSegment
    current_balance: float
    committed_amount: float
    credit_rating: CreditRating
    pd_estimate: float = 0.0
    lgd_estimate: float = 0.0
    ead_estimate: float = 0.0
    maturity_years: float = 1.0
    
    @property
    def expected_loss(self) -> float:
        """Calculate expected loss (EL = PD × LGD × EAD)."""
        return self.pd_estimate * self.lgd_estimate * self.ead_estimate

@dataclass
class CreditRiskFactors:
    """Credit risk factors for modeling."""
    # Macroeconomic factors
    gdp_growth: float = 0.0
    unemployment_rate: float = 0.0
    interest_rates: float = 0.0
    inflation_rate: float = 0.0
    house_price_index: float = 0.0
    
    # Market factors
    credit_spreads: float = 0.0
    equity_volatility: float = 0.0
    corporate_bond_yields: float = 0.0
    
    # Bank-specific factors
    loan_growth: float = 0.0
    credit_standards: float = 0.0
    portfolio_concentration: float = 0.0
    
    # Industry factors
    industry_stress_indicators: Dict[str, float] = None
    
    def __post_init__(self):
        if self.industry_stress_indicators is None:
            self.industry_stress_indicators = {}

class CreditRiskModel:
    """
    Comprehensive credit risk modeling system.
    
    Features:
    - Multi-segment PD/LGD/EAD modeling
    - Credit migration matrix estimation
    - Portfolio-level loss distribution modeling
    - Stress testing integration
    - CECL/IFRS 9 compliance support
    - Model validation and backtesting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize credit risk model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.credit_config = config.get('credit_risk', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.CreditRisk")
        
        # Model components
        self.pd_models = {}  # PD models by segment
        self.lgd_models = {}  # LGD models by segment
        self.ead_models = {}  # EAD models by segment
        
        # Migration matrices
        self.migration_matrices = {}
        
        # Portfolio data
        self.exposures: List[CreditExposure] = []
        self.historical_data = {}
        
        # Model performance tracking
        self.model_performance = {}
        self.validation_results = {}
        
        # Scalers for feature normalization
        self.scalers = {}
        
        self.logger.info("Credit risk model initialized")
    
    def load_portfolio_data(self, portfolio_data: pd.DataFrame) -> None:
        """
        Load portfolio exposure data.
        
        Args:
            portfolio_data: DataFrame with exposure information
        """
        self.logger.info(f"Loading portfolio data with {len(portfolio_data)} exposures")
        
        self.exposures = []
        
        for _, row in portfolio_data.iterrows():
            exposure = CreditExposure(
                exposure_id=str(row.get('exposure_id', '')),
                borrower_id=str(row.get('borrower_id', '')),
                portfolio_segment=PortfolioSegment(row.get('portfolio_segment', 'RETAIL')),
                current_balance=float(row.get('current_balance', 0.0)),
                committed_amount=float(row.get('committed_amount', 0.0)),
                credit_rating=CreditRating(row.get('credit_rating', 'BBB')),
                pd_estimate=float(row.get('pd_estimate', 0.0)),
                lgd_estimate=float(row.get('lgd_estimate', 0.0)),
                ead_estimate=float(row.get('ead_estimate', 0.0)),
                maturity_years=float(row.get('maturity_years', 1.0))
            )
            self.exposures.append(exposure)
        
        self.logger.info(f"Loaded {len(self.exposures)} credit exposures")
    
    def load_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Load historical data for model training.
        
        Args:
            historical_data: Dictionary of historical datasets
        """
        self.logger.info("Loading historical credit data")
        
        self.historical_data = historical_data
        
        # Validate required datasets
        required_datasets = ['defaults', 'recoveries', 'exposures', 'economic_factors']
        for dataset in required_datasets:
            if dataset not in historical_data:
                self.logger.warning(f"Missing historical dataset: {dataset}")
        
        self.logger.info("Historical credit data loaded")
    
    def train_pd_models(self, segments: List[PortfolioSegment] = None) -> Dict[str, Any]:
        """
        Train Probability of Default (PD) models.
        
        Args:
            segments: Portfolio segments to train models for
            
        Returns:
            Training results and model performance
        """
        if segments is None:
            segments = list(PortfolioSegment)
        
        self.logger.info(f"Training PD models for {len(segments)} segments")
        
        training_results = {
            'models_trained': [],
            'performance_metrics': {},
            'feature_importance': {},
            'validation_results': {}
        }
        
        # Check if historical data is available
        if 'defaults' not in self.historical_data:
            self.logger.error("No historical default data available for PD model training")
            return training_results
        
        defaults_data = self.historical_data['defaults']
        
        for segment in segments:
            self.logger.info(f"Training PD model for {segment.value}")
            
            # Filter data for segment
            segment_data = defaults_data[defaults_data['portfolio_segment'] == segment.value].copy()
            
            if len(segment_data) < 100:  # Minimum sample size
                self.logger.warning(f"Insufficient data for {segment.value} PD model: {len(segment_data)} observations")
                continue
            
            # Prepare features and target
            features, target = self._prepare_pd_features(segment_data)
            
            if features is None or len(features) == 0:
                self.logger.warning(f"No valid features for {segment.value} PD model")
                continue
            
            # Train model
            model_results = self._train_pd_model_segment(segment, features, target)
            
            if model_results['model'] is not None:
                self.pd_models[segment] = model_results['model']
                training_results['models_trained'].append(segment.value)
                training_results['performance_metrics'][segment.value] = model_results['performance']
                training_results['feature_importance'][segment.value] = model_results['feature_importance']
                training_results['validation_results'][segment.value] = model_results['validation']
        
        self.logger.info(f"PD model training completed for {len(training_results['models_trained'])} segments")
        return training_results
    
    def _prepare_pd_features(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare features for PD model training."""
        try:
            # Define feature columns
            feature_columns = [
                'borrower_age', 'credit_score', 'debt_to_income', 'loan_to_value',
                'employment_length', 'income', 'loan_amount', 'interest_rate',
                'gdp_growth', 'unemployment_rate', 'house_price_index'
            ]
            
            # Select available features
            available_features = [col for col in feature_columns if col in data.columns]
            
            if not available_features:
                return None, None
            
            features = data[available_features].copy()
            target = data['default_flag'] if 'default_flag' in data.columns else None
            
            if target is None:
                return None, None
            
            # Handle missing values
            features = features.fillna(features.median())
            
            # Remove rows with missing target
            valid_indices = target.notna()
            features = features[valid_indices]
            target = target[valid_indices]
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error preparing PD features: {str(e)}")
            return None, None
    
    def _train_pd_model_segment(self, segment: PortfolioSegment, 
                               features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Train PD model for a specific segment."""
        results = {
            'model': None,
            'performance': {},
            'feature_importance': {},
            'validation': {}
        }
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[f'pd_{segment.value}'] = scaler
            
            # Train multiple models and select best
            models = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
            }
            
            best_model = None
            best_score = 0
            best_model_name = None
            
            for model_name, model in models.items():
                try:
                    if model_name == 'lightgbm':
                        model.fit(X_train, y_train)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        model.fit(X_train_scaled, y_train)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    if auc_score > best_score:
                        best_score = auc_score
                        best_model = model
                        best_model_name = model_name
                        
                except Exception as e:
                    self.logger.warning(f"Error training {model_name} for {segment.value}: {str(e)}")
                    continue
            
            if best_model is None:
                return results
            
            # Calculate performance metrics
            if best_model_name == 'lightgbm':
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                y_pred = best_model.predict(X_test)
            else:
                y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                y_pred = best_model.predict(X_test_scaled)
            
            results['performance'] = {
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'model_type': best_model_name,
                'sample_size': len(X_train)
            }
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_dict = dict(zip(features.columns, best_model.feature_importances_))
                results['feature_importance'] = dict(sorted(importance_dict.items(), 
                                                          key=lambda x: x[1], reverse=True))
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train_scaled if best_model_name != 'lightgbm' else X_train, 
                                      y_train, cv=5, scoring='roc_auc')
            results['validation'] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            results['model'] = best_model
            
        except Exception as e:
            self.logger.error(f"Error training PD model for {segment.value}: {str(e)}")
        
        return results
    
    def train_lgd_models(self, segments: List[PortfolioSegment] = None) -> Dict[str, Any]:
        """
        Train Loss Given Default (LGD) models.
        
        Args:
            segments: Portfolio segments to train models for
            
        Returns:
            Training results and model performance
        """
        if segments is None:
            segments = list(PortfolioSegment)
        
        self.logger.info(f"Training LGD models for {len(segments)} segments")
        
        training_results = {
            'models_trained': [],
            'performance_metrics': {},
            'feature_importance': {},
            'validation_results': {}
        }
        
        # Check if historical data is available
        if 'recoveries' not in self.historical_data:
            self.logger.error("No historical recovery data available for LGD model training")
            return training_results
        
        recoveries_data = self.historical_data['recoveries']
        
        for segment in segments:
            self.logger.info(f"Training LGD model for {segment.value}")
            
            # Filter data for segment
            segment_data = recoveries_data[recoveries_data['portfolio_segment'] == segment.value].copy()
            
            if len(segment_data) < 50:  # Minimum sample size for LGD
                self.logger.warning(f"Insufficient data for {segment.value} LGD model: {len(segment_data)} observations")
                continue
            
            # Prepare features and target
            features, target = self._prepare_lgd_features(segment_data)
            
            if features is None or len(features) == 0:
                self.logger.warning(f"No valid features for {segment.value} LGD model")
                continue
            
            # Train model
            model_results = self._train_lgd_model_segment(segment, features, target)
            
            if model_results['model'] is not None:
                self.lgd_models[segment] = model_results['model']
                training_results['models_trained'].append(segment.value)
                training_results['performance_metrics'][segment.value] = model_results['performance']
                training_results['feature_importance'][segment.value] = model_results['feature_importance']
                training_results['validation_results'][segment.value] = model_results['validation']
        
        self.logger.info(f"LGD model training completed for {len(training_results['models_trained'])} segments")
        return training_results
    
    def _prepare_lgd_features(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare features for LGD model training."""
        try:
            # Define feature columns for LGD
            feature_columns = [
                'loan_to_value', 'collateral_type', 'seniority', 'recovery_time',
                'workout_strategy', 'collateral_value', 'loan_amount',
                'gdp_growth', 'house_price_index', 'credit_spreads'
            ]
            
            # Select available features
            available_features = [col for col in feature_columns if col in data.columns]
            
            if not available_features:
                return None, None
            
            features = data[available_features].copy()
            
            # Calculate LGD from recovery data
            if 'recovery_amount' in data.columns and 'exposure_at_default' in data.columns:
                target = 1 - (data['recovery_amount'] / data['exposure_at_default'])
                target = target.clip(0, 1)  # Ensure LGD is between 0 and 1
            elif 'lgd' in data.columns:
                target = data['lgd']
            else:
                return None, None
            
            # Handle missing values
            features = features.fillna(features.median())
            
            # Remove rows with missing target
            valid_indices = target.notna()
            features = features[valid_indices]
            target = target[valid_indices]
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error preparing LGD features: {str(e)}")
            return None, None
    
    def _train_lgd_model_segment(self, segment: PortfolioSegment,
                                features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Train LGD model for a specific segment."""
        results = {
            'model': None,
            'performance': {},
            'feature_importance': {},
            'validation': {}
        }
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[f'lgd_{segment.value}'] = scaler
            
            # Train multiple models and select best
            models = {
                'linear': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'lightgbm': lgb.LGBMRegressor(random_state=42, verbose=-1)
            }
            
            best_model = None
            best_score = float('inf')
            best_model_name = None
            
            for model_name, model in models.items():
                try:
                    if model_name == 'lightgbm':
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    else:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    
                    if mse < best_score:
                        best_score = mse
                        best_model = model
                        best_model_name = model_name
                        
                except Exception as e:
                    self.logger.warning(f"Error training {model_name} for {segment.value} LGD: {str(e)}")
                    continue
            
            if best_model is None:
                return results
            
            # Calculate performance metrics
            if best_model_name == 'lightgbm':
                y_pred = best_model.predict(X_test)
            else:
                y_pred = best_model.predict(X_test_scaled)
            
            results['performance'] = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': best_model.score(X_test_scaled if best_model_name != 'lightgbm' else X_test, y_test),
                'model_type': best_model_name,
                'sample_size': len(X_train)
            }
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_dict = dict(zip(features.columns, best_model.feature_importances_))
                results['feature_importance'] = dict(sorted(importance_dict.items(),
                                                          key=lambda x: x[1], reverse=True))
            
            results['model'] = best_model
            
        except Exception as e:
            self.logger.error(f"Error training LGD model for {segment.value}: {str(e)}")
        
        return results
    
    def estimate_migration_matrix(self, historical_ratings: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Estimate credit migration matrices from historical rating data.
        
        Args:
            historical_ratings: DataFrame with historical rating transitions
            
        Returns:
            Migration matrices by portfolio segment
        """
        self.logger.info("Estimating credit migration matrices")
        
        migration_matrices = {}
        
        # Get unique segments
        segments = historical_ratings['portfolio_segment'].unique()
        
        for segment in segments:
            segment_data = historical_ratings[historical_ratings['portfolio_segment'] == segment]
            
            # Create migration matrix
            ratings = [rating.value for rating in CreditRating]
            matrix = np.zeros((len(ratings), len(ratings)))
            
            # Count transitions
            for _, row in segment_data.iterrows():
                from_rating = row['rating_from']
                to_rating = row['rating_to']
                
                if from_rating in ratings and to_rating in ratings:
                    from_idx = ratings.index(from_rating)
                    to_idx = ratings.index(to_rating)
                    matrix[from_idx, to_idx] += 1
            
            # Normalize to probabilities
            row_sums = matrix.sum(axis=1)
            for i in range(len(ratings)):
                if row_sums[i] > 0:
                    matrix[i, :] = matrix[i, :] / row_sums[i]
                else:
                    # If no transitions observed, assume no migration
                    matrix[i, i] = 1.0
            
            migration_matrices[segment] = {
                'matrix': matrix,
                'ratings': ratings,
                'sample_size': len(segment_data)
            }
        
        self.migration_matrices = migration_matrices
        self.logger.info(f"Migration matrices estimated for {len(migration_matrices)} segments")
        
        return migration_matrices
    
    def predict_portfolio_losses(self, risk_factors: CreditRiskFactors,
                               time_horizon: int = 12) -> Dict[str, Any]:
        """
        Predict portfolio-level credit losses.
        
        Args:
            risk_factors: Credit risk factors for prediction
            time_horizon: Prediction horizon in months
            
        Returns:
            Portfolio loss predictions
        """
        self.logger.info(f"Predicting portfolio losses for {time_horizon} month horizon")
        
        predictions = {
            'total_expected_loss': 0.0,
            'segment_losses': {},
            'loss_distribution': {},
            'risk_contributions': {},
            'confidence_intervals': {}
        }
        
        if not self.exposures:
            self.logger.warning("No portfolio exposures loaded")
            return predictions
        
        # Group exposures by segment
        segment_exposures = {}
        for exposure in self.exposures:
            segment = exposure.portfolio_segment
            if segment not in segment_exposures:
                segment_exposures[segment] = []
            segment_exposures[segment].append(exposure)
        
        # Predict losses by segment
        total_el = 0.0
        
        for segment, exposures in segment_exposures.items():
            segment_predictions = self._predict_segment_losses(segment, exposures, risk_factors, time_horizon)
            
            predictions['segment_losses'][segment.value] = segment_predictions
            total_el += segment_predictions['expected_loss']
        
        predictions['total_expected_loss'] = total_el
        
        # Calculate portfolio-level statistics
        predictions['loss_distribution'] = self._estimate_loss_distribution(segment_exposures, risk_factors)
        predictions['risk_contributions'] = self._calculate_risk_contributions(segment_exposures)
        predictions['confidence_intervals'] = self._calculate_confidence_intervals(predictions['loss_distribution'])
        
        self.logger.info(f"Portfolio loss prediction completed. Expected loss: ${total_el:,.2f}")
        return predictions
    
    def _predict_segment_losses(self, segment: PortfolioSegment, exposures: List[CreditExposure],
                               risk_factors: CreditRiskFactors, time_horizon: int) -> Dict[str, Any]:
        """Predict losses for a specific portfolio segment."""
        segment_results = {
            'expected_loss': 0.0,
            'exposure_count': len(exposures),
            'total_exposure': sum(exp.current_balance for exp in exposures),
            'average_pd': 0.0,
            'average_lgd': 0.0,
            'individual_predictions': []
        }
        
        if segment not in self.pd_models or segment not in self.lgd_models:
            # Use current estimates if models not available
            total_el = sum(exp.expected_loss for exp in exposures)
            segment_results['expected_loss'] = total_el
            segment_results['average_pd'] = np.mean([exp.pd_estimate for exp in exposures])
            segment_results['average_lgd'] = np.mean([exp.lgd_estimate for exp in exposures])
            return segment_results
        
        # Prepare risk factor features
        risk_factor_features = self._prepare_risk_factor_features(risk_factors)
        
        pd_model = self.pd_models[segment]
        lgd_model = self.lgd_models[segment]
        
        total_el = 0.0
        pd_sum = 0.0
        lgd_sum = 0.0
        
        for exposure in exposures:
            # Prepare features for this exposure
            exposure_features = self._prepare_exposure_features(exposure, risk_factor_features)
            
            if exposure_features is not None:
                # Predict PD
                try:
                    pd_scaler = self.scalers.get(f'pd_{segment.value}')
                    if pd_scaler and hasattr(pd_model, 'predict_proba'):
                        if hasattr(pd_model, 'predict_proba'):
                            features_scaled = pd_scaler.transform([exposure_features])
                            pd_pred = pd_model.predict_proba(features_scaled)[0, 1]
                        else:
                            pd_pred = exposure.pd_estimate
                    else:
                        pd_pred = exposure.pd_estimate
                except:
                    pd_pred = exposure.pd_estimate
                
                # Predict LGD
                try:
                    lgd_scaler = self.scalers.get(f'lgd_{segment.value}')
                    if lgd_scaler and hasattr(lgd_model, 'predict'):
                        features_scaled = lgd_scaler.transform([exposure_features])
                        lgd_pred = max(0, min(1, lgd_model.predict(features_scaled)[0]))
                    else:
                        lgd_pred = exposure.lgd_estimate
                except:
                    lgd_pred = exposure.lgd_estimate
                
                # Use EAD estimate (could be enhanced with EAD model)
                ead_pred = exposure.ead_estimate if exposure.ead_estimate > 0 else exposure.current_balance
                
                # Calculate expected loss
                el = pd_pred * lgd_pred * ead_pred
                total_el += el
                pd_sum += pd_pred
                lgd_sum += lgd_pred
                
                segment_results['individual_predictions'].append({
                    'exposure_id': exposure.exposure_id,
                    'pd': pd_pred,
                    'lgd': lgd_pred,
                    'ead': ead_pred,
                    'expected_loss': el
                })
            else:
                # Fallback to current estimates
                el = exposure.expected_loss
                total_el += el
                pd_sum += exposure.pd_estimate
                lgd_sum += exposure.lgd_estimate
        
        segment_results['expected_loss'] = total_el
        segment_results['average_pd'] = pd_sum / len(exposures) if exposures else 0.0
        segment_results['average_lgd'] = lgd_sum / len(exposures) if exposures else 0.0
        
        return segment_results
    
    def _prepare_risk_factor_features(self, risk_factors: CreditRiskFactors) -> Dict[str, float]:
        """Prepare risk factor features for prediction."""
        return {
            'gdp_growth': risk_factors.gdp_growth,
            'unemployment_rate': risk_factors.unemployment_rate,
            'interest_rates': risk_factors.interest_rates,
            'inflation_rate': risk_factors.inflation_rate,
            'house_price_index': risk_factors.house_price_index,
            'credit_spreads': risk_factors.credit_spreads,
            'equity_volatility': risk_factors.equity_volatility,
            'corporate_bond_yields': risk_factors.corporate_bond_yields,
            'loan_growth': risk_factors.loan_growth,
            'credit_standards': risk_factors.credit_standards,
            'portfolio_concentration': risk_factors.portfolio_concentration
        }
    
    def _prepare_exposure_features(self, exposure: CreditExposure, 
                                  risk_factors: Dict[str, float]) -> Optional[List[float]]:
        """Prepare features for individual exposure prediction."""
        try:
            # This would need to be customized based on actual model features
            # For now, return a simplified feature vector
            features = [
                exposure.current_balance,
                exposure.maturity_years,
                risk_factors.get('gdp_growth', 0.0),
                risk_factors.get('unemployment_rate', 0.0),
                risk_factors.get('house_price_index', 0.0)
            ]
            return features
        except:
            return None
    
    def _estimate_loss_distribution(self, segment_exposures: Dict[PortfolioSegment, List[CreditExposure]],
                                   risk_factors: CreditRiskFactors) -> Dict[str, Any]:
        """Estimate portfolio loss distribution using Monte Carlo simulation."""
        distribution = {
            'percentiles': {},
            'var_95': 0.0,
            'var_99': 0.0,
            'expected_shortfall_95': 0.0,
            'simulation_results': []
        }
        
        # Simple Monte Carlo simulation
        n_simulations = 1000
        simulated_losses = []
        
        for _ in range(n_simulations):
            total_loss = 0.0
            
            for segment, exposures in segment_exposures.items():
                for exposure in exposures:
                    # Simulate default
                    pd = exposure.pd_estimate
                    if np.random.random() < pd:
                        # Default occurred, calculate loss
                        lgd = exposure.lgd_estimate
                        ead = exposure.ead_estimate if exposure.ead_estimate > 0 else exposure.current_balance
                        loss = lgd * ead
                        total_loss += loss
            
            simulated_losses.append(total_loss)
        
        simulated_losses = np.array(simulated_losses)
        
        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            distribution['percentiles'][f'p{p}'] = np.percentile(simulated_losses, p)
        
        distribution['var_95'] = np.percentile(simulated_losses, 95)
        distribution['var_99'] = np.percentile(simulated_losses, 99)
        
        # Expected shortfall (conditional VaR)
        var_95_threshold = distribution['var_95']
        tail_losses = simulated_losses[simulated_losses >= var_95_threshold]
        distribution['expected_shortfall_95'] = np.mean(tail_losses) if len(tail_losses) > 0 else var_95_threshold
        
        return distribution
    
    def _calculate_risk_contributions(self, segment_exposures: Dict[PortfolioSegment, List[CreditExposure]]) -> Dict[str, float]:
        """Calculate risk contributions by segment."""
        contributions = {}
        total_exposure = 0.0
        
        # Calculate total exposure
        for exposures in segment_exposures.values():
            total_exposure += sum(exp.current_balance for exp in exposures)
        
        # Calculate contributions
        for segment, exposures in segment_exposures.items():
            segment_exposure = sum(exp.current_balance for exp in exposures)
            contribution = segment_exposure / total_exposure if total_exposure > 0 else 0.0
            contributions[segment.value] = contribution
        
        return contributions
    
    def _calculate_confidence_intervals(self, loss_distribution: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for loss predictions."""
        intervals = {}
        
        percentiles = loss_distribution.get('percentiles', {})
        
        # 90% confidence interval
        if 'p5' in percentiles and 'p95' in percentiles:
            intervals['90%'] = (percentiles.get('p5', 0.0), percentiles['p95'])
        
        # 95% confidence interval
        if 'p2.5' in percentiles and 'p97.5' in percentiles:
            intervals['95%'] = (percentiles.get('p2.5', 0.0), percentiles.get('p97.5', 0.0))
        
        return intervals
    
    def stress_test_credit_losses(self, stress_scenarios: List[CreditRiskFactors]) -> Dict[str, Any]:
        """
        Perform stress testing on credit losses.
        
        Args:
            stress_scenarios: List of stress scenario risk factors
            
        Returns:
            Stress test results
        """
        self.logger.info(f"Running credit stress tests with {len(stress_scenarios)} scenarios")
        
        stress_results = {
            'baseline_losses': {},
            'scenario_results': {},
            'stress_impact': {},
            'worst_case_scenario': {}
        }
        
        # Calculate baseline losses (current risk factors)
        baseline_factors = CreditRiskFactors()  # Default/current factors
        baseline_losses = self.predict_portfolio_losses(baseline_factors)
        stress_results['baseline_losses'] = baseline_losses
        
        # Run stress scenarios
        scenario_losses = []
        
        for i, scenario_factors in enumerate(stress_scenarios):
            scenario_name = f"Scenario_{i+1}"
            scenario_losses_result = self.predict_portfolio_losses(scenario_factors)
            
            stress_results['scenario_results'][scenario_name] = scenario_losses_result
            scenario_losses.append(scenario_losses_result['total_expected_loss'])
        
        # Calculate stress impact
        baseline_el = baseline_losses['total_expected_loss']
        max_stress_el = max(scenario_losses) if scenario_losses else baseline_el
        
        stress_results['stress_impact'] = {
            'baseline_expected_loss': baseline_el,
            'maximum_stress_loss': max_stress_el,
            'stress_multiple': max_stress_el / baseline_el if baseline_el > 0 else 1.0,
            'additional_loss': max_stress_el - baseline_el
        }
        
        # Identify worst case scenario
        if scenario_losses:
            worst_case_idx = np.argmax(scenario_losses)
            stress_results['worst_case_scenario'] = {
                'scenario_name': f"Scenario_{worst_case_idx+1}",
                'expected_loss': scenario_losses[worst_case_idx],
                'scenario_factors': stress_scenarios[worst_case_idx]
            }
        
        self.logger.info(f"Credit stress testing completed. Maximum stress loss: ${max_stress_el:,.2f}")
        return stress_results
    
    def generate_credit_report(self) -> Dict[str, Any]:
        """Generate comprehensive credit risk report."""
        self.logger.info("Generating credit risk report")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'report_type': 'Credit Risk Analysis',
                'version': '1.0'
            },
            'portfolio_summary': {},
            'model_performance': {},
            'risk_metrics': {},
            'recommendations': []
        }
        
        # Portfolio summary
        if self.exposures:
            total_exposure = sum(exp.current_balance for exp in self.exposures)
            total_el = sum(exp.expected_loss for exp in self.exposures)
            
            segment_breakdown = {}
            for exposure in self.exposures:
                segment = exposure.portfolio_segment.value
                if segment not in segment_breakdown:
                    segment_breakdown[segment] = {'count': 0, 'exposure': 0.0, 'el': 0.0}
                
                segment_breakdown[segment]['count'] += 1
                segment_breakdown[segment]['exposure'] += exposure.current_balance
                segment_breakdown[segment]['el'] += exposure.expected_loss
            
            report['portfolio_summary'] = {
                'total_exposures': len(self.exposures),
                'total_exposure_amount': total_exposure,
                'total_expected_loss': total_el,
                'portfolio_loss_rate': (total_el / total_exposure * 100) if total_exposure > 0 else 0.0,
                'segment_breakdown': segment_breakdown
            }
        
        # Model performance
        report['model_performance'] = {
            'pd_models': len(self.pd_models),
            'lgd_models': len(self.lgd_models),
            'migration_matrices': len(self.migration_matrices),
            'model_validation_status': 'Pending'  # Would be updated with actual validation
        }
        
        # Risk metrics
        if self.exposures:
            pds = [exp.pd_estimate for exp in self.exposures if exp.pd_estimate > 0]
            lgds = [exp.lgd_estimate for exp in self.exposures if exp.lgd_estimate > 0]
            
            report['risk_metrics'] = {
                'average_pd': np.mean(pds) if pds else 0.0,
                'median_pd': np.median(pds) if pds else 0.0,
                'average_lgd': np.mean(lgds) if lgds else 0.0,
                'median_lgd': np.median(lgds) if lgds else 0.0,
                'concentration_risk': self._assess_concentration_risk()
            }
        
        # Recommendations
        report['recommendations'] = self._generate_credit_recommendations()
        
        self.logger.info("Credit risk report generated")
        return report
    
    def _assess_concentration_risk(self) -> Dict[str, Any]:
        """Assess portfolio concentration risk."""
        concentration = {
            'segment_concentration': {},
            'rating_concentration': {},
            'hhi_index': 0.0
        }
        
        if not self.exposures:
            return concentration
        
        total_exposure = sum(exp.current_balance for exp in self.exposures)
        
        # Segment concentration
        segment_exposures = {}
        for exposure in self.exposures:
            segment = exposure.portfolio_segment.value
            segment_exposures[segment] = segment_exposures.get(segment, 0.0) + exposure.current_balance
        
        for segment, exposure in segment_exposures.items():
            concentration['segment_concentration'][segment] = (exposure / total_exposure * 100) if total_exposure > 0 else 0.0
        
        # Rating concentration
        rating_exposures = {}
        for exposure in self.exposures:
            rating = exposure.credit_rating.value
            rating_exposures[rating] = rating_exposures.get(rating, 0.0) + exposure.current_balance
        
        for rating, exposure in rating_exposures.items():
            concentration['rating_concentration'][rating] = (exposure / total_exposure * 100) if total_exposure > 0 else 0.0
        
        # Herfindahl-Hirschman Index for segment concentration
        hhi = sum((share/100)**2 for share in concentration['segment_concentration'].values())
        concentration['hhi_index'] = hhi
        
        return concentration
    
    def _generate_credit_recommendations(self) -> List[Dict[str, Any]]:
        """Generate credit risk management recommendations."""
        recommendations = []
        
        # Model-based recommendations
        if len(self.pd_models) < len(PortfolioSegment):
            recommendations.append({
                'category': 'Model Development',
                'priority': 'HIGH',
                'recommendation': 'Develop PD models for all portfolio segments',
                'timeline': '3-6 months'
            })
        
        if len(self.lgd_models) < len(PortfolioSegment):
            recommendations.append({
                'category': 'Model Development',
                'priority': 'HIGH',
                'recommendation': 'Develop LGD models for all portfolio segments',
                'timeline': '3-6 months'
            })
        
        # Portfolio-based recommendations
        if self.exposures:
            concentration = self._assess_concentration_risk()
            max_segment_concentration = max(concentration['segment_concentration'].values()) if concentration['segment_concentration'] else 0
            
            if max_segment_concentration > 50:
                recommendations.append({
                    'category': 'Portfolio Management',
                    'priority': 'MEDIUM',
                    'recommendation': 'Reduce portfolio concentration risk through diversification',
                    'timeline': '6-12 months'
                })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Model Validation',
                'priority': 'MEDIUM',
                'recommendation': 'Implement regular model validation and backtesting',
                'timeline': '3-6 months'
            },
            {
                'category': 'Stress Testing',
                'priority': 'MEDIUM',
                'recommendation': 'Enhance stress testing scenarios and methodologies',
                'timeline': '6-12 months'
            }
        ])
        
        return recommendations