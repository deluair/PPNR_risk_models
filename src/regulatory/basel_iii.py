"""
Basel III Regulatory Compliance Module

This module implements Basel III capital adequacy requirements and calculations
for the PPNR risk models system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CapitalType(Enum):
    """Types of regulatory capital"""
    CET1 = "common_equity_tier1"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TOTAL = "total_capital"


class RiskWeightApproach(Enum):
    """Risk weight calculation approaches"""
    STANDARDIZED = "standardized"
    FOUNDATION_IRB = "foundation_irb"
    ADVANCED_IRB = "advanced_irb"


@dataclass
class CapitalRatios:
    """Basel III capital ratios"""
    cet1_ratio: float
    tier1_ratio: float
    total_capital_ratio: float
    leverage_ratio: float
    is_compliant: bool
    minimum_requirements: Dict[str, float]


@dataclass
class RWAComponents:
    """Risk-weighted assets components"""
    credit_rwa: float
    market_rwa: float
    operational_rwa: float
    total_rwa: float
    exposure_breakdown: Dict[str, float]


class BaselIIICalculator:
    """
    Basel III regulatory capital calculator
    
    Implements Basel III framework for:
    - Risk-weighted assets calculation
    - Capital ratio computation
    - Regulatory compliance assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Basel III calculator
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.minimum_ratios = {
            'cet1_ratio': 0.045,  # 4.5% minimum
            'tier1_ratio': 0.06,  # 6.0% minimum
            'total_capital_ratio': 0.08,  # 8.0% minimum
            'leverage_ratio': 0.03  # 3.0% minimum
        }
        
        # Risk weights for standardized approach
        self.risk_weights = {
            'sovereign': {'aaa_aa': 0.0, 'a': 0.2, 'bbb': 0.5, 'bb_b': 1.0, 'below_b': 1.5},
            'bank': {'aaa_aa': 0.2, 'a': 0.5, 'bbb': 0.5, 'bb_b': 1.0, 'below_b': 1.5},
            'corporate': {'aaa_aa': 0.2, 'a': 0.5, 'bbb': 1.0, 'bb_b': 1.0, 'below_b': 1.5},
            'retail': {'secured': 0.35, 'unsecured': 0.75},
            'real_estate': {'residential': 0.35, 'commercial': 1.0}
        }
    
    def calculate_credit_rwa(self, 
                           portfolio_data: pd.DataFrame,
                           approach: RiskWeightApproach = RiskWeightApproach.STANDARDIZED) -> Dict:
        """
        Calculate credit risk-weighted assets
        
        Args:
            portfolio_data: Portfolio exposure data
            approach: Risk weight calculation approach
            
        Returns:
            Dictionary with credit RWA calculations
        """
        try:
            if approach == RiskWeightApproach.STANDARDIZED:
                return self._calculate_standardized_credit_rwa(portfolio_data)
            elif approach == RiskWeightApproach.FOUNDATION_IRB:
                return self._calculate_foundation_irb_rwa(portfolio_data)
            else:
                return self._calculate_advanced_irb_rwa(portfolio_data)
                
        except Exception as e:
            logger.error(f"Error calculating credit RWA: {str(e)}")
            raise
    
    def _calculate_standardized_credit_rwa(self, portfolio_data: pd.DataFrame) -> Dict:
        """Calculate standardized approach credit RWA"""
        rwa_components = {}
        total_rwa = 0
        
        # Group by asset class and rating
        for asset_class in portfolio_data['asset_class'].unique():
            class_data = portfolio_data[portfolio_data['asset_class'] == asset_class]
            class_rwa = 0
            
            for _, exposure in class_data.iterrows():
                # Get risk weight based on asset class and rating
                risk_weight = self._get_risk_weight(
                    asset_class, 
                    exposure.get('rating', 'unrated')
                )
                
                # Calculate RWA for this exposure
                exposure_rwa = exposure['exposure_amount'] * risk_weight
                class_rwa += exposure_rwa
            
            rwa_components[asset_class] = class_rwa
            total_rwa += class_rwa
        
        return {
            'total_credit_rwa': total_rwa,
            'components': rwa_components,
            'approach': 'standardized'
        }
    
    def _calculate_foundation_irb_rwa(self, portfolio_data: pd.DataFrame) -> Dict:
        """Calculate Foundation IRB approach RWA"""
        # Simplified Foundation IRB calculation
        total_rwa = 0
        components = {}
        
        for _, exposure in portfolio_data.iterrows():
            pd_value = exposure.get('pd', 0.05)  # Default PD if not provided
            lgd_value = exposure.get('lgd', 0.45)  # Default LGD if not provided
            ead_value = exposure.get('ead', exposure['exposure_amount'])
            
            # Simplified IRB formula (actual formula is more complex)
            correlation = 0.12 * (1 - np.exp(-50 * pd_value)) / (1 - np.exp(-50))
            capital_requirement = lgd_value * self._normal_cdf(
                (self._inverse_normal_cdf(pd_value) + 
                 np.sqrt(correlation) * self._inverse_normal_cdf(0.999)) / 
                np.sqrt(1 - correlation)
            ) - pd_value * lgd_value
            
            rwa = capital_requirement * ead_value * 12.5  # 8% capital requirement
            total_rwa += rwa
            
            asset_class = exposure.get('asset_class', 'corporate')
            components[asset_class] = components.get(asset_class, 0) + rwa
        
        return {
            'total_credit_rwa': total_rwa,
            'components': components,
            'approach': 'foundation_irb'
        }
    
    def _calculate_advanced_irb_rwa(self, portfolio_data: pd.DataFrame) -> Dict:
        """Calculate Advanced IRB approach RWA"""
        # Similar to Foundation IRB but with bank's own LGD estimates
        return self._calculate_foundation_irb_rwa(portfolio_data)
    
    def calculate_market_rwa(self, market_positions: pd.DataFrame) -> Dict:
        """
        Calculate market risk-weighted assets
        
        Args:
            market_positions: Market risk positions data
            
        Returns:
            Dictionary with market RWA calculations
        """
        try:
            total_market_rwa = 0
            components = {}
            
            # Interest rate risk
            if 'interest_rate_positions' in market_positions.columns:
                ir_rwa = self._calculate_interest_rate_rwa(market_positions)
                components['interest_rate'] = ir_rwa
                total_market_rwa += ir_rwa
            
            # Equity risk
            if 'equity_positions' in market_positions.columns:
                equity_rwa = self._calculate_equity_rwa(market_positions)
                components['equity'] = equity_rwa
                total_market_rwa += equity_rwa
            
            # Foreign exchange risk
            if 'fx_positions' in market_positions.columns:
                fx_rwa = self._calculate_fx_rwa(market_positions)
                components['foreign_exchange'] = fx_rwa
                total_market_rwa += fx_rwa
            
            # Commodity risk
            if 'commodity_positions' in market_positions.columns:
                commodity_rwa = self._calculate_commodity_rwa(market_positions)
                components['commodity'] = commodity_rwa
                total_market_rwa += commodity_rwa
            
            return {
                'total_market_rwa': total_market_rwa,
                'components': components
            }
            
        except Exception as e:
            logger.error(f"Error calculating market RWA: {str(e)}")
            raise
    
    def calculate_operational_rwa(self, business_indicators: Dict) -> Dict:
        """
        Calculate operational risk-weighted assets using Standardized Approach
        
        Args:
            business_indicators: Business indicator components
            
        Returns:
            Dictionary with operational RWA calculations
        """
        try:
            # Business Indicator (BI) calculation
            interest_component = max(0, business_indicators.get('interest_income', 0) - 
                                   business_indicators.get('interest_expense', 0))
            services_component = business_indicators.get('fee_income', 0)
            financial_component = business_indicators.get('trading_income', 0)
            
            bi = interest_component + services_component + abs(financial_component)
            
            # Marginal coefficients for BI buckets
            if bi <= 1_000_000_000:  # €1bn
                marginal_coeff = 0.12
            elif bi <= 30_000_000_000:  # €30bn
                marginal_coeff = 0.15
            else:
                marginal_coeff = 0.18
            
            # Operational risk capital requirement
            op_risk_capital = bi * marginal_coeff
            
            # Convert to RWA (divide by 8% capital requirement)
            operational_rwa = op_risk_capital / 0.08
            
            return {
                'total_operational_rwa': operational_rwa,
                'business_indicator': bi,
                'marginal_coefficient': marginal_coeff,
                'capital_requirement': op_risk_capital
            }
            
        except Exception as e:
            logger.error(f"Error calculating operational RWA: {str(e)}")
            raise
    
    def calculate_capital_ratios(self, 
                               capital_data: Dict,
                               rwa_data: Dict) -> CapitalRatios:
        """
        Calculate Basel III capital ratios
        
        Args:
            capital_data: Capital components data
            rwa_data: Risk-weighted assets data
            
        Returns:
            CapitalRatios object with all ratios and compliance status
        """
        try:
            total_rwa = rwa_data.get('total_rwa', 0)
            total_exposure = capital_data.get('total_exposure', total_rwa)
            
            # Calculate ratios
            cet1_ratio = capital_data.get('cet1_capital', 0) / total_rwa if total_rwa > 0 else 0
            tier1_ratio = capital_data.get('tier1_capital', 0) / total_rwa if total_rwa > 0 else 0
            total_capital_ratio = capital_data.get('total_capital', 0) / total_rwa if total_rwa > 0 else 0
            leverage_ratio = capital_data.get('tier1_capital', 0) / total_exposure if total_exposure > 0 else 0
            
            # Check compliance
            is_compliant = (
                cet1_ratio >= self.minimum_ratios['cet1_ratio'] and
                tier1_ratio >= self.minimum_ratios['tier1_ratio'] and
                total_capital_ratio >= self.minimum_ratios['total_capital_ratio'] and
                leverage_ratio >= self.minimum_ratios['leverage_ratio']
            )
            
            return CapitalRatios(
                cet1_ratio=cet1_ratio,
                tier1_ratio=tier1_ratio,
                total_capital_ratio=total_capital_ratio,
                leverage_ratio=leverage_ratio,
                is_compliant=is_compliant,
                minimum_requirements=self.minimum_ratios
            )
            
        except Exception as e:
            logger.error(f"Error calculating capital ratios: {str(e)}")
            raise
    
    def calculate_total_rwa(self,
                          portfolio_data: pd.DataFrame,
                          market_positions: pd.DataFrame,
                          business_indicators: Dict) -> RWAComponents:
        """
        Calculate total risk-weighted assets across all risk types
        
        Args:
            portfolio_data: Credit portfolio data
            market_positions: Market risk positions
            business_indicators: Operational risk indicators
            
        Returns:
            RWAComponents with breakdown by risk type
        """
        try:
            # Calculate each RWA component
            credit_rwa_result = self.calculate_credit_rwa(portfolio_data)
            market_rwa_result = self.calculate_market_rwa(market_positions)
            operational_rwa_result = self.calculate_operational_rwa(business_indicators)
            
            # Extract totals
            credit_rwa = credit_rwa_result['total_credit_rwa']
            market_rwa = market_rwa_result['total_market_rwa']
            operational_rwa = operational_rwa_result['total_operational_rwa']
            total_rwa = credit_rwa + market_rwa + operational_rwa
            
            # Create exposure breakdown
            exposure_breakdown = {
                'credit_risk': credit_rwa,
                'market_risk': market_rwa,
                'operational_risk': operational_rwa
            }
            
            return RWAComponents(
                credit_rwa=credit_rwa,
                market_rwa=market_rwa,
                operational_rwa=operational_rwa,
                total_rwa=total_rwa,
                exposure_breakdown=exposure_breakdown
            )
            
        except Exception as e:
            logger.error(f"Error calculating total RWA: {str(e)}")
            raise
    
    def _get_risk_weight(self, asset_class: str, rating: str) -> float:
        """Get risk weight for asset class and rating"""
        if asset_class in self.risk_weights:
            weights = self.risk_weights[asset_class]
            if rating in weights:
                return weights[rating]
            else:
                # Default to highest risk weight if rating not found
                return max(weights.values())
        else:
            # Default risk weight for unknown asset classes
            return 1.0
    
    def _calculate_interest_rate_rwa(self, positions: pd.DataFrame) -> float:
        """Calculate interest rate risk RWA"""
        # Simplified calculation - actual implementation would be more complex
        return positions.get('interest_rate_positions', pd.Series([0])).abs().sum() * 0.08 / 0.08
    
    def _calculate_equity_rwa(self, positions: pd.DataFrame) -> float:
        """Calculate equity risk RWA"""
        return positions.get('equity_positions', pd.Series([0])).abs().sum() * 0.32 / 0.08
    
    def _calculate_fx_rwa(self, positions: pd.DataFrame) -> float:
        """Calculate foreign exchange risk RWA"""
        return positions.get('fx_positions', pd.Series([0])).abs().sum() * 0.08 / 0.08
    
    def _calculate_commodity_rwa(self, positions: pd.DataFrame) -> float:
        """Calculate commodity risk RWA"""
        return positions.get('commodity_positions', pd.Series([0])).abs().sum() * 0.15 / 0.08
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function"""
        from scipy.stats import norm
        return norm.cdf(x)
    
    def _inverse_normal_cdf(self, p: float) -> float:
        """Inverse standard normal cumulative distribution function"""
        from scipy.stats import norm
        return norm.ppf(p)
    
    def generate_compliance_report(self, 
                                 capital_ratios: CapitalRatios,
                                 rwa_components: RWAComponents) -> Dict:
        """
        Generate comprehensive Basel III compliance report
        
        Args:
            capital_ratios: Calculated capital ratios
            rwa_components: RWA breakdown
            
        Returns:
            Comprehensive compliance report
        """
        report = {
            'compliance_status': 'COMPLIANT' if capital_ratios.is_compliant else 'NON-COMPLIANT',
            'capital_ratios': {
                'cet1_ratio': f"{capital_ratios.cet1_ratio:.2%}",
                'tier1_ratio': f"{capital_ratios.tier1_ratio:.2%}",
                'total_capital_ratio': f"{capital_ratios.total_capital_ratio:.2%}",
                'leverage_ratio': f"{capital_ratios.leverage_ratio:.2%}"
            },
            'minimum_requirements': {
                'cet1_ratio': f"{capital_ratios.minimum_requirements['cet1_ratio']:.2%}",
                'tier1_ratio': f"{capital_ratios.minimum_requirements['tier1_ratio']:.2%}",
                'total_capital_ratio': f"{capital_ratios.minimum_requirements['total_capital_ratio']:.2%}",
                'leverage_ratio': f"{capital_ratios.minimum_requirements['leverage_ratio']:.2%}"
            },
            'rwa_breakdown': {
                'total_rwa': f"${rwa_components.total_rwa:,.0f}",
                'credit_rwa': f"${rwa_components.credit_rwa:,.0f}",
                'market_rwa': f"${rwa_components.market_rwa:,.0f}",
                'operational_rwa': f"${rwa_components.operational_rwa:,.0f}"
            },
            'gaps_and_recommendations': self._identify_gaps(capital_ratios)
        }
        
        return report
    
    def _identify_gaps(self, capital_ratios: CapitalRatios) -> List[str]:
        """Identify compliance gaps and provide recommendations"""
        gaps = []
        
        if capital_ratios.cet1_ratio < capital_ratios.minimum_requirements['cet1_ratio']:
            gap = capital_ratios.minimum_requirements['cet1_ratio'] - capital_ratios.cet1_ratio
            gaps.append(f"CET1 ratio shortfall of {gap:.2%}. Consider raising additional equity capital.")
        
        if capital_ratios.tier1_ratio < capital_ratios.minimum_requirements['tier1_ratio']:
            gap = capital_ratios.minimum_requirements['tier1_ratio'] - capital_ratios.tier1_ratio
            gaps.append(f"Tier 1 ratio shortfall of {gap:.2%}. Consider issuing additional Tier 1 instruments.")
        
        if capital_ratios.total_capital_ratio < capital_ratios.minimum_requirements['total_capital_ratio']:
            gap = capital_ratios.minimum_requirements['total_capital_ratio'] - capital_ratios.total_capital_ratio
            gaps.append(f"Total capital ratio shortfall of {gap:.2%}. Consider issuing Tier 2 capital.")
        
        if capital_ratios.leverage_ratio < capital_ratios.minimum_requirements['leverage_ratio']:
            gap = capital_ratios.minimum_requirements['leverage_ratio'] - capital_ratios.leverage_ratio
            gaps.append(f"Leverage ratio shortfall of {gap:.2%}. Consider reducing exposures or raising capital.")
        
        if not gaps:
            gaps.append("All capital ratios meet minimum requirements.")
        
        return gaps