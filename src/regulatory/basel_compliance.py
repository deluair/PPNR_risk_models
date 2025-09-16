"""
Basel III Compliance Module

Implements Basel III capital adequacy and liquidity requirements:
- Capital ratio calculations (CET1, Tier 1, Total Capital)
- Risk-weighted asset calculations
- Leverage ratio requirements
- Liquidity Coverage Ratio (LCR)
- Net Stable Funding Ratio (NSFR)
- Capital conservation buffer
- Countercyclical capital buffer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import json

@dataclass
class BaselRequirements:
    """Basel III regulatory requirements and minimum ratios."""
    # Capital ratios (as percentages)
    cet1_minimum: float = 4.5
    tier1_minimum: float = 6.0
    total_capital_minimum: float = 8.0
    
    # Buffers (as percentages)
    capital_conservation_buffer: float = 2.5
    countercyclical_buffer_max: float = 2.5
    
    # Leverage ratio
    leverage_ratio_minimum: float = 3.0
    
    # Liquidity ratios
    lcr_minimum: float = 100.0  # 100%
    nsfr_minimum: float = 100.0  # 100%
    
    # G-SIB surcharge (varies by bucket)
    gsib_surcharge_buckets: Dict[int, float] = None
    
    def __post_init__(self):
        if self.gsib_surcharge_buckets is None:
            self.gsib_surcharge_buckets = {
                1: 1.0,   # Bucket 1: 1.0%
                2: 1.5,   # Bucket 2: 1.5%
                3: 2.0,   # Bucket 3: 2.0%
                4: 2.5,   # Bucket 4: 2.5%
                5: 3.5    # Bucket 5: 3.5%
            }

@dataclass
class CapitalComponents:
    """Basel III capital components."""
    # Common Equity Tier 1
    common_stock: float = 0
    retained_earnings: float = 0
    accumulated_other_comprehensive_income: float = 0
    cet1_regulatory_adjustments: float = 0
    
    # Additional Tier 1
    additional_tier1_instruments: float = 0
    at1_regulatory_adjustments: float = 0
    
    # Tier 2
    tier2_instruments: float = 0
    tier2_regulatory_adjustments: float = 0
    
    def calculate_cet1(self) -> float:
        """Calculate Common Equity Tier 1 capital."""
        return (self.common_stock + self.retained_earnings + 
                self.accumulated_other_comprehensive_income - 
                self.cet1_regulatory_adjustments)
    
    def calculate_tier1(self) -> float:
        """Calculate Tier 1 capital."""
        return (self.calculate_cet1() + self.additional_tier1_instruments - 
                self.at1_regulatory_adjustments)
    
    def calculate_total_capital(self) -> float:
        """Calculate Total capital."""
        return (self.calculate_tier1() + self.tier2_instruments - 
                self.tier2_regulatory_adjustments)

class BaselCompliance:
    """
    Basel III compliance framework for capital and liquidity requirements.
    
    Features:
    - Capital ratio calculations and monitoring
    - Risk-weighted asset calculations
    - Leverage ratio calculations
    - Liquidity ratio calculations (LCR, NSFR)
    - Buffer requirements and restrictions
    - G-SIB surcharge calculations
    - Stress testing integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Basel III compliance framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.basel_config = config.get('basel', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.BaselCompliance")
        
        # Initialize requirements
        self.requirements = BaselRequirements()
        
        # Bank characteristics
        self.bank_type = self.basel_config.get('bank_type', 'regional')  # 'gsib', 'large', 'regional'
        self.gsib_bucket = self.basel_config.get('gsib_bucket', None)
        self.jurisdiction = self.basel_config.get('jurisdiction', 'US')
        
        # Capital components
        self.capital_components = None
        
        # Risk-weighted assets
        self.rwa_components = {}
        
        # Liquidity components
        self.liquidity_components = {}
        
        # Results storage
        self.compliance_results = {}
        self.capital_ratios = {}
        self.buffer_requirements = {}
        
        # Calculation date
        self.calculation_date = self.basel_config.get('calculation_date', datetime.now())
        
        self.logger.info(f"Basel III compliance framework initialized for {self.bank_type} bank")
    
    def load_capital_data(self, capital_data: Dict[str, float]) -> None:
        """Load capital component data."""
        self.capital_components = CapitalComponents(
            common_stock=capital_data.get('common_stock', 0),
            retained_earnings=capital_data.get('retained_earnings', 0),
            accumulated_other_comprehensive_income=capital_data.get('aoci', 0),
            cet1_regulatory_adjustments=capital_data.get('cet1_adjustments', 0),
            additional_tier1_instruments=capital_data.get('at1_instruments', 0),
            at1_regulatory_adjustments=capital_data.get('at1_adjustments', 0),
            tier2_instruments=capital_data.get('tier2_instruments', 0),
            tier2_regulatory_adjustments=capital_data.get('tier2_adjustments', 0)
        )
        
        self.logger.info("Capital component data loaded")
    
    def calculate_risk_weighted_assets(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk-weighted assets by category.
        
        Args:
            portfolio_data: Portfolio exposure and risk data
            
        Returns:
            Risk-weighted assets by category
        """
        rwa_components = {}
        
        # Credit Risk RWA
        credit_rwa = self._calculate_credit_risk_rwa(portfolio_data.get('credit_exposures', {}))
        rwa_components.update(credit_rwa)
        
        # Market Risk RWA
        market_rwa = self._calculate_market_risk_rwa(portfolio_data.get('market_exposures', {}))
        rwa_components['market_risk_rwa'] = market_rwa
        
        # Operational Risk RWA
        operational_rwa = self._calculate_operational_risk_rwa(portfolio_data.get('operational_data', {}))
        rwa_components['operational_risk_rwa'] = operational_rwa
        
        # CVA Risk RWA
        cva_rwa = self._calculate_cva_risk_rwa(portfolio_data.get('derivative_exposures', {}))
        rwa_components['cva_risk_rwa'] = cva_rwa
        
        # Total RWA
        rwa_components['total_rwa'] = sum([
            sum(credit_rwa.values()),
            market_rwa,
            operational_rwa,
            cva_rwa
        ])
        
        self.rwa_components = rwa_components
        self.logger.info(f"Total RWA calculated: ${rwa_components['total_rwa']:,.0f}")
        
        return rwa_components
    
    def _calculate_credit_risk_rwa(self, credit_exposures: Dict[str, Any]) -> Dict[str, float]:
        """Calculate credit risk RWA using standardized approach."""
        credit_rwa = {}
        
        # Standardized approach risk weights
        risk_weights = {
            'sovereign': 0.0,  # US government
            'bank': 0.2,       # Banks
            'corporate': 1.0,  # Corporate exposures
            'retail': 0.75,    # Retail exposures
            'residential_mortgage': 0.5,  # Residential mortgages
            'commercial_real_estate': 1.0,  # CRE
            'other': 1.0
        }
        
        for exposure_type, exposure_data in credit_exposures.items():
            if isinstance(exposure_data, dict):
                exposure_amount = exposure_data.get('exposure', 0)
                risk_weight = risk_weights.get(exposure_type, 1.0)
                
                # Apply credit risk mitigation
                crm_factor = exposure_data.get('crm_factor', 1.0)
                
                # Calculate RWA
                rwa = exposure_amount * risk_weight * crm_factor
                credit_rwa[f'{exposure_type}_rwa'] = rwa
            else:
                # Simple exposure amount
                risk_weight = risk_weights.get(exposure_type, 1.0)
                credit_rwa[f'{exposure_type}_rwa'] = exposure_data * risk_weight
        
        return credit_rwa
    
    def _calculate_market_risk_rwa(self, market_exposures: Dict[str, Any]) -> float:
        """Calculate market risk RWA using standardized approach."""
        if not market_exposures:
            return 0.0
        
        # Standardized approach for market risk
        interest_rate_risk = market_exposures.get('interest_rate_risk', 0)
        equity_risk = market_exposures.get('equity_risk', 0)
        fx_risk = market_exposures.get('fx_risk', 0)
        commodity_risk = market_exposures.get('commodity_risk', 0)
        
        # Calculate capital requirement (8% of RWA)
        capital_requirement = (interest_rate_risk + equity_risk + fx_risk + commodity_risk)
        
        # Convert to RWA equivalent (capital requirement / 8%)
        market_rwa = capital_requirement / 0.08
        
        return market_rwa
    
    def _calculate_operational_risk_rwa(self, operational_data: Dict[str, Any]) -> float:
        """Calculate operational risk RWA using standardized approach."""
        if not operational_data:
            return 0.0
        
        # Business Indicator Component (BIC)
        bic = operational_data.get('business_indicator_component', 0)
        
        # Internal Loss Multiplier (ILM) - simplified to 1 for standardized approach
        ilm = operational_data.get('internal_loss_multiplier', 1.0)
        
        # Operational risk capital = BIC × ILM
        op_risk_capital = bic * ilm
        
        # Convert to RWA equivalent
        operational_rwa = op_risk_capital / 0.08
        
        return operational_rwa
    
    def _calculate_cva_risk_rwa(self, derivative_exposures: Dict[str, Any]) -> float:
        """Calculate CVA risk RWA."""
        if not derivative_exposures:
            return 0.0
        
        # Simplified CVA calculation
        total_derivative_exposure = derivative_exposures.get('total_exposure', 0)
        cva_capital_charge = total_derivative_exposure * 0.01  # 1% charge
        
        # Convert to RWA equivalent
        cva_rwa = cva_capital_charge / 0.08
        
        return cva_rwa
    
    def calculate_capital_ratios(self) -> Dict[str, float]:
        """Calculate Basel III capital ratios."""
        if not self.capital_components or not self.rwa_components:
            raise ValueError("Capital components and RWA must be calculated first")
        
        # Calculate capital amounts
        cet1_capital = self.capital_components.calculate_cet1()
        tier1_capital = self.capital_components.calculate_tier1()
        total_capital = self.capital_components.calculate_total_capital()
        
        # Get total RWA
        total_rwa = self.rwa_components['total_rwa']
        
        # Calculate ratios (as percentages)
        ratios = {
            'cet1_ratio': (cet1_capital / total_rwa) * 100,
            'tier1_ratio': (tier1_capital / total_rwa) * 100,
            'total_capital_ratio': (total_capital / total_rwa) * 100,
            'cet1_capital': cet1_capital,
            'tier1_capital': tier1_capital,
            'total_capital': total_capital,
            'total_rwa': total_rwa
        }
        
        self.capital_ratios = ratios
        self.logger.info(f"Capital ratios calculated - CET1: {ratios['cet1_ratio']:.2f}%")
        
        return ratios
    
    def calculate_leverage_ratio(self, balance_sheet_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate Basel III leverage ratio."""
        # Tier 1 capital
        tier1_capital = self.capital_ratios.get('tier1_capital', 0)
        
        # Leverage exposure measure
        on_balance_sheet = balance_sheet_data.get('total_assets', 0)
        derivative_exposures = balance_sheet_data.get('derivative_exposures', 0)
        securities_financing = balance_sheet_data.get('securities_financing_transactions', 0)
        off_balance_sheet = balance_sheet_data.get('off_balance_sheet_exposures', 0)
        
        # Regulatory adjustments
        regulatory_adjustments = balance_sheet_data.get('leverage_regulatory_adjustments', 0)
        
        # Total leverage exposure
        total_leverage_exposure = (on_balance_sheet + derivative_exposures + 
                                 securities_financing + off_balance_sheet - 
                                 regulatory_adjustments)
        
        # Leverage ratio (as percentage)
        leverage_ratio = (tier1_capital / total_leverage_exposure) * 100
        
        leverage_results = {
            'leverage_ratio': leverage_ratio,
            'tier1_capital': tier1_capital,
            'total_leverage_exposure': total_leverage_exposure,
            'on_balance_sheet_exposure': on_balance_sheet,
            'derivative_exposure': derivative_exposures,
            'securities_financing_exposure': securities_financing,
            'off_balance_sheet_exposure': off_balance_sheet
        }
        
        self.logger.info(f"Leverage ratio calculated: {leverage_ratio:.2f}%")
        return leverage_results
    
    def calculate_buffer_requirements(self) -> Dict[str, float]:
        """Calculate Basel III buffer requirements."""
        buffers = {
            'capital_conservation_buffer': self.requirements.capital_conservation_buffer,
            'countercyclical_buffer': self._calculate_countercyclical_buffer(),
            'gsib_surcharge': self._calculate_gsib_surcharge(),
            'dsib_surcharge': self._calculate_dsib_surcharge()
        }
        
        # Total buffer requirement
        buffers['total_buffer_requirement'] = sum(buffers.values())
        
        # Effective minimum ratios (minimum + buffers)
        buffers['effective_cet1_minimum'] = (self.requirements.cet1_minimum + 
                                           buffers['total_buffer_requirement'])
        buffers['effective_tier1_minimum'] = (self.requirements.tier1_minimum + 
                                            buffers['total_buffer_requirement'])
        buffers['effective_total_capital_minimum'] = (self.requirements.total_capital_minimum + 
                                                    buffers['total_buffer_requirement'])
        
        self.buffer_requirements = buffers
        self.logger.info(f"Total buffer requirement: {buffers['total_buffer_requirement']:.2f}%")
        
        return buffers
    
    def _calculate_countercyclical_buffer(self) -> float:
        """Calculate countercyclical capital buffer."""
        # This would typically be set by national regulators
        # For demonstration, using a simple economic indicator approach
        ccyb_rate = self.basel_config.get('countercyclical_buffer_rate', 0.0)
        
        # Ensure it doesn't exceed maximum
        return min(ccyb_rate, self.requirements.countercyclical_buffer_max)
    
    def _calculate_gsib_surcharge(self) -> float:
        """Calculate G-SIB surcharge if applicable."""
        if self.bank_type != 'gsib' or not self.gsib_bucket:
            return 0.0
        
        return self.requirements.gsib_surcharge_buckets.get(self.gsib_bucket, 0.0)
    
    def _calculate_dsib_surcharge(self) -> float:
        """Calculate D-SIB (Domestic Systemically Important Bank) surcharge."""
        # This would be set by national regulators
        return self.basel_config.get('dsib_surcharge', 0.0)
    
    def calculate_lcr(self, liquidity_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Liquidity Coverage Ratio (LCR)."""
        # High-Quality Liquid Assets (HQLA)
        level1_assets = liquidity_data.get('level1_hqla', 0)  # Cash, central bank reserves, government securities
        level2a_assets = liquidity_data.get('level2a_hqla', 0)  # Corporate bonds, covered bonds
        level2b_assets = liquidity_data.get('level2b_hqla', 0)  # Lower-quality assets
        
        # Apply haircuts to Level 2 assets
        adjusted_level2a = level2a_assets * 0.85  # 15% haircut
        adjusted_level2b = level2b_assets * 0.5   # 50% haircut
        
        # Ensure Level 2 assets don't exceed 40% of total HQLA
        total_level2 = adjusted_level2a + adjusted_level2b
        level2_cap = (level1_assets + total_level2) * 0.4
        
        if total_level2 > level2_cap:
            # Proportionally reduce Level 2 assets
            reduction_factor = level2_cap / total_level2
            adjusted_level2a *= reduction_factor
            adjusted_level2b *= reduction_factor
        
        total_hqla = level1_assets + adjusted_level2a + adjusted_level2b
        
        # Net Cash Outflows
        cash_outflows = self._calculate_lcr_outflows(liquidity_data.get('outflows', {}))
        cash_inflows = self._calculate_lcr_inflows(liquidity_data.get('inflows', {}))
        
        # Inflows capped at 75% of outflows
        capped_inflows = min(cash_inflows, cash_outflows * 0.75)
        net_cash_outflows = max(cash_outflows - capped_inflows, cash_outflows * 0.25)
        
        # LCR calculation
        lcr = (total_hqla / net_cash_outflows) * 100 if net_cash_outflows > 0 else float('inf')
        
        lcr_results = {
            'lcr': lcr,
            'total_hqla': total_hqla,
            'level1_hqla': level1_assets,
            'level2a_hqla': adjusted_level2a,
            'level2b_hqla': adjusted_level2b,
            'total_cash_outflows': cash_outflows,
            'total_cash_inflows': cash_inflows,
            'capped_inflows': capped_inflows,
            'net_cash_outflows': net_cash_outflows
        }
        
        self.logger.info(f"LCR calculated: {lcr:.2f}%")
        return lcr_results
    
    def _calculate_lcr_outflows(self, outflow_data: Dict[str, Any]) -> float:
        """Calculate LCR cash outflows."""
        outflows = 0
        
        # Retail deposits
        stable_retail = outflow_data.get('stable_retail_deposits', 0)
        less_stable_retail = outflow_data.get('less_stable_retail_deposits', 0)
        outflows += stable_retail * 0.03 + less_stable_retail * 0.10
        
        # Wholesale funding
        operational_deposits = outflow_data.get('operational_deposits', 0)
        non_operational_deposits = outflow_data.get('non_operational_deposits', 0)
        outflows += operational_deposits * 0.25 + non_operational_deposits * 1.0
        
        # Secured funding
        secured_funding = outflow_data.get('secured_funding_transactions', 0)
        outflows += secured_funding * 0.25
        
        # Derivatives and other commitments
        derivative_outflows = outflow_data.get('derivative_outflows', 0)
        credit_facilities = outflow_data.get('undrawn_credit_facilities', 0)
        outflows += derivative_outflows + credit_facilities * 0.10
        
        return outflows
    
    def _calculate_lcr_inflows(self, inflow_data: Dict[str, Any]) -> float:
        """Calculate LCR cash inflows."""
        inflows = 0
        
        # Performing loans
        performing_loans = inflow_data.get('performing_loans', 0)
        inflows += performing_loans * 0.50
        
        # Securities maturing
        maturing_securities = inflow_data.get('maturing_securities', 0)
        inflows += maturing_securities * 1.0
        
        # Committed facilities
        committed_facilities = inflow_data.get('committed_facilities', 0)
        inflows += committed_facilities * 1.0
        
        # Other inflows
        other_inflows = inflow_data.get('other_contractual_inflows', 0)
        inflows += other_inflows * 1.0
        
        return inflows
    
    def calculate_nsfr(self, funding_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Net Stable Funding Ratio (NSFR)."""
        # Available Stable Funding (ASF)
        asf = self._calculate_available_stable_funding(funding_data.get('funding_sources', {}))
        
        # Required Stable Funding (RSF)
        rsf = self._calculate_required_stable_funding(funding_data.get('assets_and_exposures', {}))
        
        # NSFR calculation
        nsfr = (asf / rsf) * 100 if rsf > 0 else float('inf')
        
        nsfr_results = {
            'nsfr': nsfr,
            'available_stable_funding': asf,
            'required_stable_funding': rsf
        }
        
        self.logger.info(f"NSFR calculated: {nsfr:.2f}%")
        return nsfr_results
    
    def _calculate_available_stable_funding(self, funding_sources: Dict[str, Any]) -> float:
        """Calculate Available Stable Funding for NSFR."""
        asf = 0
        
        # Capital and instruments with effective maturity ≥ 1 year
        capital = funding_sources.get('regulatory_capital', 0)
        long_term_instruments = funding_sources.get('long_term_instruments', 0)
        asf += (capital + long_term_instruments) * 1.0  # 100% ASF factor
        
        # Stable retail deposits
        stable_retail = funding_sources.get('stable_retail_deposits', 0)
        asf += stable_retail * 0.95  # 95% ASF factor
        
        # Less stable retail deposits
        less_stable_retail = funding_sources.get('less_stable_retail_deposits', 0)
        asf += less_stable_retail * 0.90  # 90% ASF factor
        
        # Wholesale funding with maturity ≥ 1 year
        wholesale_long_term = funding_sources.get('wholesale_funding_long_term', 0)
        asf += wholesale_long_term * 1.0  # 100% ASF factor
        
        # Wholesale funding with maturity < 1 year
        wholesale_short_term = funding_sources.get('wholesale_funding_short_term', 0)
        asf += wholesale_short_term * 0.50  # 50% ASF factor
        
        return asf
    
    def _calculate_required_stable_funding(self, assets_exposures: Dict[str, Any]) -> float:
        """Calculate Required Stable Funding for NSFR."""
        rsf = 0
        
        # Cash and short-term exposures
        cash = assets_exposures.get('cash', 0)
        short_term_exposures = assets_exposures.get('short_term_exposures', 0)
        rsf += (cash + short_term_exposures) * 0.0  # 0% RSF factor
        
        # Government securities
        government_securities = assets_exposures.get('government_securities', 0)
        rsf += government_securities * 0.05  # 5% RSF factor
        
        # Corporate bonds and covered bonds
        corporate_bonds = assets_exposures.get('corporate_bonds', 0)
        rsf += corporate_bonds * 0.20  # 20% RSF factor
        
        # Loans to financial institutions with maturity < 1 year
        fi_loans_short = assets_exposures.get('fi_loans_short_term', 0)
        rsf += fi_loans_short * 0.50  # 50% RSF factor
        
        # Retail and SME loans
        retail_loans = assets_exposures.get('retail_loans', 0)
        sme_loans = assets_exposures.get('sme_loans', 0)
        rsf += (retail_loans + sme_loans) * 0.85  # 85% RSF factor
        
        # Other loans
        other_loans = assets_exposures.get('other_loans', 0)
        rsf += other_loans * 1.0  # 100% RSF factor
        
        # Other assets
        other_assets = assets_exposures.get('other_assets', 0)
        rsf += other_assets * 1.0  # 100% RSF factor
        
        return rsf
    
    def assess_basel_compliance(self) -> Dict[str, Any]:
        """Assess overall Basel III compliance."""
        if not all([self.capital_ratios, self.buffer_requirements]):
            raise ValueError("Capital ratios and buffer requirements must be calculated first")
        
        compliance = {
            'assessment_date': datetime.now().isoformat(),
            'capital_compliance': {},
            'buffer_compliance': {},
            'liquidity_compliance': {},
            'overall_status': 'COMPLIANT',
            'deficiencies': [],
            'recommendations': []
        }
        
        # Capital ratio compliance
        cet1_ratio = self.capital_ratios['cet1_ratio']
        tier1_ratio = self.capital_ratios['tier1_ratio']
        total_capital_ratio = self.capital_ratios['total_capital_ratio']
        
        effective_cet1_min = self.buffer_requirements['effective_cet1_minimum']
        effective_tier1_min = self.buffer_requirements['effective_tier1_minimum']
        effective_total_min = self.buffer_requirements['effective_total_capital_minimum']
        
        compliance['capital_compliance'] = {
            'cet1_compliant': cet1_ratio >= effective_cet1_min,
            'tier1_compliant': tier1_ratio >= effective_tier1_min,
            'total_capital_compliant': total_capital_ratio >= effective_total_min,
            'cet1_excess': cet1_ratio - effective_cet1_min,
            'tier1_excess': tier1_ratio - effective_tier1_min,
            'total_capital_excess': total_capital_ratio - effective_total_min
        }
        
        # Check for capital deficiencies
        if not compliance['capital_compliance']['cet1_compliant']:
            compliance['deficiencies'].append(f"CET1 ratio below requirement: {cet1_ratio:.2f}% < {effective_cet1_min:.2f}%")
            compliance['overall_status'] = 'NON_COMPLIANT'
        
        if not compliance['capital_compliance']['tier1_compliant']:
            compliance['deficiencies'].append(f"Tier 1 ratio below requirement: {tier1_ratio:.2f}% < {effective_tier1_min:.2f}%")
            compliance['overall_status'] = 'NON_COMPLIANT'
        
        if not compliance['capital_compliance']['total_capital_compliant']:
            compliance['deficiencies'].append(f"Total capital ratio below requirement: {total_capital_ratio:.2f}% < {effective_total_min:.2f}%")
            compliance['overall_status'] = 'NON_COMPLIANT'
        
        # Buffer compliance assessment
        compliance['buffer_compliance'] = self._assess_buffer_compliance()
        
        # Generate recommendations
        compliance['recommendations'] = self._generate_basel_recommendations(compliance)
        
        self.compliance_results = compliance
        return compliance
    
    def _assess_buffer_compliance(self) -> Dict[str, Any]:
        """Assess capital buffer compliance and restrictions."""
        cet1_ratio = self.capital_ratios['cet1_ratio']
        minimum_cet1 = self.requirements.cet1_minimum
        buffer_requirement = self.buffer_requirements['total_buffer_requirement']
        
        # Calculate buffer level above minimum
        buffer_level = cet1_ratio - minimum_cet1
        
        # Determine restriction zone
        if buffer_level >= buffer_requirement:
            restriction_zone = 'none'
            distribution_restrictions = 0.0
        elif buffer_level >= buffer_requirement * 0.75:
            restriction_zone = 'first_quartile'
            distribution_restrictions = 0.6  # 60% of earnings retained
        elif buffer_level >= buffer_requirement * 0.5:
            restriction_zone = 'second_quartile'
            distribution_restrictions = 0.8  # 80% of earnings retained
        elif buffer_level >= buffer_requirement * 0.25:
            restriction_zone = 'third_quartile'
            distribution_restrictions = 1.0  # 100% of earnings retained
        else:
            restriction_zone = 'fourth_quartile'
            distribution_restrictions = 1.0  # 100% of earnings retained + additional restrictions
        
        return {
            'buffer_level': buffer_level,
            'buffer_requirement': buffer_requirement,
            'restriction_zone': restriction_zone,
            'distribution_restrictions': distribution_restrictions,
            'buffer_compliant': buffer_level >= buffer_requirement
        }
    
    def _generate_basel_recommendations(self, compliance: Dict[str, Any]) -> List[str]:
        """Generate Basel III compliance recommendations."""
        recommendations = []
        
        # Capital recommendations
        if not compliance['capital_compliance']['cet1_compliant']:
            recommendations.append("Increase CET1 capital through retained earnings or equity issuance")
        
        if not compliance['capital_compliance']['tier1_compliant']:
            recommendations.append("Consider Additional Tier 1 capital instruments")
        
        if not compliance['capital_compliance']['total_capital_compliant']:
            recommendations.append("Issue Tier 2 capital instruments to meet total capital requirements")
        
        # Buffer recommendations
        buffer_compliance = compliance['buffer_compliance']
        if buffer_compliance['restriction_zone'] != 'none':
            recommendations.append(f"Capital in {buffer_compliance['restriction_zone']} - consider capital conservation actions")
        
        # General recommendations
        recommendations.extend([
            "Monitor RWA optimization opportunities",
            "Enhance capital planning processes",
            "Strengthen liquidity management",
            "Regular stress testing of capital adequacy"
        ])
        
        return recommendations
    
    def project_basel_ratios_under_stress(self, stress_scenarios: Dict[str, Any],
                                        ppnr_projections: Dict[str, Any],
                                        loss_projections: Dict[str, Any]) -> Dict[str, Any]:
        """Project Basel III ratios under stress scenarios."""
        stress_projections = {}
        
        for scenario_name, scenario_data in stress_scenarios.items():
            # Get PPNR and loss projections for this scenario
            scenario_ppnr = ppnr_projections.get(scenario_name, {})
            scenario_losses = loss_projections.get(scenario_name, {})
            
            # Project capital evolution
            capital_projection = self._project_capital_under_stress(
                scenario_ppnr, scenario_losses, scenario_data
            )
            
            # Project RWA evolution
            rwa_projection = self._project_rwa_under_stress(scenario_data)
            
            # Calculate projected ratios
            projected_ratios = self._calculate_projected_ratios(
                capital_projection, rwa_projection
            )
            
            stress_projections[scenario_name] = {
                'capital_projection': capital_projection,
                'rwa_projection': rwa_projection,
                'projected_ratios': projected_ratios,
                'minimum_ratios': self._find_minimum_ratios(projected_ratios),
                'compliance_assessment': self._assess_stress_compliance(projected_ratios)
            }
        
        return stress_projections
    
    def _project_capital_under_stress(self, ppnr_projections: Dict[str, Any],
                                    loss_projections: Dict[str, Any],
                                    scenario_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Project capital evolution under stress scenario."""
        # Starting capital levels
        initial_cet1 = self.capital_ratios['cet1_capital']
        initial_tier1 = self.capital_ratios['tier1_capital']
        initial_total = self.capital_ratios['total_capital']
        
        # Initialize projections
        cet1_evolution = [initial_cet1]
        tier1_evolution = [initial_tier1]
        total_evolution = [initial_total]
        
        # Project quarterly
        quarters = scenario_data.get('projection_quarters', 9)
        dividend_payout_ratio = scenario_data.get('dividend_payout_ratio', 0.3)
        
        for quarter in range(quarters):
            # Get quarterly net income
            if isinstance(ppnr_projections, list) and quarter < len(ppnr_projections):
                quarterly_ppnr = ppnr_projections[quarter]
            else:
                quarterly_ppnr = ppnr_projections.get('quarterly_average', 0)
            
            if isinstance(loss_projections, list) and quarter < len(loss_projections):
                quarterly_losses = loss_projections[quarter]
            else:
                quarterly_losses = loss_projections.get('quarterly_average', 0)
            
            # Calculate net income
            pre_tax_income = quarterly_ppnr - quarterly_losses
            tax_rate = 0.21
            net_income = pre_tax_income * (1 - tax_rate) if pre_tax_income > 0 else pre_tax_income
            
            # Calculate retained earnings
            dividends = max(0, net_income * dividend_payout_ratio)
            retained_earnings = net_income - dividends
            
            # Update capital levels
            new_cet1 = cet1_evolution[-1] + retained_earnings
            new_tier1 = tier1_evolution[-1] + retained_earnings
            new_total = total_evolution[-1] + retained_earnings
            
            cet1_evolution.append(new_cet1)
            tier1_evolution.append(new_tier1)
            total_evolution.append(new_total)
        
        return {
            'cet1_capital': cet1_evolution,
            'tier1_capital': tier1_evolution,
            'total_capital': total_evolution
        }
    
    def _project_rwa_under_stress(self, scenario_data: Dict[str, Any]) -> List[float]:
        """Project RWA evolution under stress scenario."""
        initial_rwa = self.rwa_components['total_rwa']
        rwa_evolution = [initial_rwa]
        
        # RWA growth assumptions under stress
        quarters = scenario_data.get('projection_quarters', 9)
        rwa_growth_rate = scenario_data.get('rwa_growth_rate', 0.02)  # 2% quarterly growth
        
        for quarter in range(quarters):
            new_rwa = rwa_evolution[-1] * (1 + rwa_growth_rate)
            rwa_evolution.append(new_rwa)
        
        return rwa_evolution
    
    def _calculate_projected_ratios(self, capital_projection: Dict[str, List[float]],
                                  rwa_projection: List[float]) -> Dict[str, List[float]]:
        """Calculate projected capital ratios."""
        cet1_ratios = []
        tier1_ratios = []
        total_ratios = []
        
        for i in range(len(rwa_projection)):
            if i < len(capital_projection['cet1_capital']):
                cet1_ratio = (capital_projection['cet1_capital'][i] / rwa_projection[i]) * 100
                tier1_ratio = (capital_projection['tier1_capital'][i] / rwa_projection[i]) * 100
                total_ratio = (capital_projection['total_capital'][i] / rwa_projection[i]) * 100
                
                cet1_ratios.append(cet1_ratio)
                tier1_ratios.append(tier1_ratio)
                total_ratios.append(total_ratio)
        
        return {
            'cet1_ratios': cet1_ratios,
            'tier1_ratios': tier1_ratios,
            'total_capital_ratios': total_ratios
        }
    
    def _find_minimum_ratios(self, projected_ratios: Dict[str, List[float]]) -> Dict[str, float]:
        """Find minimum ratios during stress period."""
        return {
            'minimum_cet1': min(projected_ratios['cet1_ratios']),
            'minimum_tier1': min(projected_ratios['tier1_ratios']),
            'minimum_total_capital': min(projected_ratios['total_capital_ratios'])
        }
    
    def _assess_stress_compliance(self, projected_ratios: Dict[str, List[float]]) -> Dict[str, Any]:
        """Assess compliance under stress scenarios."""
        minimum_ratios = self._find_minimum_ratios(projected_ratios)
        
        return {
            'cet1_compliant': minimum_ratios['minimum_cet1'] >= self.buffer_requirements['effective_cet1_minimum'],
            'tier1_compliant': minimum_ratios['minimum_tier1'] >= self.buffer_requirements['effective_tier1_minimum'],
            'total_capital_compliant': minimum_ratios['minimum_total_capital'] >= self.buffer_requirements['effective_total_capital_minimum'],
            'overall_compliant': all([
                minimum_ratios['minimum_cet1'] >= self.buffer_requirements['effective_cet1_minimum'],
                minimum_ratios['minimum_tier1'] >= self.buffer_requirements['effective_tier1_minimum'],
                minimum_ratios['minimum_total_capital'] >= self.buffer_requirements['effective_total_capital_minimum']
            ])
        }
    
    def generate_basel_report(self) -> Dict[str, Any]:
        """Generate comprehensive Basel III compliance report."""
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'calculation_date': self.calculation_date.isoformat(),
                'bank_type': self.bank_type,
                'jurisdiction': self.jurisdiction
            },
            'capital_adequacy': {
                'capital_ratios': self.capital_ratios,
                'buffer_requirements': self.buffer_requirements,
                'compliance_assessment': self.compliance_results
            },
            'risk_weighted_assets': self.rwa_components,
            'regulatory_requirements': {
                'minimum_ratios': {
                    'cet1_minimum': self.requirements.cet1_minimum,
                    'tier1_minimum': self.requirements.tier1_minimum,
                    'total_capital_minimum': self.requirements.total_capital_minimum
                },
                'buffer_requirements': self.buffer_requirements
            },
            'recommendations': self._generate_comprehensive_recommendations()
        }
        
        return report
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive Basel III recommendations."""
        recommendations = []
        
        if self.compliance_results:
            recommendations.extend(self.compliance_results.get('recommendations', []))
        
        # Add general Basel III recommendations
        recommendations.extend([
            "Maintain robust capital planning processes",
            "Regular monitoring of regulatory developments",
            "Optimize balance sheet composition for capital efficiency",
            "Enhance stress testing capabilities",
            "Strengthen liquidity risk management"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def export_basel_results(self, filepath: str, export_type: str = 'summary') -> None:
        """Export Basel III results to file."""
        if export_type == 'full':
            report = self.generate_basel_report()
        else:
            report = {
                'capital_ratios': self.capital_ratios,
                'compliance_status': self.compliance_results.get('overall_status', 'UNKNOWN'),
                'buffer_requirements': self.buffer_requirements
            }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            # Export as CSV summary
            self._export_basel_csv_summary(filepath)
        
        self.logger.info(f"Basel III results exported to {filepath}")
    
    def _export_basel_csv_summary(self, filepath: str) -> None:
        """Export Basel III summary as CSV."""
        summary_data = [{
            'cet1_ratio': self.capital_ratios.get('cet1_ratio', 0),
            'tier1_ratio': self.capital_ratios.get('tier1_ratio', 0),
            'total_capital_ratio': self.capital_ratios.get('total_capital_ratio', 0),
            'total_rwa': self.capital_ratios.get('total_rwa', 0),
            'buffer_requirement': self.buffer_requirements.get('total_buffer_requirement', 0),
            'compliance_status': self.compliance_results.get('overall_status', 'UNKNOWN')
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)