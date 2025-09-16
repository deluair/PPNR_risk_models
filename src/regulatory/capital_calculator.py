"""
Capital Calculator Module

Unified capital calculation engine for regulatory compliance:
- Basel III capital ratio calculations
- CCAR capital planning and projections
- DFAST capital adequacy assessments
- Stress testing capital impact analysis
- Buffer requirement calculations
- Capital action planning and optimization
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

class CapitalTier(Enum):
    """Capital tier classifications."""
    CET1 = "Common Equity Tier 1"
    TIER1 = "Tier 1"
    TIER2 = "Tier 2"
    TOTAL = "Total Capital"

class RiskCategory(Enum):
    """Risk-weighted asset categories."""
    CREDIT = "Credit Risk"
    MARKET = "Market Risk"
    OPERATIONAL = "Operational Risk"
    CVA = "Credit Valuation Adjustment"

@dataclass
class CapitalComponent:
    """Individual capital component."""
    name: str
    amount: float
    tier: CapitalTier
    regulatory_adjustments: float = 0.0
    phase_in_amount: float = 0.0
    
    @property
    def net_amount(self) -> float:
        """Net amount after regulatory adjustments."""
        return self.amount - self.regulatory_adjustments + self.phase_in_amount

@dataclass
class RWAComponent:
    """Risk-weighted asset component."""
    category: RiskCategory
    exposure: float
    risk_weight: float
    rwa_amount: float
    
    def __post_init__(self):
        if self.rwa_amount == 0:
            self.rwa_amount = self.exposure * self.risk_weight

@dataclass
class CapitalRequirement:
    """Capital requirement specification."""
    name: str
    minimum_ratio: float
    buffer_requirement: float = 0.0
    applicable_capital: CapitalTier = CapitalTier.CET1
    
    @property
    def total_requirement(self) -> float:
        """Total requirement including buffers."""
        return self.minimum_ratio + self.buffer_requirement

@dataclass
class StressScenarioImpact:
    """Capital impact under stress scenario."""
    scenario_name: str
    revenue_impact: float
    loss_impact: float
    rwa_impact: float
    capital_impact: float
    
    @property
    def net_impact(self) -> float:
        """Net impact on capital."""
        return self.revenue_impact - self.loss_impact - self.capital_impact

class CapitalCalculator:
    """
    Unified capital calculation engine for regulatory compliance.
    
    Features:
    - Multi-framework capital calculations (Basel III, CCAR, DFAST)
    - Dynamic capital planning and optimization
    - Stress testing capital impact analysis
    - Buffer requirement calculations
    - Capital action scenario modeling
    - Regulatory reporting preparation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize capital calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.capital_config = config.get('capital_calculation', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.CapitalCalculator")
        
        # Initialize capital components
        self.capital_components: List[CapitalComponent] = []
        self.rwa_components: List[RWAComponent] = []
        
        # Regulatory requirements
        self.regulatory_requirements = self._initialize_regulatory_requirements()
        
        # Calculation history
        self.calculation_history = []
        
        # Current capital position
        self.current_position = {}
        
        self.logger.info("Capital calculator initialized")
    
    def _initialize_regulatory_requirements(self) -> Dict[str, CapitalRequirement]:
        """Initialize regulatory capital requirements."""
        requirements = {
            # Basel III requirements
            'basel_cet1_minimum': CapitalRequirement(
                name='Basel III CET1 Minimum',
                minimum_ratio=4.5,
                applicable_capital=CapitalTier.CET1
            ),
            'basel_tier1_minimum': CapitalRequirement(
                name='Basel III Tier 1 Minimum',
                minimum_ratio=6.0,
                applicable_capital=CapitalTier.TIER1
            ),
            'basel_total_minimum': CapitalRequirement(
                name='Basel III Total Capital Minimum',
                minimum_ratio=8.0,
                applicable_capital=CapitalTier.TOTAL
            ),
            
            # Basel III buffers
            'capital_conservation_buffer': CapitalRequirement(
                name='Capital Conservation Buffer',
                minimum_ratio=0.0,
                buffer_requirement=2.5,
                applicable_capital=CapitalTier.CET1
            ),
            'countercyclical_buffer': CapitalRequirement(
                name='Countercyclical Buffer',
                minimum_ratio=0.0,
                buffer_requirement=0.0,  # Variable based on jurisdiction
                applicable_capital=CapitalTier.CET1
            ),
            
            # CCAR requirements
            'ccar_cet1_minimum': CapitalRequirement(
                name='CCAR CET1 Minimum',
                minimum_ratio=4.5,
                applicable_capital=CapitalTier.CET1
            ),
            'ccar_tier1_minimum': CapitalRequirement(
                name='CCAR Tier 1 Minimum',
                minimum_ratio=6.0,
                applicable_capital=CapitalTier.TIER1
            ),
            'ccar_total_minimum': CapitalRequirement(
                name='CCAR Total Capital Minimum',
                minimum_ratio=8.0,
                applicable_capital=CapitalTier.TOTAL
            ),
            
            # Leverage ratio
            'leverage_ratio': CapitalRequirement(
                name='Leverage Ratio',
                minimum_ratio=4.0,  # For G-SIBs
                applicable_capital=CapitalTier.TIER1
            )
        }
        
        # Load custom requirements from config
        custom_requirements = self.capital_config.get('custom_requirements', {})
        for name, req_config in custom_requirements.items():
            requirements[name] = CapitalRequirement(
                name=req_config.get('name', name),
                minimum_ratio=req_config.get('minimum_ratio', 0.0),
                buffer_requirement=req_config.get('buffer_requirement', 0.0),
                applicable_capital=CapitalTier(req_config.get('applicable_capital', 'CET1'))
            )
        
        return requirements
    
    def add_capital_component(self, component: CapitalComponent) -> None:
        """Add a capital component."""
        self.capital_components.append(component)
        self.logger.info(f"Added capital component: {component.name}")
    
    def add_rwa_component(self, component: RWAComponent) -> None:
        """Add a risk-weighted asset component."""
        self.rwa_components.append(component)
        self.logger.info(f"Added RWA component: {component.category.value}")
    
    def calculate_capital_ratios(self, as_of_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate all capital ratios.
        
        Args:
            as_of_date: Calculation date
            
        Returns:
            Capital ratio calculations
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        self.logger.info(f"Calculating capital ratios as of {as_of_date.date()}")
        
        # Calculate capital amounts by tier
        capital_amounts = self._calculate_capital_amounts()
        
        # Calculate total RWA
        total_rwa = self._calculate_total_rwa()
        
        # Calculate leverage exposure
        leverage_exposure = self._calculate_leverage_exposure()
        
        # Calculate ratios
        ratios = {
            'calculation_date': as_of_date.isoformat(),
            'capital_amounts': capital_amounts,
            'total_rwa': total_rwa,
            'leverage_exposure': leverage_exposure,
            'capital_ratios': {},
            'regulatory_compliance': {},
            'buffer_analysis': {}
        }
        
        # Calculate capital ratios
        if total_rwa > 0:
            ratios['capital_ratios'] = {
                'cet1_ratio': (capital_amounts['cet1'] / total_rwa) * 100,
                'tier1_ratio': (capital_amounts['tier1'] / total_rwa) * 100,
                'total_capital_ratio': (capital_amounts['total'] / total_rwa) * 100
            }
        
        # Calculate leverage ratio
        if leverage_exposure > 0:
            ratios['capital_ratios']['leverage_ratio'] = (capital_amounts['tier1'] / leverage_exposure) * 100
        
        # Assess regulatory compliance
        ratios['regulatory_compliance'] = self._assess_regulatory_compliance(ratios['capital_ratios'])
        
        # Buffer analysis
        ratios['buffer_analysis'] = self._analyze_capital_buffers(ratios['capital_ratios'])
        
        # Store calculation
        self.current_position = ratios
        self.calculation_history.append(ratios)
        
        self.logger.info("Capital ratio calculation completed")
        return ratios
    
    def _calculate_capital_amounts(self) -> Dict[str, float]:
        """Calculate capital amounts by tier."""
        amounts = {
            'cet1': 0.0,
            'additional_tier1': 0.0,
            'tier1': 0.0,
            'tier2': 0.0,
            'total': 0.0
        }
        
        for component in self.capital_components:
            net_amount = component.net_amount
            
            if component.tier == CapitalTier.CET1:
                amounts['cet1'] += net_amount
            elif component.tier == CapitalTier.TIER1:
                amounts['additional_tier1'] += net_amount
            elif component.tier == CapitalTier.TIER2:
                amounts['tier2'] += net_amount
        
        # Calculate derived amounts
        amounts['tier1'] = amounts['cet1'] + amounts['additional_tier1']
        amounts['total'] = amounts['tier1'] + amounts['tier2']
        
        return amounts
    
    def _calculate_total_rwa(self) -> float:
        """Calculate total risk-weighted assets."""
        total_rwa = 0.0
        
        for component in self.rwa_components:
            total_rwa += component.rwa_amount
        
        return total_rwa
    
    def _calculate_leverage_exposure(self) -> float:
        """Calculate leverage exposure for leverage ratio."""
        # This would typically include on-balance sheet assets,
        # derivatives exposure, securities financing transactions, etc.
        # For now, use a simplified calculation
        
        leverage_exposure = 0.0
        
        # Add on-balance sheet assets (simplified)
        for component in self.rwa_components:
            if component.category == RiskCategory.CREDIT:
                leverage_exposure += component.exposure
        
        # Add derivatives exposure (would need additional data)
        derivatives_exposure = self.capital_config.get('derivatives_exposure', 0.0)
        leverage_exposure += derivatives_exposure
        
        # Add securities financing transactions (would need additional data)
        sft_exposure = self.capital_config.get('sft_exposure', 0.0)
        leverage_exposure += sft_exposure
        
        return leverage_exposure
    
    def _assess_regulatory_compliance(self, capital_ratios: Dict[str, float]) -> Dict[str, Any]:
        """Assess compliance with regulatory requirements."""
        compliance = {
            'overall_status': 'COMPLIANT',
            'requirement_compliance': {},
            'violations': [],
            'warnings': []
        }
        
        for req_name, requirement in self.regulatory_requirements.items():
            # Get applicable ratio
            if requirement.applicable_capital == CapitalTier.CET1:
                current_ratio = capital_ratios.get('cet1_ratio', 0.0)
            elif requirement.applicable_capital == CapitalTier.TIER1:
                if req_name == 'leverage_ratio':
                    current_ratio = capital_ratios.get('leverage_ratio', 0.0)
                else:
                    current_ratio = capital_ratios.get('tier1_ratio', 0.0)
            elif requirement.applicable_capital == CapitalTier.TOTAL:
                current_ratio = capital_ratios.get('total_capital_ratio', 0.0)
            else:
                continue
            
            # Check compliance
            required_ratio = requirement.total_requirement
            is_compliant = current_ratio >= required_ratio
            
            compliance['requirement_compliance'][req_name] = {
                'requirement_name': requirement.name,
                'required_ratio': required_ratio,
                'current_ratio': current_ratio,
                'excess_ratio': current_ratio - required_ratio,
                'compliant': is_compliant
            }
            
            if not is_compliant:
                compliance['overall_status'] = 'NON_COMPLIANT'
                compliance['violations'].append({
                    'requirement': requirement.name,
                    'shortfall': required_ratio - current_ratio
                })
            elif current_ratio < required_ratio + 1.0:  # Warning threshold
                compliance['warnings'].append({
                    'requirement': requirement.name,
                    'buffer': current_ratio - required_ratio
                })
        
        return compliance
    
    def _analyze_capital_buffers(self, capital_ratios: Dict[str, float]) -> Dict[str, Any]:
        """Analyze capital buffer positions."""
        buffer_analysis = {
            'total_buffers': {},
            'buffer_utilization': {},
            'buffer_adequacy': {}
        }
        
        # Calculate total buffer requirements
        cet1_ratio = capital_ratios.get('cet1_ratio', 0.0)
        
        # Basel III buffers
        conservation_buffer = 2.5
        countercyclical_buffer = self.capital_config.get('countercyclical_buffer', 0.0)
        gsib_buffer = self.capital_config.get('gsib_buffer', 0.0)
        
        total_buffer_requirement = conservation_buffer + countercyclical_buffer + gsib_buffer
        minimum_plus_buffers = 4.5 + total_buffer_requirement
        
        buffer_analysis['total_buffers'] = {
            'conservation_buffer': conservation_buffer,
            'countercyclical_buffer': countercyclical_buffer,
            'gsib_buffer': gsib_buffer,
            'total_buffer_requirement': total_buffer_requirement,
            'minimum_plus_buffers': minimum_plus_buffers
        }
        
        # Buffer utilization
        available_buffer = max(0, cet1_ratio - 4.5)  # Above minimum
        buffer_utilization = max(0, total_buffer_requirement - available_buffer)
        
        buffer_analysis['buffer_utilization'] = {
            'available_buffer': available_buffer,
            'required_buffer': total_buffer_requirement,
            'buffer_utilization': buffer_utilization,
            'utilization_ratio': buffer_utilization / total_buffer_requirement if total_buffer_requirement > 0 else 0
        }
        
        # Buffer adequacy assessment
        if cet1_ratio >= minimum_plus_buffers:
            adequacy_status = 'ADEQUATE'
        elif cet1_ratio >= 4.5 + conservation_buffer:
            adequacy_status = 'MARGINAL'
        else:
            adequacy_status = 'INADEQUATE'
        
        buffer_analysis['buffer_adequacy'] = {
            'status': adequacy_status,
            'excess_above_minimum': cet1_ratio - 4.5,
            'excess_above_buffers': cet1_ratio - minimum_plus_buffers
        }
        
        return buffer_analysis
    
    def project_capital_under_stress(self, stress_scenarios: List[StressScenarioImpact],
                                   projection_quarters: int = 9) -> Dict[str, Any]:
        """
        Project capital ratios under stress scenarios.
        
        Args:
            stress_scenarios: List of stress scenario impacts
            projection_quarters: Number of quarters to project
            
        Returns:
            Capital projections under stress
        """
        self.logger.info(f"Projecting capital under {len(stress_scenarios)} stress scenarios")
        
        projections = {
            'base_position': self.current_position,
            'scenario_projections': {},
            'minimum_ratios': {},
            'capital_actions_needed': {}
        }
        
        for scenario in stress_scenarios:
            scenario_projection = self._project_single_scenario(scenario, projection_quarters)
            projections['scenario_projections'][scenario.scenario_name] = scenario_projection
        
        # Calculate minimum ratios across scenarios
        projections['minimum_ratios'] = self._calculate_minimum_ratios(projections['scenario_projections'])
        
        # Assess capital actions needed
        projections['capital_actions_needed'] = self._assess_capital_actions_needed(projections['minimum_ratios'])
        
        self.logger.info("Capital stress projections completed")
        return projections
    
    def _project_single_scenario(self, scenario: StressScenarioImpact,
                               projection_quarters: int) -> Dict[str, Any]:
        """Project capital under a single stress scenario."""
        # Start with current position
        current_capital = self.current_position['capital_amounts']
        current_rwa = self.current_position['total_rwa']
        
        projection = {
            'scenario_name': scenario.scenario_name,
            'quarterly_projections': [],
            'minimum_ratios': {},
            'capital_depletion': {}
        }
        
        # Project quarterly
        for quarter in range(1, projection_quarters + 1):
            # Apply scenario impacts (simplified linear application)
            impact_factor = quarter / projection_quarters
            
            # Calculate impacted capital
            impacted_capital = {
                'cet1': current_capital['cet1'] + (scenario.net_impact * impact_factor),
                'tier1': current_capital['tier1'] + (scenario.net_impact * impact_factor),
                'total': current_capital['total'] + (scenario.net_impact * impact_factor)
            }
            
            # Calculate impacted RWA
            impacted_rwa = current_rwa + (scenario.rwa_impact * impact_factor)
            
            # Calculate ratios
            if impacted_rwa > 0:
                quarter_ratios = {
                    'quarter': quarter,
                    'cet1_ratio': (impacted_capital['cet1'] / impacted_rwa) * 100,
                    'tier1_ratio': (impacted_capital['tier1'] / impacted_rwa) * 100,
                    'total_capital_ratio': (impacted_capital['total'] / impacted_rwa) * 100,
                    'capital_amounts': impacted_capital,
                    'rwa': impacted_rwa
                }
            else:
                quarter_ratios = {
                    'quarter': quarter,
                    'cet1_ratio': 0.0,
                    'tier1_ratio': 0.0,
                    'total_capital_ratio': 0.0,
                    'capital_amounts': impacted_capital,
                    'rwa': impacted_rwa
                }
            
            projection['quarterly_projections'].append(quarter_ratios)
        
        # Calculate minimum ratios over projection period
        if projection['quarterly_projections']:
            projection['minimum_ratios'] = {
                'cet1_ratio': min(q['cet1_ratio'] for q in projection['quarterly_projections']),
                'tier1_ratio': min(q['tier1_ratio'] for q in projection['quarterly_projections']),
                'total_capital_ratio': min(q['total_capital_ratio'] for q in projection['quarterly_projections'])
            }
        
        return projection
    
    def _calculate_minimum_ratios(self, scenario_projections: Dict[str, Any]) -> Dict[str, float]:
        """Calculate minimum ratios across all scenarios."""
        minimum_ratios = {
            'cet1_ratio': float('inf'),
            'tier1_ratio': float('inf'),
            'total_capital_ratio': float('inf'),
            'worst_scenario': {}
        }
        
        for scenario_name, projection in scenario_projections.items():
            scenario_minimums = projection.get('minimum_ratios', {})
            
            for ratio_name in ['cet1_ratio', 'tier1_ratio', 'total_capital_ratio']:
                scenario_min = scenario_minimums.get(ratio_name, float('inf'))
                if scenario_min < minimum_ratios[ratio_name]:
                    minimum_ratios[ratio_name] = scenario_min
                    minimum_ratios['worst_scenario'][ratio_name] = scenario_name
        
        return minimum_ratios
    
    def _assess_capital_actions_needed(self, minimum_ratios: Dict[str, float]) -> Dict[str, Any]:
        """Assess capital actions needed based on minimum ratios."""
        actions_needed = {
            'actions_required': False,
            'priority_actions': [],
            'capital_shortfall': {},
            'recommended_actions': []
        }
        
        # Check against regulatory minimums
        regulatory_minimums = {
            'cet1_ratio': 4.5,
            'tier1_ratio': 6.0,
            'total_capital_ratio': 8.0
        }
        
        for ratio_name, minimum_ratio in minimum_ratios.items():
            if ratio_name in regulatory_minimums:
                required_minimum = regulatory_minimums[ratio_name]
                
                if minimum_ratio < required_minimum:
                    actions_needed['actions_required'] = True
                    shortfall = required_minimum - minimum_ratio
                    
                    actions_needed['capital_shortfall'][ratio_name] = {
                        'minimum_projected': minimum_ratio,
                        'regulatory_minimum': required_minimum,
                        'shortfall': shortfall,
                        'worst_scenario': minimum_ratios['worst_scenario'].get(ratio_name, 'Unknown')
                    }
                    
                    actions_needed['priority_actions'].append({
                        'ratio': ratio_name,
                        'action': f'Address {ratio_name} shortfall of {shortfall:.2f}%',
                        'priority': 'HIGH'
                    })
        
        # Recommended actions
        if actions_needed['actions_required']:
            actions_needed['recommended_actions'] = [
                'Consider capital raising (equity issuance)',
                'Reduce risk-weighted assets',
                'Optimize capital structure',
                'Implement capital conservation measures',
                'Review dividend and share repurchase policies'
            ]
        else:
            actions_needed['recommended_actions'] = [
                'Monitor capital ratios regularly',
                'Maintain capital planning discipline',
                'Consider proactive capital management'
            ]
        
        return actions_needed
    
    def optimize_capital_structure(self, target_ratios: Dict[str, float],
                                 constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize capital structure to meet target ratios.
        
        Args:
            target_ratios: Target capital ratios
            constraints: Optimization constraints
            
        Returns:
            Capital optimization recommendations
        """
        if constraints is None:
            constraints = {}
        
        self.logger.info("Optimizing capital structure")
        
        optimization = {
            'current_position': self.current_position,
            'target_ratios': target_ratios,
            'optimization_actions': [],
            'projected_outcome': {},
            'cost_benefit_analysis': {}
        }
        
        # Current ratios
        current_ratios = self.current_position.get('capital_ratios', {})
        
        # Calculate gaps
        ratio_gaps = {}
        for ratio_name, target_ratio in target_ratios.items():
            current_ratio = current_ratios.get(ratio_name, 0.0)
            ratio_gaps[ratio_name] = target_ratio - current_ratio
        
        # Generate optimization actions
        optimization['optimization_actions'] = self._generate_optimization_actions(ratio_gaps, constraints)
        
        # Project outcome
        optimization['projected_outcome'] = self._project_optimization_outcome(
            optimization['optimization_actions']
        )
        
        # Cost-benefit analysis
        optimization['cost_benefit_analysis'] = self._analyze_optimization_costs(
            optimization['optimization_actions']
        )
        
        self.logger.info("Capital structure optimization completed")
        return optimization
    
    def _generate_optimization_actions(self, ratio_gaps: Dict[str, float],
                                     constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate capital optimization actions."""
        actions = []
        
        for ratio_name, gap in ratio_gaps.items():
            if gap > 0:  # Need to increase ratio
                if ratio_name in ['cet1_ratio', 'tier1_ratio', 'total_capital_ratio']:
                    # Capital increase actions
                    actions.extend([
                        {
                            'action_type': 'capital_raise',
                            'description': f'Raise CET1 capital to improve {ratio_name}',
                            'target_ratio': ratio_name,
                            'estimated_amount': self._estimate_capital_needed(ratio_name, gap),
                            'implementation_timeline': '3-6 months',
                            'regulatory_approval_needed': True
                        },
                        {
                            'action_type': 'rwa_reduction',
                            'description': f'Reduce RWA to improve {ratio_name}',
                            'target_ratio': ratio_name,
                            'estimated_rwa_reduction': self._estimate_rwa_reduction_needed(ratio_name, gap),
                            'implementation_timeline': '1-3 months',
                            'regulatory_approval_needed': False
                        }
                    ])
                elif ratio_name == 'leverage_ratio':
                    # Leverage ratio specific actions
                    actions.append({
                        'action_type': 'leverage_reduction',
                        'description': 'Reduce leverage exposure',
                        'target_ratio': ratio_name,
                        'estimated_exposure_reduction': self._estimate_leverage_reduction_needed(gap),
                        'implementation_timeline': '1-2 months',
                        'regulatory_approval_needed': False
                    })
        
        # Apply constraints
        max_capital_raise = constraints.get('max_capital_raise', float('inf'))
        max_rwa_reduction = constraints.get('max_rwa_reduction', float('inf'))
        
        # Filter actions based on constraints
        filtered_actions = []
        for action in actions:
            if action['action_type'] == 'capital_raise':
                if action.get('estimated_amount', 0) <= max_capital_raise:
                    filtered_actions.append(action)
            elif action['action_type'] == 'rwa_reduction':
                if action.get('estimated_rwa_reduction', 0) <= max_rwa_reduction:
                    filtered_actions.append(action)
            else:
                filtered_actions.append(action)
        
        return filtered_actions
    
    def _estimate_capital_needed(self, ratio_name: str, gap: float) -> float:
        """Estimate capital needed to close ratio gap."""
        current_rwa = self.current_position.get('total_rwa', 0)
        
        if current_rwa > 0:
            # Simplified calculation: gap% * RWA / 100
            return (gap * current_rwa) / 100
        
        return 0.0
    
    def _estimate_rwa_reduction_needed(self, ratio_name: str, gap: float) -> float:
        """Estimate RWA reduction needed to close ratio gap."""
        current_capital = self.current_position.get('capital_amounts', {})
        current_rwa = self.current_position.get('total_rwa', 0)
        
        if ratio_name == 'cet1_ratio':
            capital_amount = current_capital.get('cet1', 0)
        elif ratio_name == 'tier1_ratio':
            capital_amount = current_capital.get('tier1', 0)
        else:
            capital_amount = current_capital.get('total', 0)
        
        if capital_amount > 0 and current_rwa > 0:
            current_ratio = (capital_amount / current_rwa) * 100
            target_ratio = current_ratio + gap
            
            # Calculate required RWA: capital / (target_ratio / 100)
            required_rwa = capital_amount / (target_ratio / 100)
            rwa_reduction = current_rwa - required_rwa
            
            return max(0, rwa_reduction)
        
        return 0.0
    
    def _estimate_leverage_reduction_needed(self, gap: float) -> float:
        """Estimate leverage exposure reduction needed."""
        current_capital = self.current_position.get('capital_amounts', {}).get('tier1', 0)
        current_exposure = self.current_position.get('leverage_exposure', 0)
        
        if current_capital > 0 and current_exposure > 0:
            current_ratio = (current_capital / current_exposure) * 100
            target_ratio = current_ratio + gap
            
            # Calculate required exposure: capital / (target_ratio / 100)
            required_exposure = current_capital / (target_ratio / 100)
            exposure_reduction = current_exposure - required_exposure
            
            return max(0, exposure_reduction)
        
        return 0.0
    
    def _project_optimization_outcome(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Project outcome of optimization actions."""
        # This would simulate the impact of proposed actions
        # For now, return a simplified projection
        
        outcome = {
            'projected_ratios': {},
            'implementation_timeline': '6-12 months',
            'success_probability': 0.85,
            'key_risks': [
                'Market conditions may affect capital raising',
                'Regulatory approval timeline uncertainty',
                'Business impact of RWA reduction'
            ]
        }
        
        return outcome
    
    def _analyze_optimization_costs(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze costs and benefits of optimization actions."""
        cost_analysis = {
            'total_estimated_cost': 0.0,
            'cost_breakdown': {},
            'benefits': [],
            'net_benefit': 0.0,
            'payback_period': '12-18 months'
        }
        
        for action in actions:
            if action['action_type'] == 'capital_raise':
                # Estimate cost of capital raising
                amount = action.get('estimated_amount', 0)
                cost = amount * 0.05  # 5% cost assumption
                cost_analysis['cost_breakdown']['capital_raise'] = cost
                cost_analysis['total_estimated_cost'] += cost
            elif action['action_type'] == 'rwa_reduction':
                # Estimate cost of RWA reduction (opportunity cost)
                rwa_reduction = action.get('estimated_rwa_reduction', 0)
                cost = rwa_reduction * 0.02  # 2% opportunity cost assumption
                cost_analysis['cost_breakdown']['rwa_reduction'] = cost
                cost_analysis['total_estimated_cost'] += cost
        
        # Benefits
        cost_analysis['benefits'] = [
            'Improved regulatory compliance',
            'Enhanced financial stability',
            'Better credit ratings',
            'Increased investor confidence',
            'Reduced regulatory scrutiny'
        ]
        
        # Simplified net benefit calculation
        cost_analysis['net_benefit'] = -cost_analysis['total_estimated_cost']  # Costs are negative
        
        return cost_analysis
    
    def generate_capital_report(self) -> Dict[str, Any]:
        """Generate comprehensive capital report."""
        self.logger.info("Generating capital report")
        
        report = {
            'report_metadata': {
                'generation_date': datetime.now().isoformat(),
                'report_type': 'Capital Analysis Report',
                'version': '1.0'
            },
            'executive_summary': {},
            'current_position': self.current_position,
            'regulatory_compliance': {},
            'capital_planning': {},
            'recommendations': []
        }
        
        # Executive summary
        if self.current_position:
            capital_ratios = self.current_position.get('capital_ratios', {})
            compliance = self.current_position.get('regulatory_compliance', {})
            
            report['executive_summary'] = {
                'overall_capital_strength': 'STRONG' if compliance.get('overall_status') == 'COMPLIANT' else 'WEAK',
                'key_ratios': {
                    'cet1_ratio': capital_ratios.get('cet1_ratio', 0.0),
                    'tier1_ratio': capital_ratios.get('tier1_ratio', 0.0),
                    'total_capital_ratio': capital_ratios.get('total_capital_ratio', 0.0),
                    'leverage_ratio': capital_ratios.get('leverage_ratio', 0.0)
                },
                'compliance_status': compliance.get('overall_status', 'UNKNOWN'),
                'key_findings': self._generate_key_findings()
            }
        
        # Regulatory compliance summary
        if self.current_position:
            report['regulatory_compliance'] = self.current_position.get('regulatory_compliance', {})
        
        # Capital planning insights
        report['capital_planning'] = {
            'planning_horizon': '3 years',
            'key_considerations': [
                'Regulatory requirement changes',
                'Business growth plans',
                'Market conditions',
                'Stress testing results'
            ],
            'strategic_priorities': [
                'Maintain strong capital ratios',
                'Optimize capital efficiency',
                'Prepare for regulatory changes',
                'Support business growth'
            ]
        }
        
        # Recommendations
        report['recommendations'] = self._generate_capital_recommendations()
        
        self.logger.info("Capital report generated")
        return report
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings for capital report."""
        findings = []
        
        if self.current_position:
            capital_ratios = self.current_position.get('capital_ratios', {})
            compliance = self.current_position.get('regulatory_compliance', {})
            
            # Capital strength findings
            cet1_ratio = capital_ratios.get('cet1_ratio', 0.0)
            if cet1_ratio >= 12.0:
                findings.append("Strong CET1 capital position well above regulatory minimums")
            elif cet1_ratio >= 8.0:
                findings.append("Adequate CET1 capital position above regulatory minimums")
            else:
                findings.append("CET1 capital position requires attention")
            
            # Compliance findings
            if compliance.get('overall_status') == 'COMPLIANT':
                findings.append("All regulatory capital requirements met")
            else:
                violations = compliance.get('violations', [])
                findings.append(f"Regulatory violations identified: {len(violations)} requirements")
            
            # Buffer findings
            buffer_analysis = self.current_position.get('buffer_analysis', {})
            buffer_adequacy = buffer_analysis.get('buffer_adequacy', {})
            if buffer_adequacy.get('status') == 'ADEQUATE':
                findings.append("Capital buffers are adequate for stress scenarios")
            else:
                findings.append("Capital buffer adequacy requires monitoring")
        
        return findings
    
    def _generate_capital_recommendations(self) -> List[Dict[str, Any]]:
        """Generate capital management recommendations."""
        recommendations = []
        
        if self.current_position:
            compliance = self.current_position.get('regulatory_compliance', {})
            
            # Compliance-based recommendations
            if compliance.get('overall_status') != 'COMPLIANT':
                recommendations.append({
                    'category': 'Regulatory Compliance',
                    'priority': 'HIGH',
                    'recommendation': 'Address regulatory capital violations immediately',
                    'timeline': '1-3 months'
                })
            
            # Buffer-based recommendations
            buffer_analysis = self.current_position.get('buffer_analysis', {})
            if buffer_analysis.get('buffer_adequacy', {}).get('status') != 'ADEQUATE':
                recommendations.append({
                    'category': 'Capital Buffers',
                    'priority': 'MEDIUM',
                    'recommendation': 'Strengthen capital buffers for stress resilience',
                    'timeline': '3-6 months'
                })
            
            # General recommendations
            recommendations.extend([
                {
                    'category': 'Capital Planning',
                    'priority': 'MEDIUM',
                    'recommendation': 'Develop comprehensive capital planning framework',
                    'timeline': '6-12 months'
                },
                {
                    'category': 'Optimization',
                    'priority': 'LOW',
                    'recommendation': 'Optimize capital structure for efficiency',
                    'timeline': '12+ months'
                }
            ])
        
        return recommendations