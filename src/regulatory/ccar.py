"""
CCAR (Comprehensive Capital Analysis and Review) Module

This module implements CCAR stress testing framework and capital planning
requirements for the PPNR risk models system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StressScenario(Enum):
    """CCAR stress test scenarios"""
    BASELINE = "baseline"
    ADVERSE = "adverse"
    SEVERELY_ADVERSE = "severely_adverse"


class CapitalAction(Enum):
    """Types of capital actions"""
    DIVIDEND_PAYMENT = "dividend_payment"
    SHARE_REPURCHASE = "share_repurchase"
    CAPITAL_ISSUANCE = "capital_issuance"
    CAPITAL_REDEMPTION = "capital_redemption"


@dataclass
class StressTestResults:
    """Results from CCAR stress testing"""
    scenario: str
    min_cet1_ratio: float
    min_tier1_ratio: float
    min_total_capital_ratio: float
    min_leverage_ratio: float
    projected_losses: Dict[str, float]
    capital_trajectory: pd.DataFrame
    pass_status: bool


@dataclass
class CapitalPlan:
    """Bank's capital plan submission"""
    planning_horizon: int  # years
    baseline_projections: Dict
    stress_projections: Dict
    planned_actions: List[Dict]
    capital_targets: Dict
    risk_appetite: Dict


class CCARStressTester:
    """
    CCAR Stress Testing Framework
    
    Implements comprehensive capital analysis and review requirements including:
    - Stress scenario modeling
    - Capital projection under stress
    - Loss forecasting
    - Capital action planning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CCAR stress tester
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # CCAR minimum ratios (including stress capital buffer)
        self.minimum_ratios = {
            'cet1_ratio': 0.045,  # 4.5% minimum
            'tier1_ratio': 0.06,  # 6.0% minimum
            'total_capital_ratio': 0.08,  # 8.0% minimum
            'leverage_ratio': 0.04  # 4.0% minimum (enhanced)
        }
        
        # Stress capital buffer requirements
        self.stress_capital_buffer = 0.025  # 2.5% additional buffer
        
        # Default stress scenarios (simplified)
        self.stress_scenarios = {
            StressScenario.BASELINE: {
                'gdp_growth': [2.5, 2.3, 2.1],
                'unemployment_rate': [4.0, 4.2, 4.1],
                'house_price_growth': [3.0, 2.5, 2.0],
                'equity_market_decline': [0.0, 0.0, 0.0],
                'interest_rate_change': [0.5, 0.75, 1.0]
            },
            StressScenario.ADVERSE: {
                'gdp_growth': [-1.0, -0.5, 1.0],
                'unemployment_rate': [6.5, 7.0, 6.5],
                'house_price_growth': [-5.0, -3.0, 0.0],
                'equity_market_decline': [-25.0, -10.0, 0.0],
                'interest_rate_change': [-1.0, -0.5, 0.0]
            },
            StressScenario.SEVERELY_ADVERSE: {
                'gdp_growth': [-3.5, -2.0, 1.5],
                'unemployment_rate': [10.0, 9.5, 8.0],
                'house_price_growth': [-15.0, -10.0, -5.0],
                'equity_market_decline': [-50.0, -20.0, 10.0],
                'interest_rate_change': [-2.0, -1.0, 0.5]
            }
        }
    
    def run_stress_test(self,
                       portfolio_data: pd.DataFrame,
                       capital_data: Dict,
                       scenario: StressScenario,
                       projection_quarters: int = 9) -> StressTestResults:
        """
        Run CCAR stress test for specified scenario
        
        Args:
            portfolio_data: Bank's portfolio data
            capital_data: Current capital position
            scenario: Stress scenario to apply
            projection_quarters: Number of quarters to project
            
        Returns:
            StressTestResults with comprehensive results
        """
        try:
            logger.info(f"Running CCAR stress test for {scenario.value} scenario")
            
            # Get scenario parameters
            scenario_params = self.stress_scenarios[scenario]
            
            # Project losses under stress
            projected_losses = self._project_stress_losses(
                portfolio_data, scenario_params, projection_quarters
            )
            
            # Project capital trajectory
            capital_trajectory = self._project_capital_trajectory(
                capital_data, projected_losses, projection_quarters
            )
            
            # Calculate minimum ratios during stress period
            min_ratios = self._calculate_minimum_ratios(capital_trajectory)
            
            # Determine pass/fail status
            pass_status = self._evaluate_pass_status(min_ratios)
            
            return StressTestResults(
                scenario=scenario.value,
                min_cet1_ratio=min_ratios['cet1_ratio'],
                min_tier1_ratio=min_ratios['tier1_ratio'],
                min_total_capital_ratio=min_ratios['total_capital_ratio'],
                min_leverage_ratio=min_ratios['leverage_ratio'],
                projected_losses=projected_losses,
                capital_trajectory=capital_trajectory,
                pass_status=pass_status
            )
            
        except Exception as e:
            logger.error(f"Error running stress test: {str(e)}")
            raise
    
    def _project_stress_losses(self,
                             portfolio_data: pd.DataFrame,
                             scenario_params: Dict,
                             quarters: int) -> Dict[str, float]:
        """Project losses under stress scenario"""
        losses = {
            'credit_losses': 0.0,
            'trading_losses': 0.0,
            'operational_losses': 0.0,
            'other_losses': 0.0
        }
        
        # Credit losses based on portfolio composition and stress factors
        if 'loan_portfolio' in portfolio_data.columns:
            base_loss_rate = 0.02  # 2% base loss rate
            
            # Adjust for stress scenario
            gdp_impact = max(0, -scenario_params['gdp_growth'][0] * 0.5)
            unemployment_impact = (scenario_params['unemployment_rate'][0] - 4.0) * 0.3
            house_price_impact = max(0, -scenario_params['house_price_growth'][0] * 0.2)
            
            stress_multiplier = 1 + gdp_impact + unemployment_impact + house_price_impact
            
            total_loans = portfolio_data.get('loan_portfolio', pd.Series([0])).sum()
            losses['credit_losses'] = total_loans * base_loss_rate * stress_multiplier
        
        # Trading losses from market stress
        if 'trading_assets' in portfolio_data.columns:
            equity_decline = scenario_params['equity_market_decline'][0] / 100
            trading_assets = portfolio_data.get('trading_assets', pd.Series([0])).sum()
            losses['trading_losses'] = trading_assets * abs(equity_decline) * 0.7
        
        # Operational losses (simplified)
        total_assets = portfolio_data.sum().sum() if not portfolio_data.empty else 1000000
        losses['operational_losses'] = total_assets * 0.001  # 0.1% of assets
        
        return losses
    
    def _project_capital_trajectory(self,
                                  capital_data: Dict,
                                  projected_losses: Dict,
                                  quarters: int) -> pd.DataFrame:
        """Project capital levels over stress period"""
        
        # Initialize trajectory DataFrame
        trajectory = pd.DataFrame(index=range(quarters + 1))
        
        # Starting capital levels
        trajectory.loc[0, 'cet1_capital'] = capital_data.get('cet1_capital', 0)
        trajectory.loc[0, 'tier1_capital'] = capital_data.get('tier1_capital', 0)
        trajectory.loc[0, 'total_capital'] = capital_data.get('total_capital', 0)
        trajectory.loc[0, 'total_rwa'] = capital_data.get('total_rwa', 1)
        trajectory.loc[0, 'total_exposure'] = capital_data.get('total_exposure', 1)
        
        # Project forward quarter by quarter
        for q in range(1, quarters + 1):
            prev_quarter = trajectory.loc[q-1]
            
            # Apply losses (distributed over quarters)
            quarterly_credit_loss = projected_losses['credit_losses'] / quarters
            quarterly_trading_loss = projected_losses['trading_losses'] / quarters
            quarterly_op_loss = projected_losses['operational_losses'] / quarters
            
            total_quarterly_loss = (quarterly_credit_loss + 
                                  quarterly_trading_loss + 
                                  quarterly_op_loss)
            
            # Update capital (losses reduce CET1 first)
            new_cet1 = prev_quarter['cet1_capital'] - total_quarterly_loss
            new_tier1 = max(new_cet1, prev_quarter['tier1_capital'] - total_quarterly_loss)
            new_total = max(new_tier1, prev_quarter['total_capital'] - total_quarterly_loss)
            
            # Assume RWA grows modestly under stress
            rwa_growth = 1.02 if q <= quarters/2 else 1.01
            new_rwa = prev_quarter['total_rwa'] * rwa_growth
            new_exposure = prev_quarter['total_exposure'] * rwa_growth
            
            # Store projections
            trajectory.loc[q, 'cet1_capital'] = max(0, new_cet1)
            trajectory.loc[q, 'tier1_capital'] = max(0, new_tier1)
            trajectory.loc[q, 'total_capital'] = max(0, new_total)
            trajectory.loc[q, 'total_rwa'] = new_rwa
            trajectory.loc[q, 'total_exposure'] = new_exposure
            
            # Calculate ratios
            trajectory.loc[q, 'cet1_ratio'] = trajectory.loc[q, 'cet1_capital'] / new_rwa
            trajectory.loc[q, 'tier1_ratio'] = trajectory.loc[q, 'tier1_capital'] / new_rwa
            trajectory.loc[q, 'total_capital_ratio'] = trajectory.loc[q, 'total_capital'] / new_rwa
            trajectory.loc[q, 'leverage_ratio'] = trajectory.loc[q, 'tier1_capital'] / new_exposure
        
        return trajectory
    
    def _calculate_minimum_ratios(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        """Calculate minimum ratios during stress period"""
        return {
            'cet1_ratio': trajectory['cet1_ratio'].min(),
            'tier1_ratio': trajectory['tier1_ratio'].min(),
            'total_capital_ratio': trajectory['total_capital_ratio'].min(),
            'leverage_ratio': trajectory['leverage_ratio'].min()
        }
    
    def _evaluate_pass_status(self, min_ratios: Dict[str, float]) -> bool:
        """Evaluate whether bank passes stress test"""
        return (
            min_ratios['cet1_ratio'] >= self.minimum_ratios['cet1_ratio'] and
            min_ratios['tier1_ratio'] >= self.minimum_ratios['tier1_ratio'] and
            min_ratios['total_capital_ratio'] >= self.minimum_ratios['total_capital_ratio'] and
            min_ratios['leverage_ratio'] >= self.minimum_ratios['leverage_ratio']
        )
    
    def evaluate_capital_plan(self,
                            capital_plan: CapitalPlan,
                            stress_results: Dict[str, StressTestResults]) -> Dict:
        """
        Evaluate bank's capital plan against CCAR requirements
        
        Args:
            capital_plan: Bank's submitted capital plan
            stress_results: Results from all stress scenarios
            
        Returns:
            Evaluation results and recommendations
        """
        try:
            evaluation = {
                'overall_assessment': 'PASS',
                'scenario_results': {},
                'capital_adequacy': True,
                'plan_credibility': True,
                'recommendations': []
            }
            
            # Evaluate each scenario
            for scenario_name, results in stress_results.items():
                scenario_eval = {
                    'pass_status': results.pass_status,
                    'min_cet1_ratio': f"{results.min_cet1_ratio:.2%}",
                    'buffer_above_minimum': results.min_cet1_ratio - self.minimum_ratios['cet1_ratio']
                }
                
                if not results.pass_status:
                    evaluation['overall_assessment'] = 'CONDITIONAL PASS' if scenario_name != 'severely_adverse' else 'FAIL'
                    evaluation['capital_adequacy'] = False
                    evaluation['recommendations'].append(
                        f"Strengthen capital position for {scenario_name} scenario"
                    )
                
                evaluation['scenario_results'][scenario_name] = scenario_eval
            
            # Evaluate planned capital actions
            action_evaluation = self._evaluate_capital_actions(capital_plan.planned_actions)
            evaluation['capital_actions'] = action_evaluation
            
            # Check plan credibility
            credibility_check = self._assess_plan_credibility(capital_plan)
            evaluation['plan_credibility'] = credibility_check['credible']
            evaluation['credibility_issues'] = credibility_check.get('issues', [])
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating capital plan: {str(e)}")
            raise
    
    def _evaluate_capital_actions(self, planned_actions: List[Dict]) -> Dict:
        """Evaluate planned capital actions"""
        evaluation = {
            'total_dividends': 0,
            'total_repurchases': 0,
            'net_issuances': 0,
            'payout_ratio': 0,
            'action_assessment': 'APPROPRIATE'
        }
        
        for action in planned_actions:
            action_type = action.get('type')
            amount = action.get('amount', 0)
            
            if action_type == CapitalAction.DIVIDEND_PAYMENT.value:
                evaluation['total_dividends'] += amount
            elif action_type == CapitalAction.SHARE_REPURCHASE.value:
                evaluation['total_repurchases'] += amount
            elif action_type == CapitalAction.CAPITAL_ISSUANCE.value:
                evaluation['net_issuances'] += amount
            elif action_type == CapitalAction.CAPITAL_REDEMPTION.value:
                evaluation['net_issuances'] -= amount
        
        # Calculate payout ratio (simplified)
        total_payouts = evaluation['total_dividends'] + evaluation['total_repurchases']
        estimated_earnings = 1000000  # Placeholder - would use actual projections
        evaluation['payout_ratio'] = total_payouts / estimated_earnings if estimated_earnings > 0 else 0
        
        # Assess appropriateness
        if evaluation['payout_ratio'] > 0.8:
            evaluation['action_assessment'] = 'EXCESSIVE'
        elif evaluation['payout_ratio'] > 0.6:
            evaluation['action_assessment'] = 'ELEVATED'
        
        return evaluation
    
    def _assess_plan_credibility(self, capital_plan: CapitalPlan) -> Dict:
        """Assess credibility of capital plan assumptions"""
        issues = []
        
        # Check planning horizon
        if capital_plan.planning_horizon < 2:
            issues.append("Planning horizon too short for adequate stress testing")
        
        # Check baseline assumptions (simplified)
        baseline = capital_plan.baseline_projections
        if baseline.get('revenue_growth', 0) > 0.15:
            issues.append("Revenue growth assumptions appear optimistic")
        
        if baseline.get('loss_rate', 0.05) < 0.01:
            issues.append("Loss rate assumptions appear optimistic")
        
        # Check stress assumptions
        stress = capital_plan.stress_projections
        if stress.get('loss_multiplier', 2.0) < 1.5:
            issues.append("Stress loss assumptions appear insufficient")
        
        return {
            'credible': len(issues) == 0,
            'issues': issues
        }
    
    def generate_ccar_report(self,
                           stress_results: Dict[str, StressTestResults],
                           capital_plan_evaluation: Dict) -> Dict:
        """
        Generate comprehensive CCAR report
        
        Args:
            stress_results: Results from all stress scenarios
            capital_plan_evaluation: Capital plan evaluation results
            
        Returns:
            Comprehensive CCAR report
        """
        report = {
            'executive_summary': {
                'overall_result': capital_plan_evaluation['overall_assessment'],
                'capital_adequacy': capital_plan_evaluation['capital_adequacy'],
                'key_findings': []
            },
            'stress_test_results': {},
            'capital_plan_assessment': capital_plan_evaluation,
            'regulatory_requirements': {
                'minimum_ratios': self.minimum_ratios,
                'stress_capital_buffer': self.stress_capital_buffer
            },
            'recommendations': capital_plan_evaluation.get('recommendations', [])
        }
        
        # Add stress test results
        for scenario_name, results in stress_results.items():
            report['stress_test_results'][scenario_name] = {
                'pass_status': 'PASS' if results.pass_status else 'FAIL',
                'minimum_cet1_ratio': f"{results.min_cet1_ratio:.2%}",
                'minimum_tier1_ratio': f"{results.min_tier1_ratio:.2%}",
                'minimum_total_capital_ratio': f"{results.min_total_capital_ratio:.2%}",
                'minimum_leverage_ratio': f"{results.min_leverage_ratio:.2%}",
                'projected_losses': {
                    k: f"${v:,.0f}" for k, v in results.projected_losses.items()
                }
            }
        
        # Add key findings
        if capital_plan_evaluation['overall_assessment'] == 'PASS':
            report['executive_summary']['key_findings'].append(
                "Bank demonstrates adequate capital strength under all stress scenarios"
            )
        else:
            report['executive_summary']['key_findings'].append(
                "Bank requires capital strengthening to meet stress requirements"
            )
        
        return report
    
    def calculate_stress_capital_buffer(self, stress_results: StressTestResults) -> float:
        """
        Calculate stress capital buffer based on stress test results
        
        Args:
            stress_results: Results from severely adverse scenario
            
        Returns:
            Required stress capital buffer
        """
        # Stress capital buffer is based on losses in severely adverse scenario
        # Simplified calculation - actual methodology is more complex
        
        total_losses = sum(stress_results.projected_losses.values())
        # Assume starting capital of $10B for calculation
        starting_capital = 10_000_000_000
        
        loss_rate = total_losses / starting_capital if starting_capital > 0 else 0
        
        # Buffer should cover losses plus maintain minimum ratios
        buffer = max(self.stress_capital_buffer, loss_rate + 0.01)
        
        return min(buffer, 0.10)  # Cap at 10%


class CCARReporter:
    """
    CCAR reporting and documentation generator.
    
    Generates comprehensive CCAR reports including:
    - Stress test results summaries
    - Capital plan assessments
    - Regulatory compliance documentation
    - Executive summaries
    """
    
    def __init__(self):
        """Initialize CCAR reporter."""
        self.report_templates = {
            'executive_summary': self._get_executive_template(),
            'detailed_results': self._get_detailed_template(),
            'regulatory_submission': self._get_regulatory_template()
        }
    
    def generate_comprehensive_report(self, 
                                    stress_results: Dict[str, StressTestResults],
                                    capital_plan: CapitalPlan,
                                    assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive CCAR report.
        
        Args:
            stress_results: Results from all stress scenarios
            capital_plan: Bank's capital plan
            assessment_results: Capital plan assessment results
            
        Returns:
            Comprehensive CCAR report
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'executive_summary': self._generate_executive_summary(
                stress_results, capital_plan, assessment_results
            ),
            'stress_test_results': self._format_stress_results(stress_results),
            'capital_plan_assessment': assessment_results,
            'regulatory_compliance': self._assess_regulatory_compliance(
                stress_results, assessment_results
            ),
            'recommendations': self._generate_recommendations(
                stress_results, assessment_results
            )
        }
        
        return report
    
    def _generate_executive_summary(self, 
                                  stress_results: Dict[str, StressTestResults],
                                  capital_plan: CapitalPlan,
                                  assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary section."""
        return {
            'overall_assessment': assessment_results.get('overall_assessment', 'PENDING'),
            'capital_adequacy': assessment_results.get('capital_adequacy', 'ADEQUATE'),
            'key_findings': [
                f"Stress testing conducted across {len(stress_results)} scenarios",
                f"Capital plan includes {len(capital_plan.planned_actions)} planned actions",
                f"Assessment result: {assessment_results.get('overall_assessment', 'PENDING')}"
            ],
            'next_steps': assessment_results.get('recommendations', [])
        }
    
    def _format_stress_results(self, stress_results: Dict[str, StressTestResults]) -> Dict[str, Any]:
        """Format stress test results for reporting."""
        formatted_results = {}
        
        for scenario_name, results in stress_results.items():
            formatted_results[scenario_name] = {
                'pass_status': 'PASS' if results.pass_status else 'FAIL',
                'capital_ratios': {
                    'min_cet1_ratio': f"{results.min_cet1_ratio:.2%}",
                    'min_tier1_ratio': f"{results.min_tier1_ratio:.2%}",
                    'min_total_capital_ratio': f"{results.min_total_capital_ratio:.2%}",
                    'min_leverage_ratio': f"{results.min_leverage_ratio:.2%}"
                },
                'projected_losses': {
                    category: f"${amount:,.0f}" 
                    for category, amount in results.projected_losses.items()
                },
                'total_losses': f"${sum(results.projected_losses.values()):,.0f}"
            }
        
        return formatted_results
    
    def _assess_regulatory_compliance(self, 
                                    stress_results: Dict[str, StressTestResults],
                                    assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess regulatory compliance status."""
        compliance_status = {
            'overall_compliance': 'COMPLIANT',
            'stress_test_compliance': True,
            'capital_plan_compliance': True,
            'issues': []
        }
        
        # Check stress test compliance
        for scenario_name, results in stress_results.items():
            if not results.pass_status:
                compliance_status['stress_test_compliance'] = False
                compliance_status['issues'].append(
                    f"Failed stress test under {scenario_name} scenario"
                )
        
        # Check capital plan compliance
        if assessment_results.get('overall_assessment') != 'PASS':
            compliance_status['capital_plan_compliance'] = False
            compliance_status['issues'].append("Capital plan assessment failed")
        
        if compliance_status['issues']:
            compliance_status['overall_compliance'] = 'NON_COMPLIANT'
        
        return compliance_status
    
    def _generate_recommendations(self, 
                                stress_results: Dict[str, StressTestResults],
                                assessment_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Check for failed scenarios
        failed_scenarios = [
            name for name, results in stress_results.items() 
            if not results.pass_status
        ]
        
        if failed_scenarios:
            recommendations.append(
                f"Address capital deficiencies identified in {', '.join(failed_scenarios)} scenarios"
            )
        
        # Add assessment-based recommendations
        if 'recommendations' in assessment_results:
            recommendations.extend(assessment_results['recommendations'])
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Continue monitoring capital adequacy",
                "Review and update capital planning processes",
                "Maintain strong risk management practices"
            ])
        
        return recommendations
    
    def _get_executive_template(self) -> str:
        """Get executive summary template."""
        return "CCAR Executive Summary Template"
    
    def _get_detailed_template(self) -> str:
        """Get detailed results template."""
        return "CCAR Detailed Results Template"
    
    def _get_regulatory_template(self) -> str:
        """Get regulatory submission template."""
        return "CCAR Regulatory Submission Template"