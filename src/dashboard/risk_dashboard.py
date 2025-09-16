"""
Risk Dashboard Module

Comprehensive risk monitoring dashboard for PPNR models:
- Real-time risk metrics display
- Interactive portfolio analysis
- Risk factor monitoring
- Alert and notification system
- Drill-down capabilities
- Multi-timeframe analysis
- Risk attribution views
- Regulatory compliance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import warnings

class DashboardTheme(Enum):
    """Dashboard theme options."""
    LIGHT = "light"
    DARK = "dark"
    CORPORATE = "corporate"
    REGULATORY = "regulatory"

class TimeFrame(Enum):
    """Time frame options for analysis."""
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"
    QUARTERLY = "3M"
    YEARLY = "1Y"
    ALL = "ALL"

class RiskMetricType(Enum):
    """Risk metric types for display."""
    VAR = "Value at Risk"
    EXPECTED_SHORTFALL = "Expected Shortfall"
    EXPECTED_LOSS = "Expected Loss"
    ECONOMIC_CAPITAL = "Economic Capital"
    REGULATORY_CAPITAL = "Regulatory Capital"
    CONCENTRATION_RISK = "Concentration Risk"

@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    theme: DashboardTheme = DashboardTheme.CORPORATE
    auto_refresh_interval: int = 30  # seconds
    default_timeframe: TimeFrame = TimeFrame.MONTHLY
    confidence_levels: List[float] = None
    risk_metrics: List[RiskMetricType] = None
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99, 0.999]
        if self.risk_metrics is None:
            self.risk_metrics = [RiskMetricType.VAR, RiskMetricType.EXPECTED_SHORTFALL, RiskMetricType.ECONOMIC_CAPITAL]
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'var_99_threshold': 1000000,  # $1M
                'concentration_threshold': 0.25,  # 25%
                'correlation_threshold': 0.8  # 80%
            }

@dataclass
class AlertMessage:
    """Alert message structure."""
    alert_id: str
    timestamp: datetime
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    category: str
    message: str
    metric_value: float
    threshold_value: float
    portfolio_id: Optional[str] = None

class RiskDashboard:
    """
    Comprehensive risk monitoring dashboard.
    
    Features:
    - Real-time risk metrics visualization
    - Interactive portfolio analysis
    - Risk factor monitoring and alerts
    - Multi-timeframe analysis capabilities
    - Drill-down functionality
    - Regulatory compliance tracking
    - Executive summary views
    - Export and reporting capabilities
    """
    
    def __init__(self, config: DashboardConfig = None):
        """
        Initialize risk dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.RiskDashboard")
        
        # Dashboard state
        self.app = None
        self.data_sources = {}
        self.current_data = {}
        self.alerts = []
        
        # Risk models (to be injected)
        self.credit_risk_model = None
        self.market_risk_model = None
        self.operational_risk_model = None
        self.risk_integration_model = None
        
        # Cache for performance
        self.chart_cache = {}
        self.data_cache = {}
        self.cache_timestamp = {}
        
        self.logger.info("Risk dashboard initialized")
    
    def register_risk_models(self, credit_model=None, market_model=None, 
                           operational_model=None, integration_model=None):
        """
        Register risk models for dashboard integration.
        
        Args:
            credit_model: Credit risk model instance
            market_model: Market risk model instance
            operational_model: Operational risk model instance
            integration_model: Risk integration model instance
        """
        self.logger.info("Registering risk models with dashboard")
        
        if credit_model:
            self.credit_risk_model = credit_model
        if market_model:
            self.market_risk_model = market_model
        if operational_model:
            self.operational_risk_model = operational_model
        if integration_model:
            self.risk_integration_model = integration_model
    
    def initialize_dashboard(self, port: int = 8050, debug: bool = False) -> dash.Dash:
        """
        Initialize Dash application.
        
        Args:
            port: Port number for dashboard
            debug: Enable debug mode
            
        Returns:
            Dash application instance
        """
        self.logger.info(f"Initializing dashboard on port {port}")
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Set app title
        self.app.title = "PPNR Risk Dashboard"
        
        # Create layout
        self.app.layout = self._create_main_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        self.logger.info("Dashboard initialized successfully")
        return self.app
    
    def _create_main_layout(self) -> html.Div:
        """Create main dashboard layout."""
        return html.Div([
            # Header
            self._create_header(),
            
            # Navigation tabs
            dbc.Tabs([
                dbc.Tab(label="Risk Overview", tab_id="risk-overview"),
                dbc.Tab(label="Portfolio Analysis", tab_id="portfolio-analysis"),
                dbc.Tab(label="Stress Testing", tab_id="stress-testing"),
                dbc.Tab(label="Regulatory Compliance", tab_id="regulatory-compliance"),
                dbc.Tab(label="Performance Metrics", tab_id="performance-metrics"),
                dbc.Tab(label="Alerts & Monitoring", tab_id="alerts-monitoring")
            ], id="main-tabs", active_tab="risk-overview"),
            
            # Main content area
            html.Div(id="tab-content", className="mt-3"),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.config.auto_refresh_interval * 1000,  # milliseconds
                n_intervals=0
            ),
            
            # Store components for data sharing
            dcc.Store(id='risk-data-store'),
            dcc.Store(id='portfolio-data-store'),
            dcc.Store(id='alerts-store')
        ])
    
    def _create_header(self) -> dbc.Navbar:
        """Create dashboard header."""
        return dbc.Navbar([
            dbc.Row([
                dbc.Col([
                    html.Img(src="/assets/logo.png", height="40px", className="me-2"),
                    dbc.NavbarBrand("PPNR Risk Dashboard", className="ms-2")
                ], width="auto"),
                dbc.Col([
                    html.Div([
                        html.Span("Last Updated: ", className="text-muted"),
                        html.Span(id="last-update-time", className="fw-bold")
                    ])
                ], width="auto"),
                dbc.Col([
                    dbc.Button(
                        "Refresh Data",
                        id="refresh-button",
                        color="primary",
                        size="sm",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Export Report",
                        id="export-button",
                        color="secondary",
                        size="sm"
                    )
                ], width="auto")
            ], className="w-100", justify="between")
        ], color="dark", dark=True, className="mb-3")
    
    def _register_callbacks(self):
        """Register dashboard callbacks."""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'active_tab')
        )
        def render_tab_content(active_tab):
            """Render content based on active tab."""
            if active_tab == "risk-overview":
                return self._create_risk_overview_tab()
            elif active_tab == "portfolio-analysis":
                return self._create_portfolio_analysis_tab()
            elif active_tab == "stress-testing":
                return self._create_stress_testing_tab()
            elif active_tab == "regulatory-compliance":
                return self._create_regulatory_compliance_tab()
            elif active_tab == "performance-metrics":
                return self._create_performance_metrics_tab()
            elif active_tab == "alerts-monitoring":
                return self._create_alerts_monitoring_tab()
            else:
                return html.Div("Tab content not found")
        
        @self.app.callback(
            [Output('risk-data-store', 'data'),
             Output('last-update-time', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_risk_data(n_intervals, n_clicks):
            """Update risk data periodically or on refresh."""
            try:
                # Fetch latest risk data
                risk_data = self._fetch_risk_data()
                
                # Update timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return risk_data, timestamp
            
            except Exception as e:
                self.logger.error(f"Error updating risk data: {e}")
                return {}, "Error updating data"
        
        @self.app.callback(
            Output('alerts-store', 'data'),
            Input('risk-data-store', 'data')
        )
        def update_alerts(risk_data):
            """Update alerts based on risk data."""
            if not risk_data:
                return []
            
            try:
                alerts = self._check_risk_alerts(risk_data)
                return [asdict(alert) for alert in alerts]
            
            except Exception as e:
                self.logger.error(f"Error updating alerts: {e}")
                return []
    
    def _create_risk_overview_tab(self) -> html.Div:
        """Create risk overview tab content."""
        return html.Div([
            dbc.Row([
                # Key risk metrics cards
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Portfolio VaR (99%)", className="card-title"),
                            html.H2(id="portfolio-var-99", className="text-primary"),
                            html.P("vs. Previous Period", className="text-muted"),
                            html.Span(id="var-change", className="badge")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Expected Shortfall", className="card-title"),
                            html.H2(id="expected-shortfall", className="text-warning"),
                            html.P("99% Confidence Level", className="text-muted"),
                            html.Span(id="es-change", className="badge")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Economic Capital", className="card-title"),
                            html.H2(id="economic-capital", className="text-info"),
                            html.P("Total Allocation", className="text-muted"),
                            html.Span(id="ec-change", className="badge")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Utilization", className="card-title"),
                            html.H2(id="risk-utilization", className="text-success"),
                            html.P("% of Risk Appetite", className="text-muted"),
                            html.Span(id="utilization-status", className="badge")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                # Risk trend chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics Trend"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-trend-chart")
                        ])
                    ])
                ], width=8),
                
                # Risk breakdown pie chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Type Breakdown"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-breakdown-chart")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                # Risk factor heatmap
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Factor Correlation Heatmap"),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-heatmap")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def _create_portfolio_analysis_tab(self) -> html.Div:
        """Create portfolio analysis tab content."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Selection"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="portfolio-selector",
                                placeholder="Select Portfolio(s)",
                                multi=True
                            ),
                            html.Hr(),
                            dcc.Dropdown(
                                id="timeframe-selector",
                                options=[
                                    {"label": tf.value, "value": tf.name} 
                                    for tf in TimeFrame
                                ],
                                value=self.config.default_timeframe.name,
                                placeholder="Select Timeframe"
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Risk Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="portfolio-risk-chart")
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Attribution"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-attribution-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Concentration Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="concentration-chart")
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_stress_testing_tab(self) -> html.Div:
        """Create stress testing tab content."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stress Test Configuration"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="scenario-selector",
                                placeholder="Select Stress Scenario",
                                multi=True
                            ),
                            html.Hr(),
                            dbc.Button(
                                "Run Stress Test",
                                id="run-stress-test",
                                color="primary",
                                className="w-100"
                            )
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stress Test Results"),
                        dbc.CardBody([
                            dcc.Graph(id="stress-test-results-chart")
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Scenario Impact Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="scenario-impact-chart")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Factor Sensitivity"),
                        dbc.CardBody([
                            dcc.Graph(id="sensitivity-chart")
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def _create_regulatory_compliance_tab(self) -> html.Div:
        """Create regulatory compliance tab content."""
        return html.Div([
            dbc.Row([
                # Compliance status cards
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("CCAR Compliance", className="card-title"),
                            html.H2(id="ccar-status", className="text-success"),
                            html.P("Capital Adequacy Ratio", className="text-muted")
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("DFAST Status", className="card-title"),
                            html.H2(id="dfast-status", className="text-success"),
                            html.P("Stress Test Results", className="text-muted")
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Basel III Ratios", className="card-title"),
                            html.H2(id="basel-ratios", className="text-success"),
                            html.P("Capital & Liquidity", className="text-muted")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Regulatory Capital Trends"),
                        dbc.CardBody([
                            dcc.Graph(id="regulatory-capital-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Compliance Reports"),
                        dbc.CardBody([
                            html.Div(id="compliance-reports-table")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def _create_performance_metrics_tab(self) -> html.Div:
        """Create performance metrics tab content."""
        return html.Div([
            dbc.Row([
                # Performance KPI cards
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("RAROC", className="card-title"),
                            html.H2(id="raroc-metric", className="text-primary"),
                            html.P("Risk-Adjusted Return", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Economic Value Added", className="card-title"),
                            html.H2(id="eva-metric", className="text-success"),
                            html.P("Value Creation", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sharpe Ratio", className="card-title"),
                            html.H2(id="sharpe-metric", className="text-info"),
                            html.P("Risk-Adjusted Performance", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Risk Efficiency", className="card-title"),
                            html.H2(id="risk-efficiency-metric", className="text-warning"),
                            html.P("Revenue per Risk Unit", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Trends"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-trends-chart")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk-Return Scatter"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-return-scatter")
                        ])
                    ])
                ], width=4)
            ])
        ])
    
    def _create_alerts_monitoring_tab(self) -> html.Div:
        """Create alerts and monitoring tab content."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Alerts"),
                        dbc.CardBody([
                            html.Div(id="active-alerts-list")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alert Configuration"),
                        dbc.CardBody([
                            html.Div(id="alert-config-form")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alert History"),
                        dbc.CardBody([
                            dcc.Graph(id="alert-history-chart")
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def _fetch_risk_data(self) -> Dict[str, Any]:
        """Fetch latest risk data from models."""
        risk_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': {},
            'risk_factors': {},
            'correlations': {},
            'stress_results': {},
            'regulatory_metrics': {},
            'performance_metrics': {}
        }
        
        try:
            # Fetch portfolio risk metrics
            if self.risk_integration_model and hasattr(self.risk_integration_model, 'portfolio_risks'):
                risk_data['portfolio_metrics'] = {
                    portfolio_id: {
                        'var_95': portfolio.var_95,
                        'var_99': portfolio.var_99,
                        'var_999': portfolio.var_999,
                        'expected_shortfall_95': portfolio.expected_shortfall_95,
                        'expected_shortfall_99': portfolio.expected_shortfall_99,
                        'expected_loss': portfolio.expected_loss,
                        'unexpected_loss': portfolio.unexpected_loss,
                        'diversification_benefit': portfolio.diversification_benefit
                    }
                    for portfolio_id, portfolio in self.risk_integration_model.portfolio_risks.items()
                }
            
            # Fetch risk factor data
            if self.risk_integration_model and hasattr(self.risk_integration_model, 'risk_factors'):
                risk_data['risk_factors'] = {
                    factor_id: {
                        'current_value': factor.current_value,
                        'volatility': factor.volatility,
                        'risk_type': factor.risk_type.value
                    }
                    for factor_id, factor in self.risk_integration_model.risk_factors.items()
                }
            
            # Fetch correlation matrix
            if self.risk_integration_model and hasattr(self.risk_integration_model, 'correlation_matrix'):
                if not self.risk_integration_model.correlation_matrix.empty:
                    risk_data['correlations'] = self.risk_integration_model.correlation_matrix.to_dict()
            
            # Fetch stress test results
            if self.risk_integration_model and hasattr(self.risk_integration_model, 'stress_test_results'):
                risk_data['stress_results'] = self.risk_integration_model.stress_test_results
            
            # Fetch economic capital
            if self.risk_integration_model and hasattr(self.risk_integration_model, 'economic_capital'):
                risk_data['economic_capital'] = self.risk_integration_model.economic_capital
            
            # Fetch performance metrics
            if self.risk_integration_model and hasattr(self.risk_integration_model, 'risk_adjusted_metrics'):
                risk_data['performance_metrics'] = self.risk_integration_model.risk_adjusted_metrics
        
        except Exception as e:
            self.logger.error(f"Error fetching risk data: {e}")
        
        return risk_data
    
    def _check_risk_alerts(self, risk_data: Dict[str, Any]) -> List[AlertMessage]:
        """Check for risk alerts based on current data."""
        alerts = []
        
        try:
            # Check VaR thresholds
            portfolio_metrics = risk_data.get('portfolio_metrics', {})
            
            for portfolio_id, metrics in portfolio_metrics.items():
                var_99 = metrics.get('var_99', 0.0)
                threshold = self.config.alert_thresholds.get('var_99_threshold', 1000000)
                
                if var_99 > threshold:
                    alert = AlertMessage(
                        alert_id=f"var_alert_{portfolio_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        timestamp=datetime.now(),
                        severity='HIGH',
                        category='VaR Breach',
                        message=f"Portfolio {portfolio_id} VaR exceeds threshold",
                        metric_value=var_99,
                        threshold_value=threshold,
                        portfolio_id=portfolio_id
                    )
                    alerts.append(alert)
            
            # Check correlation alerts
            correlations = risk_data.get('correlations', {})
            correlation_threshold = self.config.alert_thresholds.get('correlation_threshold', 0.8)
            
            for factor1, corr_dict in correlations.items():
                if isinstance(corr_dict, dict):
                    for factor2, correlation in corr_dict.items():
                        if factor1 != factor2 and abs(correlation) > correlation_threshold:
                            alert = AlertMessage(
                                alert_id=f"corr_alert_{factor1}_{factor2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                timestamp=datetime.now(),
                                severity='MEDIUM',
                                category='High Correlation',
                                message=f"High correlation detected between {factor1} and {factor2}",
                                metric_value=abs(correlation),
                                threshold_value=correlation_threshold
                            )
                            alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
        
        return alerts
    
    def create_risk_trend_chart(self, risk_data: Dict[str, Any]) -> go.Figure:
        """Create risk metrics trend chart."""
        fig = go.Figure()
        
        try:
            # Sample data for demonstration
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
            
            # Add VaR trend
            var_values = np.random.normal(1000000, 100000, len(dates))
            fig.add_trace(go.Scatter(
                x=dates,
                y=var_values,
                mode='lines+markers',
                name='VaR 99%',
                line=dict(color='red', width=2)
            ))
            
            # Add Expected Shortfall trend
            es_values = var_values * 1.2  # ES typically higher than VaR
            fig.add_trace(go.Scatter(
                x=dates,
                y=es_values,
                mode='lines+markers',
                name='Expected Shortfall 99%',
                line=dict(color='orange', width=2)
            ))
            
            # Add Economic Capital trend
            ec_values = var_values * 0.8
            fig.add_trace(go.Scatter(
                x=dates,
                y=ec_values,
                mode='lines+markers',
                name='Economic Capital',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Risk Metrics Trend",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                hovermode='x unified',
                template='plotly_white'
            )
        
        except Exception as e:
            self.logger.error(f"Error creating risk trend chart: {e}")
            fig.add_annotation(text="Error loading chart data", x=0.5, y=0.5)
        
        return fig
    
    def create_risk_breakdown_chart(self, risk_data: Dict[str, Any]) -> go.Figure:
        """Create risk type breakdown pie chart."""
        fig = go.Figure()
        
        try:
            # Sample risk breakdown data
            risk_types = ['Credit Risk', 'Market Risk', 'Operational Risk']
            risk_values = [60, 25, 15]  # Percentages
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            fig.add_trace(go.Pie(
                labels=risk_types,
                values=risk_values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Risk Type Breakdown",
                template='plotly_white',
                showlegend=True
            )
        
        except Exception as e:
            self.logger.error(f"Error creating risk breakdown chart: {e}")
            fig.add_annotation(text="Error loading chart data", x=0.5, y=0.5)
        
        return fig
    
    def create_correlation_heatmap(self, risk_data: Dict[str, Any]) -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure()
        
        try:
            correlations = risk_data.get('correlations', {})
            
            if correlations:
                # Convert to DataFrame for easier handling
                corr_df = pd.DataFrame(correlations)
                
                fig.add_trace(go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_df.values,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 10},
                    hoverongaps=False
                ))
            else:
                # Sample correlation matrix
                factors = ['Interest Rate', 'Credit Spread', 'Equity Index', 'FX Rate', 'Commodity']
                n_factors = len(factors)
                corr_matrix = np.random.uniform(-0.8, 0.8, (n_factors, n_factors))
                np.fill_diagonal(corr_matrix, 1.0)
                
                fig.add_trace(go.Heatmap(
                    z=corr_matrix,
                    x=factors,
                    y=factors,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 10}
                ))
            
            fig.update_layout(
                title="Risk Factor Correlation Matrix",
                template='plotly_white',
                width=800,
                height=600
            )
        
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            fig.add_annotation(text="Error loading chart data", x=0.5, y=0.5)
        
        return fig
    
    def run_dashboard(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
        """
        Run the dashboard application.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        if not self.app:
            self.initialize_dashboard(port=port, debug=debug)
        
        self.logger.info(f"Starting dashboard on http://{host}:{port}")
        
        try:
            self.app.run_server(host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
            raise
    
    def export_dashboard_data(self, format: str = 'json') -> Union[str, bytes]:
        """
        Export dashboard data in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'excel')
            
        Returns:
            Exported data
        """
        self.logger.info(f"Exporting dashboard data in {format} format")
        
        try:
            # Fetch current data
            data = self._fetch_risk_data()
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Convert to CSV format
                output = []
                for section, section_data in data.items():
                    if isinstance(section_data, dict):
                        df = pd.DataFrame(section_data).T
                        output.append(f"# {section}")
                        output.append(df.to_csv())
                        output.append("")
                
                return "\n".join(output)
            
            elif format.lower() == 'excel':
                # This would require additional implementation
                # for creating Excel files with multiple sheets
                return json.dumps(data, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
            return f"Error exporting data: {e}"