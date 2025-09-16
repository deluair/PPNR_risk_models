"""
Dashboard and Visualization Package

Interactive dashboard and visualization tools for PPNR risk models:
- Real-time risk monitoring dashboards
- Interactive stress testing visualizations
- Regulatory reporting dashboards
- Portfolio risk analytics
- Performance measurement dashboards
- Executive summary reports
- Model validation visualizations
- Scenario analysis tools
"""

from .risk_dashboard import RiskDashboard, DashboardConfig
from .visualization_engine import VisualizationEngine, ChartType, PlotStyle
from .report_generator import ReportGenerator, ReportType, ReportFormat
from .interactive_charts import InteractiveCharts, ChartInteraction
from .executive_dashboard import ExecutiveDashboard, ExecutiveMetrics
from .stress_test_visualizer import StressTestVisualizer, ScenarioVisualization
from .performance_dashboard import PerformanceDashboard, PerformanceMetrics

__all__ = [
    'RiskDashboard',
    'DashboardConfig',
    'VisualizationEngine',
    'ChartType',
    'PlotStyle',
    'ReportGenerator',
    'ReportType',
    'ReportFormat',
    'InteractiveCharts',
    'ChartInteraction',
    'ExecutiveDashboard',
    'ExecutiveMetrics',
    'StressTestVisualizer',
    'ScenarioVisualization',
    'PerformanceDashboard',
    'PerformanceMetrics'
]

__version__ = "1.0.0"
__author__ = "PPNR Risk Models Team"