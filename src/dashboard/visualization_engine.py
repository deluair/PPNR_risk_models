"""
Visualization Engine Module

Advanced visualization capabilities for PPNR risk models:
- Interactive chart generation
- Statistical plot creation
- Risk-specific visualizations
- Multi-dimensional data plotting
- Export and customization options
- Theme and styling management
- Animation and real-time updates
- Performance optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import json

class ChartType(Enum):
    """Chart type options."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    PIE = "pie"
    TREEMAP = "treemap"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    SURFACE = "surface"
    CONTOUR = "contour"
    RADAR = "radar"
    SANKEY = "sankey"

class PlotStyle(Enum):
    """Plot style themes."""
    CORPORATE = "plotly_white"
    DARK = "plotly_dark"
    MINIMAL = "simple_white"
    PRESENTATION = "presentation"
    SEABORN = "seaborn"
    GGPLOT2 = "ggplot2"

class ColorScheme(Enum):
    """Color scheme options."""
    RISK_GRADIENT = ["#2E8B57", "#FFD700", "#FF6347", "#DC143C"]  # Green to Red
    CORPORATE_BLUE = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    VIRIDIS = "Viridis"
    PLASMA = "Plasma"
    BLUES = "Blues"
    REDS = "Reds"

@dataclass
class ChartConfig:
    """Chart configuration settings."""
    chart_type: ChartType = ChartType.LINE
    style: PlotStyle = PlotStyle.CORPORATE
    color_scheme: ColorScheme = ColorScheme.CORPORATE_BLUE
    width: int = 800
    height: int = 600
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    show_legend: bool = True
    interactive: bool = True
    export_format: str = "html"

class VisualizationEngine:
    """
    Advanced visualization engine for risk analytics.
    
    Features:
    - Multi-format chart generation (Plotly, Matplotlib, Seaborn)
    - Risk-specific visualization templates
    - Interactive and static plotting capabilities
    - Customizable themes and styling
    - Export functionality (HTML, PNG, PDF, SVG)
    - Animation and real-time update support
    - Performance optimization for large datasets
    - Statistical visualization tools
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualization engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.viz_config = self.config.get('visualization', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.VisualizationEngine")
        
        # Default settings
        self.default_config = ChartConfig()
        self.color_palettes = self._initialize_color_palettes()
        
        # Chart cache for performance
        self.chart_cache = {}
        self.cache_enabled = self.viz_config.get('enable_cache', True)
        
        # Matplotlib settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info("Visualization engine initialized")
    
    def _initialize_color_palettes(self) -> Dict[str, List[str]]:
        """Initialize color palettes for different chart types."""
        return {
            'risk_levels': ['#2E8B57', '#FFD700', '#FF6347', '#DC143C'],  # Green, Yellow, Orange, Red
            'risk_types': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],  # Blue, Orange, Green, Red, Purple
            'performance': ['#2E8B57', '#32CD32', '#FFD700', '#FF6347'],  # Performance gradient
            'regulatory': ['#4169E1', '#1E90FF', '#87CEEB', '#B0C4DE'],  # Blue shades
            'stress_test': ['#8B0000', '#DC143C', '#FF6347', '#FFA07A'],  # Red gradient
            'correlation': ['#000080', '#4169E1', '#FFFFFF', '#FF6347', '#8B0000']  # Blue-White-Red
        }
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str],
                         config: ChartConfig = None) -> go.Figure:
        """
        Create interactive line chart.
        
        Args:
            data: DataFrame with data
            x_col: X-axis column name
            y_cols: Y-axis column names
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        fig = go.Figure()
        
        colors = self.color_palettes.get('risk_types', px.colors.qualitative.Set1)
        
        for i, y_col in enumerate(y_cols):
            if y_col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{y_col}</b><br>' +
                                 f'{x_col}: %{{x}}<br>' +
                                 f'Value: %{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_title,
            yaxis_title=config.y_title,
            template=config.style.value,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            hovermode='x unified'
        )
        
        return fig
    
    def create_risk_heatmap(self, correlation_matrix: pd.DataFrame,
                           config: ChartConfig = None) -> go.Figure:
        """
        Create risk correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        fig = go.Figure()
        
        # Create heatmap
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=config.title or "Risk Factor Correlation Matrix",
            template=config.style.value,
            width=config.width,
            height=config.height,
            xaxis=dict(side="bottom"),
            yaxis=dict(side="left")
        )
        
        return fig
    
    def create_var_distribution(self, var_data: np.ndarray, confidence_levels: List[float],
                               config: ChartConfig = None) -> go.Figure:
        """
        Create VaR distribution visualization.
        
        Args:
            var_data: VaR simulation data
            confidence_levels: Confidence levels to highlight
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=var_data,
            nbinsx=50,
            name='P&L Distribution',
            opacity=0.7,
            marker_color='lightblue',
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add VaR lines
        colors = ['red', 'orange', 'yellow']
        for i, confidence_level in enumerate(confidence_levels):
            var_value = np.percentile(var_data, (1 - confidence_level) * 100)
            
            fig.add_vline(
                x=var_value,
                line_dash="dash",
                line_color=colors[i % len(colors)],
                annotation_text=f"VaR {confidence_level*100}%: {var_value:,.0f}",
                annotation_position="top"
            )
        
        fig.update_layout(
            title=config.title or "VaR Distribution Analysis",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            template=config.style.value,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_stress_test_waterfall(self, baseline: float, stress_impacts: Dict[str, float],
                                   config: ChartConfig = None) -> go.Figure:
        """
        Create stress test waterfall chart.
        
        Args:
            baseline: Baseline value
            stress_impacts: Dictionary of stress impacts by factor
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        # Prepare data for waterfall
        categories = ['Baseline'] + list(stress_impacts.keys()) + ['Final']
        values = [baseline] + list(stress_impacts.values())
        
        # Calculate cumulative values
        cumulative = [baseline]
        for impact in stress_impacts.values():
            cumulative.append(cumulative[-1] + impact)
        
        final_value = cumulative[-1]
        
        fig = go.Figure()
        
        # Add baseline bar
        fig.add_trace(go.Bar(
            x=['Baseline'],
            y=[baseline],
            name='Baseline',
            marker_color='blue',
            text=[f'${baseline:,.0f}'],
            textposition='outside'
        ))
        
        # Add impact bars
        colors = self.color_palettes.get('stress_test', ['red'] * len(stress_impacts))
        
        for i, (factor, impact) in enumerate(stress_impacts.items()):
            color = 'red' if impact < 0 else 'green'
            
            fig.add_trace(go.Bar(
                x=[factor],
                y=[abs(impact)],
                base=[cumulative[i] if impact > 0 else cumulative[i] + impact],
                name=factor,
                marker_color=color,
                text=[f'${impact:+,.0f}'],
                textposition='outside'
            ))
        
        # Add final bar
        fig.add_trace(go.Bar(
            x=['Final'],
            y=[final_value],
            name='Final',
            marker_color='darkblue',
            text=[f'${final_value:,.0f}'],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=config.title or "Stress Test Impact Analysis",
            xaxis_title="Factors",
            yaxis_title="Value ($)",
            template=config.style.value,
            width=config.width,
            height=config.height,
            showlegend=False,
            barmode='relative'
        )
        
        return fig
    
    def create_risk_attribution_treemap(self, attribution_data: Dict[str, float],
                                      config: ChartConfig = None) -> go.Figure:
        """
        Create risk attribution treemap.
        
        Args:
            attribution_data: Risk attribution by factor/portfolio
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        # Prepare data
        labels = list(attribution_data.keys())
        values = [abs(v) for v in attribution_data.values()]
        parents = [''] * len(labels)  # All top-level for simplicity
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            values=values,
            parents=parents,
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: %{value:,.0f}<br>' +
                         'Percentage: %{percentParent}<extra></extra>',
            marker_colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title=config.title or "Risk Attribution Analysis",
            template=config.style.value,
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_performance_scatter(self, risk_data: np.ndarray, return_data: np.ndarray,
                                 labels: List[str] = None, config: ChartConfig = None) -> go.Figure:
        """
        Create risk-return scatter plot.
        
        Args:
            risk_data: Risk values (x-axis)
            return_data: Return values (y-axis)
            labels: Point labels
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        fig = go.Figure()
        
        # Create scatter plot
        fig.add_trace(go.Scatter(
            x=risk_data,
            y=return_data,
            mode='markers+text',
            text=labels or [''] * len(risk_data),
            textposition='top center',
            marker=dict(
                size=12,
                color=return_data,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return")
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Risk: %{x:.2f}<br>' +
                         'Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add efficient frontier line (if applicable)
        if len(risk_data) > 2:
            # Simple polynomial fit for demonstration
            z = np.polyfit(risk_data, return_data, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(risk_data), max(risk_data), 100)
            y_smooth = p(x_smooth)
            
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', dash='dash'),
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=config.title or "Risk-Return Analysis",
            xaxis_title=config.x_title or "Risk",
            yaxis_title=config.y_title or "Return (%)",
            template=config.style.value,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_regulatory_gauge(self, current_ratio: float, minimum_ratio: float,
                              target_ratio: float, metric_name: str,
                              config: ChartConfig = None) -> go.Figure:
        """
        Create regulatory ratio gauge chart.
        
        Args:
            current_ratio: Current ratio value
            minimum_ratio: Minimum required ratio
            target_ratio: Target ratio
            metric_name: Name of the metric
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        fig = go.Figure()
        
        # Determine color based on ratio level
        if current_ratio >= target_ratio:
            color = "green"
        elif current_ratio >= minimum_ratio:
            color = "yellow"
        else:
            color = "red"
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=current_ratio,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': metric_name},
            delta={'reference': target_ratio, 'suffix': "%"},
            gauge={
                'axis': {'range': [None, max(target_ratio * 1.5, current_ratio * 1.2)]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, minimum_ratio], 'color': "lightgray"},
                    {'range': [minimum_ratio, target_ratio], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': minimum_ratio
                }
            }
        ))
        
        fig.update_layout(
            template=config.style.value,
            width=config.width,
            height=config.height,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    def create_time_series_decomposition(self, data: pd.Series, 
                                       config: ChartConfig = None) -> go.Figure:
        """
        Create time series decomposition plot.
        
        Args:
            data: Time series data
            config: Chart configuration
            
        Returns:
            Plotly figure with subplots
        """
        config = config or self.default_config
        
        # Simple decomposition (trend, seasonal, residual)
        # In practice, would use statsmodels.tsa.seasonal_decompose
        
        # Calculate rolling mean as trend
        trend = data.rolling(window=12, center=True).mean()
        
        # Calculate seasonal component (simplified)
        seasonal = data.groupby(data.index.month).transform('mean') - data.mean()
        
        # Calculate residual
        residual = data - trend - seasonal
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )
        
        # Original data
        fig.add_trace(go.Scatter(
            x=data.index, y=data.values,
            mode='lines', name='Original',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend.values,
            mode='lines', name='Trend',
            line=dict(color='red')
        ), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(
            x=seasonal.index, y=seasonal.values,
            mode='lines', name='Seasonal',
            line=dict(color='green')
        ), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(
            x=residual.index, y=residual.values,
            mode='lines', name='Residual',
            line=dict(color='orange')
        ), row=4, col=1)
        
        fig.update_layout(
            title=config.title or "Time Series Decomposition",
            template=config.style.value,
            width=config.width,
            height=config.height * 1.5,  # Taller for subplots
            showlegend=False
        )
        
        return fig
    
    def create_monte_carlo_convergence(self, simulation_results: np.ndarray,
                                     config: ChartConfig = None) -> go.Figure:
        """
        Create Monte Carlo convergence plot.
        
        Args:
            simulation_results: Array of simulation results
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        config = config or self.default_config
        
        # Calculate running mean and confidence intervals
        n_sims = len(simulation_results)
        running_mean = np.cumsum(simulation_results) / np.arange(1, n_sims + 1)
        
        # Calculate running standard error
        running_std = np.array([
            np.std(simulation_results[:i+1]) / np.sqrt(i+1) 
            for i in range(n_sims)
        ])
        
        # Confidence intervals
        upper_ci = running_mean + 1.96 * running_std
        lower_ci = running_mean - 1.96 * running_std
        
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=list(range(1, n_sims + 1)) + list(range(n_sims, 0, -1)),
            y=list(upper_ci) + list(lower_ci[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))
        
        # Add running mean
        fig.add_trace(go.Scatter(
            x=list(range(1, n_sims + 1)),
            y=running_mean,
            mode='lines',
            name='Running Mean',
            line=dict(color='blue', width=2),
            hovertemplate='Simulation: %{x}<br>Mean: %{y:.4f}<extra></extra>'
        ))
        
        # Add final value line
        final_mean = running_mean[-1]
        fig.add_hline(
            y=final_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Final Mean: {final_mean:.4f}"
        )
        
        fig.update_layout(
            title=config.title or "Monte Carlo Convergence Analysis",
            xaxis_title="Number of Simulations",
            yaxis_title="Estimated Value",
            template=config.style.value,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_risk_dashboard_summary(self, summary_data: Dict[str, Any],
                                    config: ChartConfig = None) -> go.Figure:
        """
        Create comprehensive risk dashboard summary.
        
        Args:
            summary_data: Dictionary with various risk metrics
            config: Chart configuration
            
        Returns:
            Plotly figure with multiple subplots
        """
        config = config or self.default_config
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['VaR Trend', 'Risk Breakdown', 'Performance Metrics', 'Alerts'],
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": True}, {"type": "bar"}]]
        )
        
        # VaR Trend (top-left)
        if 'var_trend' in summary_data:
            var_data = summary_data['var_trend']
            fig.add_trace(go.Scatter(
                x=var_data.get('dates', []),
                y=var_data.get('values', []),
                mode='lines+markers',
                name='VaR 99%',
                line=dict(color='red')
            ), row=1, col=1)
        
        # Risk Breakdown (top-right)
        if 'risk_breakdown' in summary_data:
            breakdown = summary_data['risk_breakdown']
            fig.add_trace(go.Pie(
                labels=breakdown.get('labels', []),
                values=breakdown.get('values', []),
                name="Risk Types"
            ), row=1, col=2)
        
        # Performance Metrics (bottom-left)
        if 'performance' in summary_data:
            perf_data = summary_data['performance']
            fig.add_trace(go.Scatter(
                x=perf_data.get('dates', []),
                y=perf_data.get('returns', []),
                mode='lines',
                name='Returns',
                line=dict(color='blue')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=perf_data.get('dates', []),
                y=perf_data.get('risk', []),
                mode='lines',
                name='Risk',
                line=dict(color='orange'),
                yaxis='y2'
            ), row=2, col=1, secondary_y=True)
        
        # Alerts (bottom-right)
        if 'alerts' in summary_data:
            alerts = summary_data['alerts']
            fig.add_trace(go.Bar(
                x=alerts.get('categories', []),
                y=alerts.get('counts', []),
                name='Alert Counts',
                marker_color='red'
            ), row=2, col=2)
        
        fig.update_layout(
            title=config.title or "Risk Dashboard Summary",
            template=config.style.value,
            width=config.width * 1.5,
            height=config.height * 1.2,
            showlegend=True
        )
        
        return fig
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """
        Export chart to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            Path to exported file
        """
        self.logger.info(f"Exporting chart to {filename}.{format}")
        
        try:
            output_path = f"{filename}.{format}"
            
            if format.lower() == 'html':
                fig.write_html(output_path)
            elif format.lower() == 'png':
                fig.write_image(output_path, format='png')
            elif format.lower() == 'pdf':
                fig.write_image(output_path, format='pdf')
            elif format.lower() == 'svg':
                fig.write_image(output_path, format='svg')
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Chart exported successfully to {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error exporting chart: {e}")
            raise
    
    def create_animated_chart(self, data: pd.DataFrame, x_col: str, y_col: str,
                            animation_frame: str, config: ChartConfig = None) -> go.Figure:
        """
        Create animated chart.
        
        Args:
            data: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            animation_frame: Column to animate by
            config: Chart configuration
            
        Returns:
            Animated Plotly figure
        """
        config = config or self.default_config
        
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            animation_frame=animation_frame,
            title=config.title or "Animated Risk Analysis",
            template=config.style.value,
            width=config.width,
            height=config.height
        )
        
        # Update animation settings
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
        
        return fig
    
    def generate_chart_report(self, charts: List[go.Figure], 
                            report_title: str = "Risk Analysis Report") -> str:
        """
        Generate HTML report with multiple charts.
        
        Args:
            charts: List of Plotly figures
            report_title: Report title
            
        Returns:
            HTML report string
        """
        self.logger.info(f"Generating chart report with {len(charts)} charts")
        
        html_parts = [
            f"<html><head><title>{report_title}</title>",
            "<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>",
            "</head><body>",
            f"<h1>{report_title}</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        for i, chart in enumerate(charts):
            chart_html = chart.to_html(include_plotlyjs='inline', div_id=f"chart_{i}")
            # Extract just the div part
            div_start = chart_html.find('<div')
            div_end = chart_html.find('</div>') + 6
            chart_div = chart_html[div_start:div_end]
            
            html_parts.append(f"<h2>Chart {i+1}</h2>")
            html_parts.append(chart_div)
            html_parts.append("<hr>")
        
        html_parts.append("</body></html>")
        
        return "\n".join(html_parts)