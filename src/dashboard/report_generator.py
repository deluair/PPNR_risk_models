"""
Report Generator Module

Automated report generation for PPNR risk models:
- Regulatory compliance reports
- Executive summaries
- Technical documentation
- Stress test results
- Model validation reports
- Performance analytics
- Risk assessment summaries
- Multi-format output support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
import zipfile
from jinja2 import Template, Environment, FileSystemLoader
import markdown
from weasyprint import HTML, CSS
import plotly.graph_objects as go
from plotly.offline import plot
import base64
from io import BytesIO

class ReportType(Enum):
    """Report type options."""
    EXECUTIVE_SUMMARY = "executive_summary"
    REGULATORY_FILING = "regulatory_filing"
    STRESS_TEST = "stress_test"
    MODEL_VALIDATION = "model_validation"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_REVIEW = "performance_review"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    AUDIT_REPORT = "audit_report"

class OutputFormat(Enum):
    """Output format options."""
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    JSON = "json"
    MARKDOWN = "markdown"

class ReportFrequency(Enum):
    """Report frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    AD_HOC = "ad_hoc"

@dataclass
class ReportSection:
    """Report section configuration."""
    title: str
    content: str = ""
    charts: List[go.Figure] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    subsections: List['ReportSection'] = field(default_factory=list)
    include_in_toc: bool = True
    page_break_before: bool = False
    
@dataclass
class ReportConfig:
    """Report configuration settings."""
    report_type: ReportType = ReportType.EXECUTIVE_SUMMARY
    output_format: OutputFormat = OutputFormat.HTML
    title: str = "Risk Analysis Report"
    subtitle: str = ""
    author: str = "PPNR Risk Management System"
    company: str = ""
    date: datetime = field(default_factory=datetime.now)
    template_name: str = "default"
    include_toc: bool = True
    include_executive_summary: bool = True
    include_appendix: bool = True
    logo_path: str = ""
    footer_text: str = ""
    confidentiality_level: str = "Internal Use Only"

class ReportGenerator:
    """
    Comprehensive report generator for risk analytics.
    
    Features:
    - Multiple report types and formats
    - Template-based generation
    - Automated chart and table inclusion
    - Regulatory compliance formatting
    - Executive summary generation
    - Multi-format export (HTML, PDF, DOCX, Excel)
    - Batch report generation
    - Custom styling and branding
    - Data validation and quality checks
    - Automated distribution
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.report_config = self.config.get('reporting', {})
        
        # Set up logging
        self.logger = logging.getLogger("PPNR.ReportGenerator")
        
        # Initialize template environment
        self.template_dir = self.config.get('template_directory', 'templates')
        self.output_dir = self.config.get('output_directory', 'reports')
        
        # Create directories if they don't exist
        Path(self.template_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )
        
        # Report templates
        self.templates = self._initialize_templates()
        
        # Default styling
        self.default_css = self._get_default_css()
        
        self.logger.info("Report generator initialized")
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize report templates."""
        return {
            'executive_summary': """
            <h1>{{ title }}</h1>
            <h2>Executive Summary</h2>
            <p><strong>Report Date:</strong> {{ date.strftime('%B %d, %Y') }}</p>
            <p><strong>Reporting Period:</strong> {{ reporting_period }}</p>
            
            <h3>Key Findings</h3>
            <ul>
            {% for finding in key_findings %}
                <li>{{ finding }}</li>
            {% endfor %}
            </ul>
            
            <h3>Risk Metrics Summary</h3>
            <table class="summary-table">
                <tr><th>Metric</th><th>Current</th><th>Previous</th><th>Change</th></tr>
                {% for metric in risk_metrics %}
                <tr>
                    <td>{{ metric.name }}</td>
                    <td>{{ metric.current }}</td>
                    <td>{{ metric.previous }}</td>
                    <td class="{{ 'positive' if metric.change > 0 else 'negative' }}">
                        {{ metric.change }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            {% for section in sections %}
                <h2>{{ section.title }}</h2>
                {{ section.content }}
                {% for chart in section.charts %}
                    {{ chart }}
                {% endfor %}
            {% endfor %}
            """,
            
            'regulatory_filing': """
            <div class="header">
                <h1>{{ title }}</h1>
                <p><strong>Institution:</strong> {{ company }}</p>
                <p><strong>Filing Date:</strong> {{ date.strftime('%Y-%m-%d') }}</p>
                <p><strong>Reporting Period:</strong> {{ reporting_period }}</p>
            </div>
            
            <h2>Regulatory Capital Summary</h2>
            <table class="regulatory-table">
                {% for item in capital_summary %}
                <tr>
                    <td>{{ item.component }}</td>
                    <td class="amount">${{ "{:,.0f}".format(item.amount) }}</td>
                    <td class="ratio">{{ "{:.2f}%".format(item.ratio) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Risk-Weighted Assets</h2>
            <table class="regulatory-table">
                {% for category in rwa_breakdown %}
                <tr>
                    <td>{{ category.name }}</td>
                    <td class="amount">${{ "{:,.0f}".format(category.amount) }}</td>
                    <td class="percentage">{{ "{:.1f}%".format(category.percentage) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            {% for section in sections %}
                <h2>{{ section.title }}</h2>
                {{ section.content }}
            {% endfor %}
            """,
            
            'stress_test': """
            <h1>{{ title }}</h1>
            <h2>Stress Test Results</h2>
            
            <div class="scenario-summary">
                <h3>Scenario Overview</h3>
                <p><strong>Test Date:</strong> {{ date.strftime('%Y-%m-%d') }}</p>
                <p><strong>Scenarios Tested:</strong> {{ scenarios|length }}</p>
                
                {% for scenario in scenarios %}
                <div class="scenario">
                    <h4>{{ scenario.name }}</h4>
                    <p>{{ scenario.description }}</p>
                    <table class="results-table">
                        <tr><th>Metric</th><th>Baseline</th><th>Stressed</th><th>Impact</th></tr>
                        {% for result in scenario.results %}
                        <tr>
                            <td>{{ result.metric }}</td>
                            <td>{{ result.baseline }}</td>
                            <td>{{ result.stressed }}</td>
                            <td class="{{ 'negative' if result.impact < 0 else 'positive' }}">
                                {{ result.impact }}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endfor %}
            </div>
            
            {% for section in sections %}
                <h2>{{ section.title }}</h2>
                {{ section.content }}
            {% endfor %}
            """
        }
    
    def _get_default_css(self) -> str:
        """Get default CSS styling."""
        return """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        
        .header {
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        
        h3 {
            color: #34495e;
            font-size: 1.4em;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        
        .summary-table, .regulatory-table, .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        
        .summary-table th, .regulatory-table th, .results-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        .summary-table td, .regulatory-table td, .results-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        
        .summary-table tr:nth-child(even), 
        .regulatory-table tr:nth-child(even),
        .results-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .amount {
            text-align: right;
            font-weight: bold;
        }
        
        .ratio, .percentage {
            text-align: center;
            font-weight: bold;
        }
        
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        
        .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .scenario {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            font-size: 12px;
            color: #7f8c8d;
        }
        
        .confidential {
            background-color: #e74c3c;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 20px;
        }
        
        @media print {
            body { margin: 0; }
            .page-break { page-break-before: always; }
        }
        </style>
        """
    
    def generate_executive_summary(self, data: Dict[str, Any], 
                                 config: ReportConfig = None) -> str:
        """
        Generate executive summary report.
        
        Args:
            data: Report data dictionary
            config: Report configuration
            
        Returns:
            Generated HTML report
        """
        config = config or ReportConfig()
        
        self.logger.info("Generating executive summary report")
        
        # Prepare template data
        template_data = {
            'title': config.title,
            'date': config.date,
            'company': config.company,
            'reporting_period': data.get('reporting_period', 'Current Period'),
            'key_findings': data.get('key_findings', []),
            'risk_metrics': data.get('risk_metrics', []),
            'sections': data.get('sections', [])
        }
        
        # Render template
        template = self.jinja_env.from_string(self.templates['executive_summary'])
        content = template.render(**template_data)
        
        # Add CSS styling
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <meta charset="utf-8">
            {self.default_css}
        </head>
        <body>
            {self._add_confidentiality_banner(config)}
            {content}
            {self._add_footer(config)}
        </body>
        </html>
        """
        
        return html_report
    
    def generate_regulatory_report(self, data: Dict[str, Any],
                                 config: ReportConfig = None) -> str:
        """
        Generate regulatory compliance report.
        
        Args:
            data: Report data dictionary
            config: Report configuration
            
        Returns:
            Generated HTML report
        """
        config = config or ReportConfig(report_type=ReportType.REGULATORY_FILING)
        
        self.logger.info("Generating regulatory compliance report")
        
        # Prepare template data
        template_data = {
            'title': config.title,
            'date': config.date,
            'company': config.company,
            'reporting_period': data.get('reporting_period', 'Current Quarter'),
            'capital_summary': data.get('capital_summary', []),
            'rwa_breakdown': data.get('rwa_breakdown', []),
            'sections': data.get('sections', [])
        }
        
        # Render template
        template = self.jinja_env.from_string(self.templates['regulatory_filing'])
        content = template.render(**template_data)
        
        # Add CSS styling
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <meta charset="utf-8">
            {self.default_css}
        </head>
        <body>
            {self._add_confidentiality_banner(config)}
            {content}
            {self._add_footer(config)}
        </body>
        </html>
        """
        
        return html_report
    
    def generate_stress_test_report(self, data: Dict[str, Any],
                                  config: ReportConfig = None) -> str:
        """
        Generate stress test report.
        
        Args:
            data: Report data dictionary
            config: Report configuration
            
        Returns:
            Generated HTML report
        """
        config = config or ReportConfig(report_type=ReportType.STRESS_TEST)
        
        self.logger.info("Generating stress test report")
        
        # Prepare template data
        template_data = {
            'title': config.title,
            'date': config.date,
            'scenarios': data.get('scenarios', []),
            'sections': data.get('sections', [])
        }
        
        # Render template
        template = self.jinja_env.from_string(self.templates['stress_test'])
        content = template.render(**template_data)
        
        # Add CSS styling
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <meta charset="utf-8">
            {self.default_css}
        </head>
        <body>
            {self._add_confidentiality_banner(config)}
            {content}
            {self._add_footer(config)}
        </body>
        </html>
        """
        
        return html_report
    
    def generate_model_validation_report(self, model_results: Dict[str, Any],
                                       config: ReportConfig = None) -> str:
        """
        Generate model validation report.
        
        Args:
            model_results: Model validation results
            config: Report configuration
            
        Returns:
            Generated HTML report
        """
        config = config or ReportConfig(report_type=ReportType.MODEL_VALIDATION)
        
        self.logger.info("Generating model validation report")
        
        sections = []
        
        # Model Performance Section
        if 'performance_metrics' in model_results:
            perf_content = self._format_performance_metrics(
                model_results['performance_metrics']
            )
            sections.append(ReportSection(
                title="Model Performance Metrics",
                content=perf_content
            ))
        
        # Backtesting Results
        if 'backtesting' in model_results:
            backtest_content = self._format_backtesting_results(
                model_results['backtesting']
            )
            sections.append(ReportSection(
                title="Backtesting Results",
                content=backtest_content
            ))
        
        # Model Limitations
        if 'limitations' in model_results:
            limitations_content = self._format_model_limitations(
                model_results['limitations']
            )
            sections.append(ReportSection(
                title="Model Limitations and Assumptions",
                content=limitations_content
            ))
        
        # Generate report
        report_data = {
            'sections': sections,
            'reporting_period': model_results.get('validation_period', 'Current Period')
        }
        
        return self._generate_custom_report(report_data, config)
    
    def generate_performance_report(self, performance_data: Dict[str, Any],
                                  config: ReportConfig = None) -> str:
        """
        Generate performance analytics report.
        
        Args:
            performance_data: Performance data dictionary
            config: Report configuration
            
        Returns:
            Generated HTML report
        """
        config = config or ReportConfig(report_type=ReportType.PERFORMANCE_REVIEW)
        
        self.logger.info("Generating performance analytics report")
        
        sections = []
        
        # Portfolio Performance
        if 'portfolio_performance' in performance_data:
            portfolio_content = self._format_portfolio_performance(
                performance_data['portfolio_performance']
            )
            sections.append(ReportSection(
                title="Portfolio Performance Analysis",
                content=portfolio_content
            ))
        
        # Risk-Adjusted Returns
        if 'risk_adjusted_returns' in performance_data:
            risk_adj_content = self._format_risk_adjusted_returns(
                performance_data['risk_adjusted_returns']
            )
            sections.append(ReportSection(
                title="Risk-Adjusted Performance",
                content=risk_adj_content
            ))
        
        # Attribution Analysis
        if 'attribution' in performance_data:
            attribution_content = self._format_attribution_analysis(
                performance_data['attribution']
            )
            sections.append(ReportSection(
                title="Performance Attribution",
                content=attribution_content
            ))
        
        # Generate report
        report_data = {
            'sections': sections,
            'reporting_period': performance_data.get('period', 'Current Period')
        }
        
        return self._generate_custom_report(report_data, config)
    
    def _generate_custom_report(self, data: Dict[str, Any],
                              config: ReportConfig) -> str:
        """Generate custom report with sections."""
        
        # Build content
        content_parts = []
        
        if config.include_executive_summary and 'executive_summary' in data:
            content_parts.append(f"<h2>Executive Summary</h2>")
            content_parts.append(data['executive_summary'])
        
        # Add sections
        for section in data.get('sections', []):
            if section.page_break_before:
                content_parts.append('<div class="page-break"></div>')
            
            content_parts.append(f"<h2>{section.title}</h2>")
            content_parts.append(section.content)
            
            # Add charts
            for chart in section.charts:
                chart_html = self._embed_chart(chart)
                content_parts.append(f'<div class="chart-container">{chart_html}</div>')
            
            # Add tables
            for table in section.tables:
                table_html = table.to_html(classes='summary-table', escape=False)
                content_parts.append(table_html)
        
        content = "\n".join(content_parts)
        
        # Build full HTML
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <meta charset="utf-8">
            {self.default_css}
        </head>
        <body>
            {self._add_confidentiality_banner(config)}
            <h1>{config.title}</h1>
            {content}
            {self._add_footer(config)}
        </body>
        </html>
        """
        
        return html_report
    
    def _format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics for display."""
        content = "<table class='summary-table'>"
        content += "<tr><th>Metric</th><th>Value</th><th>Benchmark</th><th>Status</th></tr>"
        
        for metric_name, metric_data in metrics.items():
            value = metric_data.get('value', 'N/A')
            benchmark = metric_data.get('benchmark', 'N/A')
            status = metric_data.get('status', 'Unknown')
            
            status_class = 'positive' if status == 'Pass' else 'negative'
            
            content += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{value}</td>
                <td>{benchmark}</td>
                <td class='{status_class}'>{status}</td>
            </tr>
            """
        
        content += "</table>"
        return content
    
    def _format_backtesting_results(self, results: Dict[str, Any]) -> str:
        """Format backtesting results for display."""
        content = f"""
        <p><strong>Backtesting Period:</strong> {results.get('period', 'N/A')}</p>
        <p><strong>Number of Observations:</strong> {results.get('observations', 'N/A')}</p>
        <p><strong>Number of Exceptions:</strong> {results.get('exceptions', 'N/A')}</p>
        <p><strong>Exception Rate:</strong> {results.get('exception_rate', 'N/A')}%</p>
        
        <h4>Test Results</h4>
        <ul>
        """
        
        for test_name, test_result in results.get('tests', {}).items():
            status = 'Pass' if test_result.get('pass', False) else 'Fail'
            status_class = 'positive' if status == 'Pass' else 'negative'
            
            content += f"""
            <li><strong>{test_name}:</strong> 
                <span class='{status_class}'>{status}</span>
                ({test_result.get('description', '')})
            </li>
            """
        
        content += "</ul>"
        return content
    
    def _format_model_limitations(self, limitations: List[str]) -> str:
        """Format model limitations for display."""
        content = "<ul>"
        for limitation in limitations:
            content += f"<li>{limitation}</li>"
        content += "</ul>"
        return content
    
    def _format_portfolio_performance(self, performance: Dict[str, Any]) -> str:
        """Format portfolio performance data."""
        content = f"""
        <table class='summary-table'>
            <tr><th>Period</th><th>Return</th><th>Volatility</th><th>Sharpe Ratio</th></tr>
        """
        
        for period, data in performance.items():
            content += f"""
            <tr>
                <td>{period}</td>
                <td>{data.get('return', 'N/A')}%</td>
                <td>{data.get('volatility', 'N/A')}%</td>
                <td>{data.get('sharpe_ratio', 'N/A')}</td>
            </tr>
            """
        
        content += "</table>"
        return content
    
    def _format_risk_adjusted_returns(self, returns_data: Dict[str, Any]) -> str:
        """Format risk-adjusted returns data."""
        content = "<table class='summary-table'>"
        content += "<tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>"
        
        metrics = {
            'Sharpe Ratio': returns_data.get('sharpe_ratio'),
            'Sortino Ratio': returns_data.get('sortino_ratio'),
            'Calmar Ratio': returns_data.get('calmar_ratio'),
            'Information Ratio': returns_data.get('information_ratio')
        }
        
        for metric_name, value in metrics.items():
            if value is not None:
                interpretation = self._interpret_ratio(metric_name, value)
                content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value:.3f}</td>
                    <td>{interpretation}</td>
                </tr>
                """
        
        content += "</table>"
        return content
    
    def _format_attribution_analysis(self, attribution: Dict[str, Any]) -> str:
        """Format attribution analysis data."""
        content = "<table class='summary-table'>"
        content += "<tr><th>Factor</th><th>Contribution</th><th>Percentage</th></tr>"
        
        total_return = sum(attribution.values())
        
        for factor, contribution in attribution.items():
            percentage = (contribution / total_return * 100) if total_return != 0 else 0
            contribution_class = 'positive' if contribution >= 0 else 'negative'
            
            content += f"""
            <tr>
                <td>{factor}</td>
                <td class='{contribution_class}'>{contribution:+.2f}%</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        
        content += "</table>"
        return content
    
    def _interpret_ratio(self, ratio_name: str, value: float) -> str:
        """Interpret risk-adjusted ratio values."""
        interpretations = {
            'Sharpe Ratio': {
                (float('-inf'), 0): 'Poor',
                (0, 0.5): 'Below Average',
                (0.5, 1.0): 'Average',
                (1.0, 2.0): 'Good',
                (2.0, float('inf')): 'Excellent'
            }
        }
        
        if ratio_name in interpretations:
            for (low, high), interpretation in interpretations[ratio_name].items():
                if low <= value < high:
                    return interpretation
        
        return 'N/A'
    
    def _embed_chart(self, chart: go.Figure) -> str:
        """Embed Plotly chart as HTML."""
        return chart.to_html(include_plotlyjs='inline', div_id=None)
    
    def _add_confidentiality_banner(self, config: ReportConfig) -> str:
        """Add confidentiality banner to report."""
        if config.confidentiality_level:
            return f'<div class="confidential">{config.confidentiality_level}</div>'
        return ""
    
    def _add_footer(self, config: ReportConfig) -> str:
        """Add footer to report."""
        footer_text = config.footer_text or f"Generated by {config.author} on {config.date.strftime('%Y-%m-%d %H:%M:%S')}"
        return f'<div class="footer">{footer_text}</div>'
    
    def export_to_pdf(self, html_content: str, filename: str) -> str:
        """
        Export HTML report to PDF.
        
        Args:
            html_content: HTML content
            filename: Output filename
            
        Returns:
            Path to PDF file
        """
        self.logger.info(f"Exporting report to PDF: {filename}")
        
        try:
            output_path = os.path.join(self.output_dir, f"{filename}.pdf")
            
            # Convert HTML to PDF
            HTML(string=html_content).write_pdf(output_path)
            
            self.logger.info(f"PDF report exported to {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error exporting to PDF: {e}")
            raise
    
    def export_to_excel(self, data: Dict[str, pd.DataFrame], filename: str) -> str:
        """
        Export data to Excel workbook.
        
        Args:
            data: Dictionary of DataFrames (sheet_name: DataFrame)
            filename: Output filename
            
        Returns:
            Path to Excel file
        """
        self.logger.info(f"Exporting data to Excel: {filename}")
        
        try:
            output_path = os.path.join(self.output_dir, f"{filename}.xlsx")
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Excel report exported to {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            raise
    
    def create_report_package(self, reports: Dict[str, str], 
                            package_name: str) -> str:
        """
        Create ZIP package with multiple reports.
        
        Args:
            reports: Dictionary of report_name: content
            package_name: Package filename
            
        Returns:
            Path to ZIP package
        """
        self.logger.info(f"Creating report package: {package_name}")
        
        try:
            output_path = os.path.join(self.output_dir, f"{package_name}.zip")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for report_name, content in reports.items():
                    zipf.writestr(f"{report_name}.html", content)
            
            self.logger.info(f"Report package created: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error creating report package: {e}")
            raise
    
    def schedule_report_generation(self, report_config: ReportConfig,
                                 frequency: ReportFrequency) -> Dict[str, Any]:
        """
        Schedule automated report generation.
        
        Args:
            report_config: Report configuration
            frequency: Generation frequency
            
        Returns:
            Scheduling information
        """
        self.logger.info(f"Scheduling {frequency.value} report generation")
        
        # This would integrate with a task scheduler in production
        schedule_info = {
            'report_type': report_config.report_type.value,
            'frequency': frequency.value,
            'next_run': self._calculate_next_run(frequency),
            'config': report_config
        }
        
        return schedule_info
    
    def _calculate_next_run(self, frequency: ReportFrequency) -> datetime:
        """Calculate next scheduled run time."""
        now = datetime.now()
        
        if frequency == ReportFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif frequency == ReportFrequency.QUARTERLY:
            return now + timedelta(days=90)
        elif frequency == ReportFrequency.ANNUALLY:
            return now + timedelta(days=365)
        else:
            return now