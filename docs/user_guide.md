# User Guide - PPNR Risk Models System

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Management](#data-management)
3. [Model Configuration](#model-configuration)
4. [Risk Model Usage](#risk-model-usage)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Dashboard Operations](#dashboard-operations)
7. [Report Generation](#report-generation)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements Check

Before starting, ensure your system meets the minimum requirements:

```bash
# Check Python version
python --version  # Should be 3.8+

# Check available memory
# Windows PowerShell
Get-WmiObject -Class Win32_ComputerSystem | Select-Object TotalPhysicalMemory

# Check disk space
Get-WmiObject -Class Win32_LogicalDisk | Select-Object Size,FreeSpace
```

### Initial Setup

1. **Environment Setup**
   ```bash
   # Navigate to project directory
   cd C:\Users\mhossen\OneDrive - University of Tennessee\AI\PPNR_risk_models
   
   # Activate virtual environment
   venv\Scripts\activate
   
   # Verify installation
   python -c "import src; print('Installation successful')"
   ```

2. **Configuration Verification**
   ```bash
   # Check configuration file
   type config\model_config.yaml
   ```

## Data Management

### Data Loading and Validation

#### 1. Portfolio Data Loading

```python
from src.data import DataLoader, DataValidator
import pandas as pd

# Initialize data loader
loader = DataLoader(config_path='config/model_config.yaml')

# Load different data types
portfolio_data = loader.load_portfolio_data('data/raw/portfolio_data.csv')
market_data = loader.load_market_data('data/raw/market_data.csv')
macro_data = loader.load_macro_data('data/raw/macro_data.csv')

print(f"Portfolio data shape: {portfolio_data.shape}")
print(f"Market data shape: {market_data.shape}")
```

#### 2. Data Validation

```python
# Initialize validator
validator = DataValidator()

# Validate portfolio data
validation_results = validator.validate_data(portfolio_data)

# Check validation results
if validation_results['is_valid']:
    print("Data validation passed")
    validated_data = validation_results['data']
else:
    print("Validation errors found:")
    for error in validation_results['errors']:
        print(f"- {error}")
```

#### 3. Data Quality Checks

```python
# Run comprehensive data quality assessment
quality_report = validator.generate_quality_report(portfolio_data)

# View quality metrics
print("Data Quality Summary:")
print(f"Completeness: {quality_report['completeness']:.2%}")
print(f"Accuracy: {quality_report['accuracy']:.2%}")
print(f"Consistency: {quality_report['consistency']:.2%}")
```

### Data Preprocessing

#### 1. Feature Engineering

```python
from src.data import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer()

# Create risk features
enhanced_data = engineer.create_risk_features(validated_data)

# Create time-based features
time_features = engineer.create_time_features(enhanced_data, date_column='observation_date')

print(f"Original features: {validated_data.shape[1]}")
print(f"Enhanced features: {enhanced_data.shape[1]}")
```

#### 2. Data Transformation

```python
# Apply transformations
transformed_data = engineer.apply_transformations(
    enhanced_data,
    transformations=['log_transform', 'standardize', 'winsorize']
)

# Handle missing values
clean_data = engineer.handle_missing_values(
    transformed_data,
    method='advanced_imputation'
)
```

## Model Configuration

### Configuration File Structure

The main configuration file `config/model_config.yaml` contains:

```yaml
# Model Parameters
models:
  credit_risk:
    pd_model:
      algorithm: 'lightgbm'
      hyperparameters:
        n_estimators: 100
        learning_rate: 0.1
        max_depth: 6
    lgd_model:
      algorithm: 'beta_regression'
      
  market_risk:
    var_model:
      method: 'historical_simulation'
      confidence_level: 0.99
      holding_period: 1
      
  operational_risk:
    lda_model:
      frequency_distribution: 'poisson'
      severity_distribution: 'lognormal'

# Data Settings
data:
  validation_rules:
    completeness_threshold: 0.95
    outlier_threshold: 3.0
  
# Regulatory Settings
regulatory:
  frameworks: ['ccar', 'dfast', 'basel3']
  stress_scenarios: ['baseline', 'adverse', 'severely_adverse']
```

### Customizing Configuration

```python
import yaml

# Load configuration
with open('config/model_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Modify parameters
config['models']['credit_risk']['pd_model']['hyperparameters']['n_estimators'] = 200

# Save updated configuration
with open('config/custom_config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
```

## Risk Model Usage

### Credit Risk Modeling

#### 1. PD Model Training and Prediction

```python
from src.risk_factors import CreditRiskModel

# Initialize credit risk model
credit_model = CreditRiskModel(config_path='config/model_config.yaml')

# Train PD model
pd_results = credit_model.train_pd_model(
    data=clean_data,
    target_column='default_flag',
    feature_columns=['debt_to_equity', 'current_ratio', 'roa', 'gdp_growth']
)

print(f"PD Model AUC: {pd_results['auc']:.3f}")
print(f"PD Model Accuracy: {pd_results['accuracy']:.3f}")

# Generate predictions
pd_predictions = credit_model.predict_pd(new_data)
```

#### 2. LGD Model Training

```python
# Train LGD model
lgd_results = credit_model.train_lgd_model(
    data=clean_data,
    target_column='loss_given_default',
    feature_columns=['collateral_type', 'seniority', 'industry_sector']
)

# Generate LGD predictions
lgd_predictions = credit_model.predict_lgd(new_data)
```

#### 3. Portfolio Risk Assessment

```python
# Calculate portfolio-level credit risk
portfolio_risk = credit_model.calculate_portfolio_risk(
    exposures=portfolio_data,
    pd_predictions=pd_predictions,
    lgd_predictions=lgd_predictions
)

print(f"Expected Loss: ${portfolio_risk['expected_loss']:,.2f}")
print(f"Unexpected Loss (99.9%): ${portfolio_risk['unexpected_loss']:,.2f}")
```

### Market Risk Modeling

#### 1. VaR Calculation

```python
from src.risk_factors import MarketRiskModel

# Initialize market risk model
market_model = MarketRiskModel(config_path='config/model_config.yaml')

# Calculate VaR using different methods
var_results = market_model.calculate_var(
    portfolio_data=market_positions,
    method='historical_simulation',
    confidence_level=0.99,
    holding_period=1
)

print(f"1-Day VaR (99%): ${var_results['var']:,.2f}")
print(f"Expected Shortfall: ${var_results['expected_shortfall']:,.2f}")
```

#### 2. Stress Testing

```python
# Define stress scenarios
stress_scenarios = {
    'equity_crash': {'equity_shock': -0.30, 'credit_spread_shock': 0.02},
    'interest_rate_shock': {'rate_shock': 0.02, 'curve_steepening': 0.01},
    'credit_crisis': {'credit_spread_shock': 0.05, 'equity_shock': -0.20}
}

# Run stress tests
stress_results = market_model.run_stress_tests(
    portfolio_data=market_positions,
    scenarios=stress_scenarios
)

for scenario, result in stress_results.items():
    print(f"{scenario}: ${result['stressed_pnl']:,.2f}")
```

### Operational Risk Modeling

#### 1. Loss Distribution Analysis

```python
from src.risk_factors import OperationalRiskModel

# Initialize operational risk model
op_risk_model = OperationalRiskModel(config_path='config/model_config.yaml')

# Load loss data
loss_data = loader.load_loss_data('data/raw/operational_losses.csv')

# Fit loss distribution
distribution_results = op_risk_model.fit_loss_distribution(
    loss_data=loss_data,
    business_line='trading',
    event_type='external_fraud'
)

print(f"Frequency (λ): {distribution_results['frequency_param']:.4f}")
print(f"Severity (μ, σ): {distribution_results['severity_params']}")
```

#### 2. Operational VaR Calculation

```python
# Calculate operational VaR
op_var_results = op_risk_model.calculate_operational_var(
    business_lines=['trading', 'retail_banking', 'corporate_finance'],
    confidence_level=0.999,
    time_horizon=365
)

print(f"Operational VaR (99.9%): ${op_var_results['var']:,.2f}")
print(f"Regulatory Capital: ${op_var_results['regulatory_capital']:,.2f}")
```

## Regulatory Compliance

### CCAR Stress Testing

#### 1. Scenario Setup

```python
from src.regulatory import CCARCompliance

# Initialize CCAR compliance module
ccar = CCARCompliance(config_path='config/model_config.yaml')

# Load supervisory scenarios
scenarios = ccar.load_supervisory_scenarios('data/scenarios/ccar_2024.csv')

print("Available scenarios:")
for scenario in scenarios:
    print(f"- {scenario['name']}: {scenario['description']}")
```

#### 2. Stress Test Execution

```python
# Run comprehensive stress test
stress_results = ccar.run_stress_test(
    portfolio_data=validated_data,
    scenarios=['baseline', 'adverse', 'severely_adverse'],
    projection_horizon=9  # quarters
)

# View results summary
for scenario, results in stress_results.items():
    print(f"\n{scenario.upper()} Scenario:")
    print(f"  Net Income (Q9): ${results['net_income_q9']:,.0f}M")
    print(f"  CET1 Ratio (Q9): {results['cet1_ratio_q9']:.2%}")
    print(f"  Minimum CET1: {results['min_cet1_ratio']:.2%}")
```

### Basel III Capital Calculations

#### 1. Risk-Weighted Assets

```python
from src.regulatory import BaselCompliance

# Initialize Basel compliance module
basel = BaselCompliance(config_path='config/model_config.yaml')

# Calculate risk-weighted assets
rwa_results = basel.calculate_rwa(
    portfolio_data=validated_data,
    approach='internal_ratings_based'
)

print("Risk-Weighted Assets by Type:")
print(f"  Credit Risk: ${rwa_results['credit_rwa']:,.0f}M")
print(f"  Market Risk: ${rwa_results['market_rwa']:,.0f}M")
print(f"  Operational Risk: ${rwa_results['operational_rwa']:,.0f}M")
print(f"  Total RWA: ${rwa_results['total_rwa']:,.0f}M")
```

#### 2. Capital Ratios

```python
# Calculate capital ratios
capital_results = basel.calculate_capital_ratios(
    capital_data=capital_data,
    rwa_results=rwa_results
)

print("Capital Ratios:")
print(f"  CET1 Ratio: {capital_results['cet1_ratio']:.2%}")
print(f"  Tier 1 Ratio: {capital_results['tier1_ratio']:.2%}")
print(f"  Total Capital Ratio: {capital_results['total_capital_ratio']:.2%}")
print(f"  Leverage Ratio: {capital_results['leverage_ratio']:.2%}")
```

## Dashboard Operations

### Launching the Dashboard

#### 1. Basic Dashboard Setup

```python
from src.dashboard import RiskDashboard

# Initialize dashboard
dashboard = RiskDashboard(config_path='config/model_config.yaml')

# Register risk models
dashboard.register_models([credit_model, market_model, op_risk_model])

# Configure dashboard settings
dashboard.configure_dashboard(
    theme='corporate',
    refresh_interval=300,  # 5 minutes
    enable_alerts=True
)
```

#### 2. Launch Dashboard

```python
# Run dashboard (development mode)
dashboard.run(
    host='localhost',
    port=8050,
    debug=True
)

# For production deployment
dashboard.run(
    host='0.0.0.0',
    port=8050,
    debug=False
)
```

### Dashboard Features

#### 1. Real-time Monitoring

- **Risk Metrics**: Live updates of VaR, expected loss, capital ratios
- **Alert System**: Automated alerts for threshold breaches
- **Performance Tracking**: Model performance and validation metrics

#### 2. Interactive Analysis

- **Scenario Analysis**: Interactive stress testing capabilities
- **Drill-down Analysis**: Portfolio segmentation and attribution
- **Historical Trends**: Time series analysis and trending

#### 3. Customization Options

```python
# Add custom charts
dashboard.add_custom_chart(
    chart_type='risk_heatmap',
    data_source='portfolio_risk',
    update_frequency='daily'
)

# Configure alerts
dashboard.configure_alerts(
    alert_rules=[
        {'metric': 'cet1_ratio', 'threshold': 0.045, 'condition': 'below'},
        {'metric': 'var_utilization', 'threshold': 0.90, 'condition': 'above'}
    ]
)
```

## Report Generation

### Automated Reporting

#### 1. Executive Summary Reports

```python
from src.dashboard import ReportGenerator

# Initialize report generator
report_gen = ReportGenerator(config_path='config/model_config.yaml')

# Generate executive summary
exec_report = report_gen.generate_executive_summary(
    data_date='2024-01-31',
    include_sections=['risk_overview', 'key_metrics', 'alerts', 'recommendations']
)

# Export to multiple formats
report_gen.export_report(
    report=exec_report,
    output_formats=['pdf', 'html', 'excel'],
    output_path='reports/executive_summary_2024_01.pdf'
)
```

#### 2. Regulatory Reports

```python
# Generate CCAR report
ccar_report = report_gen.generate_regulatory_report(
    report_type='ccar',
    stress_results=stress_results,
    capital_plan=capital_plan_data
)

# Generate Basel III report
basel_report = report_gen.generate_regulatory_report(
    report_type='basel3',
    capital_data=capital_results,
    rwa_data=rwa_results
)
```

#### 3. Custom Reports

```python
# Create custom report template
custom_template = {
    'title': 'Monthly Risk Review',
    'sections': [
        {'type': 'summary_table', 'data': 'key_metrics'},
        {'type': 'chart', 'chart_type': 'var_trend', 'period': '12M'},
        {'type': 'text', 'content': 'risk_commentary'},
        {'type': 'table', 'data': 'top_risks'}
    ]
}

# Generate custom report
custom_report = report_gen.generate_custom_report(
    template=custom_template,
    data_sources={
        'key_metrics': monthly_metrics,
        'risk_commentary': risk_analysis_text,
        'top_risks': top_risk_exposures
    }
)
```

### Report Scheduling

```python
# Schedule automated reports
report_gen.schedule_reports([
    {
        'report_type': 'executive_summary',
        'frequency': 'monthly',
        'recipients': ['risk@company.com', 'management@company.com'],
        'format': 'pdf'
    },
    {
        'report_type': 'regulatory_filing',
        'frequency': 'quarterly',
        'recipients': ['compliance@company.com'],
        'format': 'excel'
    }
])
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Data Loading Issues

**Problem**: "File not found" error when loading data
```python
# Solution: Check file path and permissions
import os
file_path = 'data/raw/portfolio_data.csv'
if os.path.exists(file_path):
    print(f"File exists: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
else:
    print(f"File not found: {file_path}")
    print(f"Current directory: {os.getcwd()}")
```

**Problem**: Data validation failures
```python
# Solution: Review validation rules and data quality
validation_details = validator.get_validation_details(portfolio_data)
for rule, result in validation_details.items():
    if not result['passed']:
        print(f"Failed rule: {rule}")
        print(f"Details: {result['details']}")
```

#### 2. Model Training Issues

**Problem**: Model convergence issues
```python
# Solution: Adjust hyperparameters
model_config = {
    'n_estimators': 50,  # Reduce complexity
    'learning_rate': 0.05,  # Lower learning rate
    'max_depth': 4,  # Reduce depth
    'early_stopping_rounds': 10
}

credit_model = CreditRiskModel(model_config=model_config)
```

**Problem**: Memory issues with large datasets
```python
# Solution: Use chunked processing
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    chunk_result = model.predict(chunk)
    results.append(chunk_result)

final_results = pd.concat(results, ignore_index=True)
```

#### 3. Dashboard Issues

**Problem**: Dashboard not loading
```python
# Solution: Check port availability and dependencies
import socket

def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

if check_port(8050):
    print("Port 8050 is in use, try a different port")
else:
    print("Port 8050 is available")
```

#### 4. Performance Optimization

**Problem**: Slow model execution
```python
# Solution: Enable parallel processing
import multiprocessing

# Set number of cores for model training
n_cores = multiprocessing.cpu_count() - 1

model_config = {
    'n_jobs': n_cores,
    'random_state': 42
}

# Use vectorized operations
import numpy as np
# Instead of loops, use numpy operations
vectorized_result = np.where(condition, value_if_true, value_if_false)
```

### Getting Help

#### 1. Log Analysis

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Run problematic code with logging enabled
```

#### 2. System Diagnostics

```python
# Check system resources
import psutil

print(f"CPU Usage: {psutil.cpu_percent()}%")
print(f"Memory Usage: {psutil.virtual_memory().percent}%")
print(f"Disk Usage: {psutil.disk_usage('/').percent}%")
```

#### 3. Version Compatibility

```python
# Check package versions
import pkg_resources

required_packages = ['pandas', 'numpy', 'scikit-learn', 'lightgbm']
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not installed")
```

For additional support, please refer to:
- System logs in `logs/` directory
- Configuration documentation in `docs/configuration.md`
- API documentation in `docs/api_reference.md`
- Community forum at [support-url]