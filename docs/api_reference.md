# API Reference - PPNR Risk Models System

## Overview

This document provides comprehensive API documentation for all modules, classes, and functions in the PPNR Risk Models System.

## Table of Contents

1. [Data Module](#data-module)
2. [Risk Factors Module](#risk-factors-module)
3. [Regulatory Module](#regulatory-module)
4. [Dashboard Module](#dashboard-module)
5. [Utilities](#utilities)

---

## Data Module

### DataLoader

**Class**: `src.data.data_loader.DataLoader`

Handles loading and preprocessing of various data types for risk modeling.

#### Constructor

```python
DataLoader(config_path: str = None, **kwargs)
```

**Parameters:**
- `config_path` (str, optional): Path to configuration file
- `**kwargs`: Additional configuration parameters

#### Methods

##### load_portfolio_data()

```python
load_portfolio_data(
    file_path: str,
    file_format: str = 'csv',
    **kwargs
) -> pd.DataFrame
```

Load portfolio exposure data from various file formats.

**Parameters:**
- `file_path` (str): Path to the data file
- `file_format` (str): File format ('csv', 'excel', 'parquet')
- `**kwargs`: Additional pandas read parameters

**Returns:**
- `pd.DataFrame`: Loaded portfolio data

**Example:**
```python
loader = DataLoader('config/model_config.yaml')
portfolio = loader.load_portfolio_data('data/portfolio.csv')
```

##### load_market_data()

```python
load_market_data(
    file_path: str,
    date_column: str = 'date',
    price_columns: List[str] = None
) -> pd.DataFrame
```

Load market price and return data.

**Parameters:**
- `file_path` (str): Path to market data file
- `date_column` (str): Name of date column
- `price_columns` (List[str]): List of price column names

**Returns:**
- `pd.DataFrame`: Market data with calculated returns

##### load_macro_data()

```python
load_macro_data(
    file_path: str,
    scenario: str = None
) -> pd.DataFrame
```

Load macroeconomic scenario data.

**Parameters:**
- `file_path` (str): Path to macro data file
- `scenario` (str, optional): Specific scenario to load

**Returns:**
- `pd.DataFrame`: Macroeconomic data

### DataValidator

**Class**: `src.data.data_validator.DataValidator`

Provides comprehensive data validation and quality assessment.

#### Constructor

```python
DataValidator(validation_rules: Dict = None)
```

**Parameters:**
- `validation_rules` (Dict, optional): Custom validation rules

#### Methods

##### validate_data()

```python
validate_data(
    data: pd.DataFrame,
    data_type: str = 'portfolio'
) -> Dict[str, Any]
```

Perform comprehensive data validation.

**Parameters:**
- `data` (pd.DataFrame): Data to validate
- `data_type` (str): Type of data ('portfolio', 'market', 'macro')

**Returns:**
- `Dict[str, Any]`: Validation results with status and details

**Example:**
```python
validator = DataValidator()
results = validator.validate_data(portfolio_data, 'portfolio')
if results['is_valid']:
    print("Data validation passed")
```

##### check_completeness()

```python
check_completeness(
    data: pd.DataFrame,
    threshold: float = 0.95
) -> Dict[str, float]
```

Check data completeness by column.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `threshold` (float): Minimum completeness threshold

**Returns:**
- `Dict[str, float]`: Completeness ratios by column

##### detect_outliers()

```python
detect_outliers(
    data: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame
```

Detect outliers in numerical columns.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `method` (str): Detection method ('iqr', 'zscore', 'isolation_forest')
- `threshold` (float): Outlier threshold

**Returns:**
- `pd.DataFrame`: Boolean mask indicating outliers

---

## Risk Factors Module

### CreditRiskModel

**Class**: `src.risk_factors.credit_risk.CreditRiskModel`

Implements credit risk modeling including PD, LGD, and EAD models.

#### Constructor

```python
CreditRiskModel(
    config_path: str = None,
    model_config: Dict = None
)
```

**Parameters:**
- `config_path` (str, optional): Path to configuration file
- `model_config` (Dict, optional): Model configuration dictionary

#### Methods

##### train_pd_model()

```python
train_pd_model(
    data: pd.DataFrame,
    target_column: str = 'default_flag',
    feature_columns: List[str] = None,
    validation_split: float = 0.2
) -> Dict[str, Any]
```

Train probability of default model.

**Parameters:**
- `data` (pd.DataFrame): Training data
- `target_column` (str): Target variable column name
- `feature_columns` (List[str]): Feature column names
- `validation_split` (float): Validation set proportion

**Returns:**
- `Dict[str, Any]`: Training results including metrics and model

**Example:**
```python
credit_model = CreditRiskModel()
results = credit_model.train_pd_model(
    data=training_data,
    target_column='default_flag',
    feature_columns=['debt_ratio', 'roa', 'current_ratio']
)
print(f"Model AUC: {results['auc']:.3f}")
```

##### predict_pd()

```python
predict_pd(
    data: pd.DataFrame,
    model_name: str = 'default'
) -> np.ndarray
```

Generate PD predictions.

**Parameters:**
- `data` (pd.DataFrame): Input data for prediction
- `model_name` (str): Name of trained model to use

**Returns:**
- `np.ndarray`: PD predictions

##### calculate_portfolio_risk()

```python
calculate_portfolio_risk(
    exposures: pd.DataFrame,
    pd_predictions: np.ndarray,
    lgd_predictions: np.ndarray,
    correlation_matrix: np.ndarray = None
) -> Dict[str, float]
```

Calculate portfolio-level credit risk metrics.

**Parameters:**
- `exposures` (pd.DataFrame): Exposure data
- `pd_predictions` (np.ndarray): PD predictions
- `lgd_predictions` (np.ndarray): LGD predictions
- `correlation_matrix` (np.ndarray, optional): Asset correlation matrix

**Returns:**
- `Dict[str, float]`: Portfolio risk metrics

### MarketRiskModel

**Class**: `src.risk_factors.market_risk.MarketRiskModel`

Implements market risk modeling including VaR and stress testing.

#### Constructor

```python
MarketRiskModel(
    config_path: str = None,
    model_config: Dict = None
)
```

#### Methods

##### calculate_var()

```python
calculate_var(
    portfolio_data: pd.DataFrame,
    method: str = 'historical_simulation',
    confidence_level: float = 0.99,
    holding_period: int = 1
) -> Dict[str, float]
```

Calculate Value at Risk using specified method.

**Parameters:**
- `portfolio_data` (pd.DataFrame): Portfolio positions and returns
- `method` (str): VaR calculation method
- `confidence_level` (float): Confidence level (0.95, 0.99, 0.999)
- `holding_period` (int): Holding period in days

**Returns:**
- `Dict[str, float]`: VaR and related risk metrics

**Example:**
```python
market_model = MarketRiskModel()
var_results = market_model.calculate_var(
    portfolio_data=positions,
    method='monte_carlo',
    confidence_level=0.99
)
print(f"1-Day VaR: ${var_results['var']:,.2f}")
```

##### run_stress_tests()

```python
run_stress_tests(
    portfolio_data: pd.DataFrame,
    scenarios: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]
```

Run stress tests on portfolio.

**Parameters:**
- `portfolio_data` (pd.DataFrame): Portfolio data
- `scenarios` (Dict): Stress scenarios definition

**Returns:**
- `Dict[str, Dict[str, float]]`: Stress test results by scenario

### OperationalRiskModel

**Class**: `src.risk_factors.operational_risk.OperationalRiskModel`

Implements operational risk modeling using Loss Distribution Approach.

#### Constructor

```python
OperationalRiskModel(
    config_path: str = None,
    model_config: Dict = None
)
```

#### Methods

##### fit_loss_distribution()

```python
fit_loss_distribution(
    loss_data: pd.DataFrame,
    business_line: str,
    event_type: str
) -> Dict[str, Any]
```

Fit frequency and severity distributions to loss data.

**Parameters:**
- `loss_data` (pd.DataFrame): Historical loss data
- `business_line` (str): Business line identifier
- `event_type` (str): Event type identifier

**Returns:**
- `Dict[str, Any]`: Fitted distribution parameters

##### calculate_operational_var()

```python
calculate_operational_var(
    business_lines: List[str],
    confidence_level: float = 0.999,
    time_horizon: int = 365
) -> Dict[str, float]
```

Calculate operational VaR using Monte Carlo simulation.

**Parameters:**
- `business_lines` (List[str]): List of business lines
- `confidence_level` (float): Confidence level
- `time_horizon` (int): Time horizon in days

**Returns:**
- `Dict[str, float]`: Operational risk metrics

---

## Regulatory Module

### CCARCompliance

**Class**: `src.regulatory.ccar_compliance.CCARCompliance`

Implements CCAR stress testing and compliance requirements.

#### Constructor

```python
CCARCompliance(
    config_path: str = None,
    bank_config: Dict = None
)
```

#### Methods

##### run_stress_test()

```python
run_stress_test(
    portfolio_data: pd.DataFrame,
    scenarios: List[str],
    projection_horizon: int = 9
) -> Dict[str, Dict[str, Any]]
```

Execute comprehensive CCAR stress test.

**Parameters:**
- `portfolio_data` (pd.DataFrame): Portfolio data
- `scenarios` (List[str]): Stress scenarios to run
- `projection_horizon` (int): Projection horizon in quarters

**Returns:**
- `Dict[str, Dict[str, Any]]`: Stress test results by scenario

**Example:**
```python
ccar = CCARCompliance()
results = ccar.run_stress_test(
    portfolio_data=data,
    scenarios=['baseline', 'adverse', 'severely_adverse']
)
```

##### calculate_capital_projections()

```python
calculate_capital_projections(
    stress_results: Dict,
    capital_actions: Dict = None
) -> Dict[str, pd.DataFrame]
```

Calculate capital ratio projections under stress.

**Parameters:**
- `stress_results` (Dict): Stress test results
- `capital_actions` (Dict, optional): Planned capital actions

**Returns:**
- `Dict[str, pd.DataFrame]`: Capital projections by scenario

### BaselCompliance

**Class**: `src.regulatory.basel_compliance.BaselCompliance`

Implements Basel III capital and liquidity requirements.

#### Constructor

```python
BaselCompliance(
    config_path: str = None,
    jurisdiction: str = 'US'
)
```

#### Methods

##### calculate_rwa()

```python
calculate_rwa(
    portfolio_data: pd.DataFrame,
    approach: str = 'standardized'
) -> Dict[str, float]
```

Calculate risk-weighted assets.

**Parameters:**
- `portfolio_data` (pd.DataFrame): Portfolio exposure data
- `approach` (str): Calculation approach ('standardized', 'internal_ratings_based')

**Returns:**
- `Dict[str, float]`: RWA by risk type

##### calculate_capital_ratios()

```python
calculate_capital_ratios(
    capital_data: pd.DataFrame,
    rwa_results: Dict[str, float]
) -> Dict[str, float]
```

Calculate regulatory capital ratios.

**Parameters:**
- `capital_data` (pd.DataFrame): Capital component data
- `rwa_results` (Dict): Risk-weighted assets

**Returns:**
- `Dict[str, float]`: Capital ratios

---

## Dashboard Module

### RiskDashboard

**Class**: `src.dashboard.risk_dashboard.RiskDashboard`

Main dashboard application for risk monitoring and visualization.

#### Constructor

```python
RiskDashboard(
    config_path: str = None,
    theme: str = 'corporate'
)
```

#### Methods

##### register_models()

```python
register_models(
    models: List[Any]
) -> None
```

Register risk models with the dashboard.

**Parameters:**
- `models` (List[Any]): List of risk model instances

##### run()

```python
run(
    host: str = 'localhost',
    port: int = 8050,
    debug: bool = False
) -> None
```

Launch the dashboard application.

**Parameters:**
- `host` (str): Host address
- `port` (int): Port number
- `debug` (bool): Debug mode flag

**Example:**
```python
dashboard = RiskDashboard()
dashboard.register_models([credit_model, market_model])
dashboard.run(host='0.0.0.0', port=8050)
```

### VisualizationEngine

**Class**: `src.dashboard.visualization_engine.VisualizationEngine`

Provides comprehensive charting and visualization capabilities.

#### Constructor

```python
VisualizationEngine(
    theme: str = 'plotly_white',
    color_scheme: str = 'corporate'
)
```

#### Methods

##### create_risk_heatmap()

```python
create_risk_heatmap(
    risk_data: pd.DataFrame,
    risk_metric: str = 'var',
    aggregation_level: str = 'business_line'
) -> go.Figure
```

Create risk heatmap visualization.

**Parameters:**
- `risk_data` (pd.DataFrame): Risk data
- `risk_metric` (str): Risk metric to visualize
- `aggregation_level` (str): Aggregation level

**Returns:**
- `go.Figure`: Plotly figure object

##### create_var_distribution()

```python
create_var_distribution(
    pnl_data: np.ndarray,
    var_level: float,
    confidence_levels: List[float] = [0.95, 0.99, 0.999]
) -> go.Figure
```

Create VaR distribution chart.

**Parameters:**
- `pnl_data` (np.ndarray): P&L distribution data
- `var_level` (float): VaR confidence level
- `confidence_levels` (List[float]): Confidence levels to display

**Returns:**
- `go.Figure`: VaR distribution chart

### ReportGenerator

**Class**: `src.dashboard.report_generator.ReportGenerator`

Automated report generation for various stakeholders.

#### Constructor

```python
ReportGenerator(
    config_path: str = None,
    template_path: str = 'templates/'
)
```

#### Methods

##### generate_executive_summary()

```python
generate_executive_summary(
    data_date: str,
    include_sections: List[str] = None
) -> Dict[str, Any]
```

Generate executive summary report.

**Parameters:**
- `data_date` (str): Report date
- `include_sections` (List[str]): Sections to include

**Returns:**
- `Dict[str, Any]`: Generated report content

##### export_report()

```python
export_report(
    report: Dict[str, Any],
    output_formats: List[str] = ['pdf'],
    output_path: str = None
) -> List[str]
```

Export report to specified formats.

**Parameters:**
- `report` (Dict): Report content
- `output_formats` (List[str]): Output formats
- `output_path` (str): Output file path

**Returns:**
- `List[str]`: List of generated file paths

---

## Utilities

### Configuration Management

#### load_config()

```python
load_config(config_path: str) -> Dict[str, Any]
```

Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

### Logging Utilities

#### setup_logging()

```python
setup_logging(
    log_level: str = 'INFO',
    log_file: str = None
) -> logging.Logger
```

Setup logging configuration.

**Parameters:**
- `log_level` (str): Logging level
- `log_file` (str, optional): Log file path

**Returns:**
- `logging.Logger`: Configured logger instance

### Data Utilities

#### calculate_returns()

```python
calculate_returns(
    prices: pd.Series,
    method: str = 'simple'
) -> pd.Series
```

Calculate returns from price series.

**Parameters:**
- `prices` (pd.Series): Price data
- `method` (str): Return calculation method ('simple', 'log')

**Returns:**
- `pd.Series`: Calculated returns

#### winsorize_data()

```python
winsorize_data(
    data: pd.DataFrame,
    limits: Tuple[float, float] = (0.01, 0.01)
) -> pd.DataFrame
```

Winsorize data to handle outliers.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `limits` (Tuple[float, float]): Lower and upper winsorization limits

**Returns:**
- `pd.DataFrame`: Winsorized data

---

## Error Handling

### Custom Exceptions

#### DataValidationError

```python
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass
```

#### ModelTrainingError

```python
class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass
```

#### ConfigurationError

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass
```

---

## Type Hints and Enums

### Common Type Definitions

```python
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series
Array = np.ndarray
ConfigDict = Dict[str, Any]
```

### Enums

#### CreditRating

```python
from enum import Enum

class CreditRating(Enum):
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
```

#### RiskFactorType

```python
class RiskFactorType(Enum):
    EQUITY = "equity"
    INTEREST_RATE = "interest_rate"
    CREDIT_SPREAD = "credit_spread"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    VOLATILITY = "volatility"
```

---

## Examples and Best Practices

### Complete Workflow Example

```python
from src.data import DataLoader, DataValidator
from src.risk_factors import CreditRiskModel, MarketRiskModel
from src.regulatory import CCARCompliance
from src.dashboard import RiskDashboard

# 1. Data Loading and Validation
loader = DataLoader('config/model_config.yaml')
validator = DataValidator()

portfolio_data = loader.load_portfolio_data('data/portfolio.csv')
validation_results = validator.validate_data(portfolio_data)

if not validation_results['is_valid']:
    raise DataValidationError("Data validation failed")

# 2. Model Training
credit_model = CreditRiskModel()
pd_results = credit_model.train_pd_model(portfolio_data)

market_model = MarketRiskModel()
var_results = market_model.calculate_var(portfolio_data)

# 3. Regulatory Compliance
ccar = CCARCompliance()
stress_results = ccar.run_stress_test(
    portfolio_data,
    scenarios=['baseline', 'adverse', 'severely_adverse']
)

# 4. Dashboard Launch
dashboard = RiskDashboard()
dashboard.register_models([credit_model, market_model])
dashboard.run()
```

### Error Handling Best Practices

```python
import logging
from src.exceptions import DataValidationError, ModelTrainingError

logger = logging.getLogger(__name__)

try:
    # Risk model operations
    results = credit_model.train_pd_model(data)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
    # Handle validation error
except ModelTrainingError as e:
    logger.error(f"Model training failed: {e}")
    # Handle training error
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

For more detailed examples and tutorials, see the User Guide documentation.