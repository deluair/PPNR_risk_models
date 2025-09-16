# PPNR Risk Models System

A comprehensive Pre-Provision Net Revenue (PPNR) risk modeling framework designed for financial institutions to meet regulatory requirements and perform advanced risk analytics.

## 🏦 Overview

The PPNR Risk Models System is an enterprise-grade solution that provides:

- **Regulatory Compliance**: Full support for CCAR, DFAST, and Basel III requirements
- **Advanced Risk Modeling**: Credit, market, and operational risk factor models
- **Stress Testing**: Comprehensive scenario analysis and stress testing capabilities
- **Interactive Dashboards**: Real-time risk monitoring and visualization
- **Automated Reporting**: Regulatory filing and executive reporting automation

## 🚀 Key Features

### Risk Modeling
- **Credit Risk Models**: PD, LGD, and EAD modeling with portfolio segmentation
- **Market Risk Models**: VaR, Expected Shortfall, and multi-factor risk modeling
- **Operational Risk Models**: Loss distribution approach with scenario analysis
- **Risk Integration**: Correlation modeling and portfolio-level risk aggregation

### Regulatory Compliance
- **CCAR Compliance**: Capital analysis and review requirements
- **DFAST Compliance**: Dodd-Frank stress testing framework
- **Basel III**: Capital adequacy and liquidity requirements
- **Automated Reporting**: Regulatory filing generation and validation

### Data Processing
- **Data Validation**: Comprehensive data quality checks and validation rules
- **Data Transformation**: ETL pipelines with error handling and logging
- **Historical Analysis**: Time series analysis and trend identification
- **Real-time Processing**: Streaming data integration capabilities

### Visualization & Reporting
- **Interactive Dashboards**: Risk monitoring with real-time updates
- **Executive Reports**: Automated executive summary generation
- **Regulatory Reports**: Compliance reporting with multiple output formats
- **Custom Analytics**: Flexible charting and visualization engine

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended for large datasets)
- 10GB+ available disk space
- Windows/Linux/macOS support

### Python Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
lightgbm>=4.0.0
plotly>=5.0.0
dash>=2.0.0
jinja2>=3.0.0
weasyprint>=57.0
openpyxl>=3.0.0
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/ppnr-risk-models.git
cd ppnr-risk-models
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp config/model_config.yaml config/config.yaml
# Edit config.yaml with your settings
```

## 🏃‍♂️ Quick Start

### 1. Data Processing Pipeline
```python
from src.data import DataLoader, DataValidator

# Initialize data processor
loader = DataLoader(config_path='config/model_config.yaml')
validator = DataValidator()

# Load and validate data
raw_data = loader.load_portfolio_data('data/raw/portfolio_data.csv')
validated_data = validator.validate_data(raw_data)
```

### 2. Risk Model Training
```python
from src.risk_factors import CreditRiskModel, MarketRiskModel

# Credit risk modeling
credit_model = CreditRiskModel()
credit_model.train_pd_model(validated_data)
credit_model.train_lgd_model(validated_data)

# Market risk modeling
market_model = MarketRiskModel()
var_results = market_model.calculate_var(portfolio_data, confidence_level=0.99)
```

### 3. Regulatory Compliance
```python
from src.regulatory import CCARCompliance, BaselCompliance

# CCAR stress testing
ccar = CCARCompliance()
stress_results = ccar.run_stress_test(
    portfolio_data=validated_data,
    scenarios=['baseline', 'adverse', 'severely_adverse']
)

# Basel III capital calculations
basel = BaselCompliance()
capital_ratios = basel.calculate_capital_ratios(validated_data)
```

### 4. Dashboard Launch
```python
from src.dashboard import RiskDashboard

# Initialize and run dashboard
dashboard = RiskDashboard(config_path='config/model_config.yaml')
dashboard.register_models([credit_model, market_model])
dashboard.run(host='localhost', port=8050, debug=False)
```

## Project Structure

```
PPNR_risk_models/
├── src/
│   ├── models/          # Core PPNR models
│   ├── data/            # Data processing and pipeline
│   ├── stress_testing/  # Stress testing framework
│   ├── risk_factors/    # Risk factor modeling
│   ├── compliance/      # Regulatory compliance features
│   └── visualization/   # Dashboard and plotting utilities
├── data/
│   ├── raw/            # Raw data files
│   ├── processed/      # Cleaned and processed data
│   └── scenarios/      # Stress test scenarios
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for analysis
├── config/             # Configuration files
└── docs/               # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PPNR_risk_models
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Data Setup**: Place your historical banking data in the `data/raw/` directory
2. **Configuration**: Update `config/model_config.yaml` with your specific parameters
3. **Run Models**: Execute the main modeling pipeline:
```bash
python src/main.py --config config/model_config.yaml
```

## Model Components

### Net Interest Income (NII) Models
- Interest rate sensitivity analysis
- Balance sheet forecasting
- Asset-liability management integration

### Fee Income Models
- Service charge forecasting
- Investment banking fees
- Card and payment processing revenue

### Trading Revenue Models
- Market risk factor modeling
- VaR-based revenue forecasting
- Correlation analysis across trading desks

### Stress Testing
- Severely adverse scenario modeling
- Model performance under stress
- Regulatory capital impact assessment

## Regulatory Compliance

The system supports:
- **CCAR**: Comprehensive Capital Analysis and Review
- **DFAST**: Dodd-Frank Act Stress Testing
- **Basel III**: Capital adequacy requirements
- **SR 11-7**: Model Risk Management guidance

## Contributing

Please read our contributing guidelines and ensure all tests pass before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.