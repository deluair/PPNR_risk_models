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
- **Sample Data Generation**: Synthetic data creation for testing and demonstration

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
git clone https://github.com/deluair/PPNR_risk_models.git
cd PPNR_risk_models
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
# Configuration file is already provided at config/model_config.yaml
# Edit config/model_config.yaml with your specific settings if needed
```

## 📡 Using Real Data Sources

This project supports real data ingestion with graceful fallback to synthetic data when real sources are unavailable. Real sources are used in the demo and throughout the library where possible.

- Market Data: Yahoo Finance via the yfinance library (no API key required).
- Economic Indicators: FRED API (Federal Reserve Bank of St. Louis) via fredapi.
- Bank Metrics: FDIC BankFind public API (no API key required).

Quick setup for real data:
1) Copy environment template and set your FRED API key
```bash
cp .env.example .env  # On Windows: copy .env.example .env
# Edit .env and set:
# FRED_API_KEY=your_fred_api_key_here
```
2) Ensure dependencies are installed (covered by requirements.txt)
```bash
pip install -r requirements.txt
```
3) Run the demo normally
```bash
python demo_ppnr_system.py
```

Behavior and fallbacks:
- If the FRED API key is missing or fredapi is not installed, the system logs a warning and uses synthetic macro data.
- If Yahoo Finance is unreachable, market data falls back to synthetic.
- If FDIC API calls fail or return incomplete data, synthetic bank metrics are generated for demonstration.

Configuration notes:
- Environment: FRED key is read from the FRED_API_KEY variable (see .env.example).
- Data configuration: see config/data_config.yaml for provider settings and mappings.
- Detailed guide: see README_REAL_DATA.md for step-by-step instructions and troubleshooting.

## 🏃‍♂️ Quick Start

### 1. Generate Sample Data and Run Demo
```bash
# Generate comprehensive sample data
python generate_sample_data.py

# Run the complete system demonstration
python demo_ppnr_system.py
```

### 2. Data Processing Pipeline
```python
from src.data import DataLoader, DataValidator

# Initialize data processor
loader = DataLoader(config_path='config/model_config.yaml')
validator = DataValidator(config={'data_validation': {'completeness_threshold': 0.95}})

# Load and validate data
raw_data = loader.load_portfolio_data('data/processed/portfolio_data.csv')
validation_result = validator.validate_data(raw_data, data_type='portfolio')
```

### 3. Risk Model Training
```python
from src.risk_factors.credit_risk import CreditRiskModel
from src.risk_factors.market_risk import MarketRiskModel

# Credit risk modeling
credit_model = CreditRiskModel(config={'credit_risk': {}})
pd_predictions = credit_model.predict_pd(portfolio_data)

# Market risk modeling
market_model = MarketRiskModel(config={'market_risk': {}})
var_results = market_model.calculate_var(portfolio_data, confidence_level=0.99)
```

### 4. Regulatory Compliance
```python
from src.regulatory.ccar_compliance import CCARCompliance
from src.regulatory.basel_iii import BaselIII

# CCAR stress testing
ccar = CCARCompliance(config={'ccar': {}})
stress_results = ccar.run_stress_test(
    portfolio_data=validated_data,
    scenarios=['baseline', 'adverse', 'severely_adverse']
)

# Basel III capital calculations
basel = BaselIII(config={'basel_iii': {}})
capital_ratios = basel.calculate_capital_ratios(validated_data)
```

### 5. Dashboard Launch
```python
from src.dashboard.risk_dashboard import RiskDashboard

# Initialize and run dashboard
dashboard = RiskDashboard(config_path='config/model_config.yaml')
dashboard.run(host='localhost', port=8050, debug=False)
```

## 📁 Project Structure

```
PPNR_risk_models/
├── config/
│   └── model_config.yaml          # System configuration
├── data/
│   ├── processed/                 # Generated sample data
│   ├── raw/                      # Raw data files
│   └── scenarios/                # Stress test scenarios
│       ├── adverse_scenario.csv
│       ├── baseline_scenario.csv
│       └── severely_adverse_scenario.csv
├── docs/
│   ├── api_reference.md          # API documentation
│   ├── model_methodology.md      # Model methodology guide
│   └── user_guide.md            # User guide
├── src/
│   ├── dashboard/               # Dashboard and visualization
│   │   ├── report_generator.py
│   │   ├── risk_dashboard.py
│   │   └── visualization_engine.py
│   ├── data/                    # Data processing pipeline
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── bank_metrics_processor.py
│   │   ├── economic_indicators_processor.py
│   │   └── market_data_processor.py
│   ├── models/                  # Core PPNR models
│   │   ├── base_model.py
│   │   ├── fee_income_model.py
│   │   ├── nii_model.py
│   │   └── trading_revenue_model.py
│   ├── regulatory/              # Regulatory compliance
│   │   ├── basel_iii.py
│   │   ├── ccar_compliance.py
│   │   ├── dfast_compliance.py
│   │   ├── capital_calculator.py
│   │   └── regulatory_reporter.py
│   ├── risk_factors/            # Risk factor modeling
│   │   ├── credit_risk.py
│   │   ├── market_risk.py
│   │   ├── operational_risk.py
│   │   └── risk_integration.py
│   └── stress_testing/          # Stress testing framework
│       ├── scenario_generator.py
│       ├── stress_tester.py
│       └── model_validator.py
├── tests/                       # Unit and integration tests
├── demo_ppnr_system.py         # Complete system demonstration
├── generate_sample_data.py     # Sample data generation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎯 System Demonstration

The system includes a comprehensive demonstration that showcases all key capabilities:

### Running the Demo
```bash
# Generate sample data (10,000 loans, market data, scenarios)
python generate_sample_data.py

# Run complete system demonstration
python demo_ppnr_system.py
```

### Demo Features
- **Data Validation**: Quality assessment of portfolio data
- **Credit Risk Modeling**: PD/LGD calculations and portfolio analysis
- **Market Risk Assessment**: VaR calculations and risk metrics
- **Stress Testing**: Multi-scenario stress testing with regulatory scenarios
- **Regulatory Compliance**: Basel III and CCAR compliance monitoring
- **PPNR Projections**: Revenue and loss forecasting
- **Automated Reporting**: Summary report generation

### Sample Output
```
============================================================
  PPNR RISK MODELS SYSTEM DEMONSTRATION
============================================================

✓ Data Validation & Quality Assessment
  - Portfolio Quality Score: 0.95
  - Data Completeness: 100.0%
  - Missing Values: 0

✓ Credit Risk Modeling & Portfolio Analysis
  - Total Exposure: $634,013,715
  - Average PD: 1.97%
  - Expected Loss Rate: 0.79%

✓ Market Risk Assessment & VaR Calculation
  - Portfolio VaR (99%): $2,847,061
  - Expected Shortfall: $3,421,273

✓ Stress Testing with Multiple Scenarios
  - Baseline Loss Rate: 0.79%
  - Adverse Loss Rate: 2.34%
  - Severely Adverse Loss Rate: 4.12%

✓ Regulatory Compliance Monitoring
  - Tier 1 Capital Ratio: 12.06%
  - Status: WELL CAPITALIZED

✓ Summary report saved to: demo_summary_[timestamp].txt
```

## 🔧 Model Components

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

### Credit Risk Models
- Probability of Default (PD) modeling
- Loss Given Default (LGD) estimation
- Exposure at Default (EAD) calculations
- Portfolio segmentation and concentration analysis

### Market Risk Models
- Value at Risk (VaR) calculations
- Expected Shortfall modeling
- Multi-factor risk modeling
- Correlation and volatility analysis

### Stress Testing Framework
- Severely adverse scenario modeling
- Model performance under stress
- Regulatory capital impact assessment
- Multi-horizon stress projections

## 📊 Regulatory Compliance

The system supports comprehensive regulatory requirements:

- **CCAR**: Comprehensive Capital Analysis and Review
- **DFAST**: Dodd-Frank Act Stress Testing
- **Basel III**: Capital adequacy requirements
- **SR 11-7**: Model Risk Management guidance

### Key Compliance Features
- Automated regulatory reporting
- Capital adequacy calculations
- Stress testing scenarios
- Model validation frameworks
- Risk governance workflows

## 🧪 Testing

Run the test suite to validate system functionality:

```bash
# Run basic system test
python simple_test.py

# Run comprehensive test suite
python test_system.py

# Run unit tests
python -m pytest tests/
```

## 📈 Performance Metrics

The system is designed to handle enterprise-scale data:

- **Portfolio Size**: 10,000+ loans tested
- **Processing Speed**: <30 seconds for full demonstration
- **Memory Usage**: <2GB for standard datasets
- **Scalability**: Designed for millions of records

## 🤝 Contributing

We welcome contributions to improve the PPNR Risk Models System:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or support:

- Create an issue on GitHub
- Review the documentation in the `docs/` directory
- Check the demo output for system capabilities

## 🔄 Version History

- **v1.0.0**: Initial release with complete PPNR modeling framework
  - Full regulatory compliance suite
  - Comprehensive risk modeling
  - Interactive demonstration
  - Sample data generation
  - Robust error handling and fallback mechanisms