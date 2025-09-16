# Model Methodology Documentation

## Overview

This document provides detailed methodology for the PPNR Risk Models System, covering the mathematical foundations, statistical approaches, and implementation details for each risk component.

## 1. Credit Risk Modeling

### 1.1 Probability of Default (PD) Models

#### Methodology
The PD models use a combination of logistic regression and gradient boosting to predict default probabilities:

```
PD(t) = 1 / (1 + exp(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)))
```

Where:
- `PD(t)` = Probability of default at time t
- `βᵢ` = Model coefficients
- `Xᵢ` = Risk factors (financial ratios, macroeconomic variables)

#### Key Features
- **Financial Ratios**: Debt-to-equity, current ratio, interest coverage
- **Macroeconomic Variables**: GDP growth, unemployment rate, interest rates
- **Behavioral Variables**: Payment history, utilization rates
- **Industry Factors**: Sector-specific risk indicators

#### Model Validation
- **Discriminatory Power**: ROC-AUC > 0.70
- **Calibration**: Hosmer-Lemeshow test p-value > 0.05
- **Stability**: Population Stability Index (PSI) < 0.25

### 1.2 Loss Given Default (LGD) Models

#### Methodology
LGD models use beta regression to handle the bounded nature of recovery rates:

```
LGD = 1 - Recovery Rate
Recovery Rate ~ Beta(α, β)
```

#### Key Factors
- **Collateral Type**: Secured vs. unsecured exposures
- **Seniority**: Senior vs. subordinated debt
- **Industry**: Recovery rates by sector
- **Economic Conditions**: Stress impact on recoveries

### 1.3 Exposure at Default (EAD) Models

#### Methodology
EAD models predict the exposure amount at the time of default:

```
EAD = Outstanding Balance + (Credit Conversion Factor × Undrawn Amount)
```

#### Credit Conversion Factors (CCF)
- **Committed Lines**: Historical drawdown patterns
- **Letters of Credit**: Regulatory and internal estimates
- **Derivatives**: Potential future exposure calculations

## 2. Market Risk Modeling

### 2.1 Value at Risk (VaR) Models

#### Historical Simulation
```
VaR(α) = -Percentile(P&L Distribution, α)
```

#### Parametric VaR
```
VaR(α) = -μ + σ × Φ⁻¹(α)
```

Where:
- `α` = Confidence level (typically 99% or 95%)
- `μ` = Expected return
- `σ` = Portfolio volatility
- `Φ⁻¹` = Inverse normal distribution

#### Monte Carlo Simulation
1. Generate random scenarios for risk factors
2. Price portfolio under each scenario
3. Calculate P&L distribution
4. Extract VaR at desired confidence level

### 2.2 Expected Shortfall (ES)

```
ES(α) = E[Loss | Loss > VaR(α)]
```

### 2.3 Multi-Factor Risk Models

#### Factor Model Structure
```
R(t) = α + β₁F₁(t) + β₂F₂(t) + ... + βₙFₙ(t) + ε(t)
```

Where:
- `R(t)` = Asset return at time t
- `Fᵢ(t)` = Risk factor i at time t
- `βᵢ` = Factor loading for factor i
- `ε(t)` = Idiosyncratic risk

#### Risk Factors
- **Equity Factors**: Market, size, value, momentum
- **Fixed Income Factors**: Level, slope, curvature
- **Currency Factors**: Major currency pairs
- **Commodity Factors**: Energy, metals, agriculture

## 3. Operational Risk Modeling

### 3.1 Loss Distribution Approach (LDA)

#### Frequency Model
```
N ~ Poisson(λ)
```

#### Severity Model
```
X ~ LogNormal(μ, σ²)
or
X ~ Generalized Pareto Distribution
```

#### Annual Loss Distribution
```
S = X₁ + X₂ + ... + Xₙ
```

### 3.2 Key Risk Indicators (KRI)

#### Statistical Models
- **Threshold Models**: Alert when KRI exceeds limits
- **Trend Analysis**: Identify deteriorating risk patterns
- **Correlation Analysis**: Relationships between KRIs and losses

### 3.3 Scenario Analysis

#### Scenario Types
- **Historical Scenarios**: Based on past events
- **Hypothetical Scenarios**: Forward-looking stress events
- **Regulatory Scenarios**: Supervisory stress tests

## 4. Risk Integration and Correlation

### 4.1 Copula Models

#### Gaussian Copula
```
C(u₁, u₂, ..., uₙ; ρ) = Φₙ(Φ⁻¹(u₁), Φ⁻¹(u₂), ..., Φ⁻¹(uₙ); ρ)
```

#### t-Copula
```
C(u₁, u₂, ..., uₙ; ρ, ν) = tₙ,ν(t⁻¹ν(u₁), t⁻¹ν(u₂), ..., t⁻¹ν(uₙ); ρ)
```

### 4.2 Portfolio Risk Aggregation

#### Variance-Covariance Approach
```
σₚ² = w'Σw
```

Where:
- `w` = Portfolio weights vector
- `Σ` = Covariance matrix

#### Monte Carlo Aggregation
1. Simulate correlated risk factor scenarios
2. Calculate individual risk component losses
3. Aggregate to portfolio level
4. Derive loss distribution statistics

## 5. Stress Testing Methodology

### 5.1 Scenario Design

#### Macroeconomic Scenarios
- **Baseline**: Most likely economic path
- **Adverse**: Moderate recession scenario
- **Severely Adverse**: Severe recession with financial stress

#### Scenario Variables
- GDP growth rate
- Unemployment rate
- Interest rates (short and long-term)
- Equity market indices
- Real estate prices
- Credit spreads

### 5.2 Model Stress Testing

#### Direct Stress Testing
- Apply scenario shocks directly to model inputs
- Recalculate risk metrics under stress

#### Indirect Stress Testing
- Model relationships between macro variables and risk parameters
- Propagate scenario impacts through these relationships

### 5.3 Reverse Stress Testing

#### Methodology
1. Define failure threshold (e.g., capital ratio < minimum)
2. Work backwards to identify scenarios causing failure
3. Assess plausibility of identified scenarios

## 6. Model Validation Framework

### 6.1 Quantitative Validation

#### Backtesting
- **VaR Backtesting**: Kupiec and Christoffersen tests
- **PD Calibration**: Binomial test, traffic light approach
- **Model Stability**: PSI, characteristic stability index

#### Benchmarking
- Compare model performance against:
  - Industry benchmarks
  - Regulatory models
  - Alternative methodologies

### 6.2 Qualitative Validation

#### Model Documentation Review
- Methodology appropriateness
- Implementation accuracy
- Assumption validity

#### Data Quality Assessment
- Completeness and accuracy
- Representativeness
- Timeliness

### 6.3 Ongoing Monitoring

#### Performance Monitoring
- Regular backtesting
- Model performance metrics
- Early warning indicators

#### Model Recalibration
- Trigger conditions for recalibration
- Recalibration frequency
- Approval processes

## 7. Regulatory Compliance

### 7.1 CCAR Requirements

#### Capital Planning
- 9-quarter projection horizon
- Baseline and stress scenarios
- Capital action assumptions

#### Model Risk Management
- SR 11-7 compliance
- Model validation requirements
- Documentation standards

### 7.2 DFAST Requirements

#### Stress Testing Components
- Supervisory scenarios
- Company-run stress tests
- Qualitative assessments

### 7.3 Basel III Implementation

#### Risk-Weighted Assets
- Standardized approach
- Internal ratings-based approach
- Operational risk capital

#### Capital Ratios
- Common Equity Tier 1 (CET1)
- Tier 1 Capital Ratio
- Total Capital Ratio
- Leverage Ratio

## 8. Implementation Considerations

### 8.1 Data Requirements

#### Data Sources
- Internal transaction systems
- External market data providers
- Regulatory data feeds
- Macroeconomic databases

#### Data Governance
- Data lineage tracking
- Quality control processes
- Change management procedures

### 8.2 Technology Infrastructure

#### Computational Requirements
- High-performance computing for Monte Carlo simulations
- Distributed processing for large datasets
- Real-time data processing capabilities

#### Model Deployment
- Version control systems
- Automated testing frameworks
- Production monitoring

### 8.3 Governance and Controls

#### Model Governance
- Model development lifecycle
- Approval processes
- Change management

#### Risk Controls
- Model limitations and assumptions
- Use test procedures
- Override policies

## References

1. Basel Committee on Banking Supervision. (2019). "Minimum capital requirements for market risk"
2. Federal Reserve. (2020). "Supervisory Guidance on Model Risk Management"
3. McNeil, A.J., Frey, R., & Embrechts, P. (2015). "Quantitative Risk Management"
4. Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk"
5. Crouhy, M., Galai, D., & Mark, R. (2014). "The Essentials of Risk Management"