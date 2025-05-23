# insurance-pricing-engine

A full-featured life insurance engine built in Python, designed to repolicate key componenets of production-grade actuairal models. This project incorporates dynamic mortality modeling, interest rate simulation, lapse assumptions, premium solving, and profit testing - with outputs formatted for Power BI analysis.

Features:

Whole Life Insurance Modeling
- Fixed face value, level or term-based premium structures
- Lapse-adjusted survival and benefit calculations

Mortality Modeling
- Lee-Carter decompositon on VBT (2001, 2008, 2015)
- Forecasts mortality improvements using SVD and time-trend regression

Interest Rates Modeling
- CIR (Cox-Ingersoll-Ross) model calibrated from historical data
- Monte Carlo simulation of monthly rates, aggregated to annual

Premium Solve
- Brent's method used to solve for level premium given target margin

Profit Testing
- IRR, NPV, and breakeven year calculated from projected cfs

PowerBI Export
- Yearly, policy-level cfs exported to Excel for BI/dashboard use

Technology Used:
- Python (Pandas, NumPy, SciPy, Statsmodels)
- Excel / Power BI for visualization
- Lee-Carter and CIR models implemented from scratch

Project Structure:
- pricing_model.py: Main pricing engine
- CIR.py: CIR interest rate model
- Lee_Carter_mortality.py: Lee-Carter mortality projection
- Final_Mortality_Projections.xlsx: Lee-Carter output
- performance_metrics.py: Profit metrics
- Insurance_Dataset.csv: Input policy data
- Yearly_Cashflows.xlsx: Output for visualization - too large to bring into github
- Pricing Enging Dashboard.pbix: Power BI visualization
