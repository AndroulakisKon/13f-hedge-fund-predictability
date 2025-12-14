Hedge Fund Excess Performance Prediction Using Portfolio and Macro Features

Overview

This project evaluates whether next-quarter excess returns of hedge funds can be predicted using portfolio characteristics extracted from 13F filings and macroeconomic variables.

Two datasets are used:

model_dataset

model_dataset_transformed --> Transformed dataset with log-scaled size variables and squared concentration/sector metrics

Four models are compared: OLS, Ridge, Lasso, Random Forest.

The goal is purely predictive: evaluate out-of-sample performance using R² and RMSE.

📁 Project Structure
hedgefund_project/
├── README.md
├── project_report.pdf
├── environment.yml
├── main.py

├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluation.py
│   ├── build_fund_quarter.py
│   ├── build_quarterly_macro_sp500.py
│   ├── merge.py
│   ├── build_model_dataset.py
│   └── build_model_dataset_transformed.py

├── data/
│   ├── raw/                  # Original 13F + macro data
│   └── processed/            # Cleaned and engineered datasets

└── results/
    └── model_comparison.csv

--> Data Sources

13F Holdings: Exported from WhaleWisdom, cleaned and aggregated to fund–quarter level.

Macro + S&P 500: Pulled from Yahoo Finance and stored locally as macro_sp500_quarterly.csv for reproducibility.

All processed datasets required for modeling are already included.

--> Setup

Create the environment:

conda env create -f environment.yml
conda activate hedgefund_project


Environment file:

name: hedgefund_project
channels:
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - statsmodels
  - matplotlib
  - jupyter

--> Running the Project

Execute all models:

python main.py


This script:

Loads both modeling datasets

Builds features and target

Splits into train/test sets

Trains OLS, Ridge, Lasso, Random Forest

Evaluates with R² and RMSE

Saves the comparison table to:

results/model_comparison.csv

--> Key Findings (Short Summary)

Predictability is low: All models yield negative out-of-sample R², consistent with academic literature on 13F-based forecasting.

Ridge performs best: Regularization improves stability and slightly reduces RMSE.

Lasso shrinks most variables to zero: Indicates weak linear predictive signals.

Random Forest underperforms: Nonlinear models struggle with high noise and limited sample size.

Overall, while some features exhibit mild structure, public 13F filings and standard macro variables offer limited forward-looking predictive power for hedge fund excess returns.