# Intelligent Financial Forecasting & Interpretation System
This repository contains the implementation of a PhD research project in Computer Science. The system combines **gradient boosting (LightGBM/XGBoost)** with **explainable AI (SHAP)** and **cooperative game theory** to forecast financial indicators and interpret the predictions.
## Key features
- **Predictive modelling** – LightGBM (or XGBoost) with L2 regularisation, early stopping, and hyperparameter tuning.
- **Local interpretation** – SHAP (Shapley additive explanations) for single predictions (waterfall plots).
- **Global interpretation** – mean absolute SHAP values to rank the most important financial drivers.
- **Interaction analysis** – correlation of SHAP values to detect synergies between features (e.g., liquidity × profitability).
- **Reliability index** – combines prediction error with stability of feature contributions over time.
- **Two experimental setups**:
  1. *Synthetic data* – controlled non‑linear relationships (tanh, quadratic, pairwise interaction) to validate the method.
  2. *Real‑world data* – Corporate Credit Rating dataset (OpenML ID 46525) with 2029 companies, 41 financial features, and 10 rating classes.

## Repository structure
