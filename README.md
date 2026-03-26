# S&P 500 Stock Price Prediction Project

This folder contains the complete development lifecycle and results of the stock price prediction system for the S&P 500 (^GSPC).

## Project Overview

The goal was to build a machine learning system to predict the next day's closing price using historical data from 2021-2025, with a strict chronological split:
Training: 2021-01-01 to 2024-12-31
Testing: 2025-01-01 to 2025-12-31

### Development Phases

#### Phase 1: Baseline Implementation
Goal: Establish a working pipeline using Random Forest and XGBoost.
Approach: Predicted raw closing prices using 1, 2, and 3-day lags.
Result: Functional but high MSE (Mean Squared Error), due to the non-stationarity of the price series.

#### Phase 2: Model Optimization (Drastic Improvement)
Goal: Reduce MSE and improve predictive power.
Key Breakthrough: Switched to predicting Log Returns (stationary) instead of raw prices.
Feature Engineering: Expanded to 10-day lags, rolling statistics (mean, std), and momentum indicators.
Result: Reduced MSE by 98%, with XGBoost emerging as the slightly better performer.

## Folder Structure

predict_stock.py: The final, optimized Python script.
implementation_plan.md: Technical roadmap for the project (Markdown).
implementation_plan.pdf: Technical roadmap for the project (PDF).
analysis_results.md: Detailed analysis of the performance gap and the optimization strategy.
walkthrough.md: Final summary of results and evaluation metrics.
prediction_results_improved.png: Visualization of actual vs predicted prices for the 2025 test period.
task.md: Tracking of the development steps completed.

## How to Run
1. Ensure dependencies are installed: pip install yfinance pandas numpy scikit-learn xgboost matplotlib
2. Run the script: python predict_stock.py
3. View the generated plot in prediction_results_improved.png.
     
      4. ---
      5. *Developed by Antigravity AI*
      6. ---
      7. 
