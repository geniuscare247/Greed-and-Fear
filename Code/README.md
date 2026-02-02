# Baseline Model Implementation

This directory contains the implementation of the Baseline Model and Model 1 (Linear Regression) as described in the Research Proposal.

## Files
- `baseline_model.py`: The main Python script implementing the models in TensorFlow.

## Requirements
To run this code, you need the following Python packages:
- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`

## Usage
Run the script using Python:

```bash
python3 baseline_model.py
```

## Description
The script performs the following steps:
1.  **Loads Data**: Reads the volatility dataset from `../data/volatility_dataset_013026.csv`.
2.  **Preprocesses**: Filters for 'SPY' (as a demonstration), calculates daily log returns, and computes target volatility (next 5 days) and features (trailing 20d/5d volatility).
3.  **Splits Data**: Uses time-based splitting:
    - Train: 2015-2021
    - Validation: 2022-2023
    - Test: 2024-2025
4.  **Trains Model**: Trains a Linear Regression model using TensorFlow (`Model 1`).
5.  **Evaluates**:
    - Calculates MAE/RMSE for the **Persistence Heuristic** (Baseline).
    - Calculates MAE/RMSE for the **Linear Regression Model** (Model 1).
6.  **Saves**: Saves the trained TensorFlow model to `baseline_model_tf.keras`.
