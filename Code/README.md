# Code Directory

The following is a description of the Jupyter Notebooks used for our volatility prediction project: 

## Files

## 1. Exploratory Data Analysis (EDA)
* **`EDA (Diogo).ipynb`** - **[Primary]** This is the main EDA notebook. The findings and visualizations generated here are the ones referenced in the main body of our paper.
* **`EDA_ETF_Volatility_TimeSeries_vb_done_021126_VB_git_021026.ipynb`** - Supplementary EDA code. The outputs from this notebook are included in the appendix of the Final Report.

## 2. Data Preparation & Splits
* **`data_split_for_volatility_dataset_vb_done_git_040426_040426.ipynb`** - Contains the logic and code execution for setting up our training, validation, and test sets.

## 3. Baseline Models
* **`xgboost_walk_forward_vol_baseline_vb_git_030326_1551.ipynb`** - Walk-forward baseline model utilizing XGBoost.
* **`baseline_model_TF_vb_done_041626_1658.ipynb`** - Contains our other standard baseline models, including naive averages and Ordinary Least Squares (OLS) regressions.

## 4. Advanced Models
* **`elastic_net_vol_v1v2_TF_vb_done_041826.ipynb` (Vinit)** - Implementation and hyperparameter tuning for our Elastic Net model.
* **`LSTM_basic (3).ipynb` (Sid)** - Implementation of the Long Short-Term Memory (LSTM) neural network.
* **`Transformer_1.ipynb` (Diogo)** - This is our more performant Transformer model, that we included the results of the paper.
* **`train_transformer.ipynb` (Ronald)** - An alternative Transformer model. While its performance did not exceed the model above, it represents independent modeling and architectural work.

## Running

Make sure you've updated `DATA_PATH` in `baseline_model.py` to point to your local copy of the volatility dataset before running. The path in the script is hardcoded to the original dev environment.

```bash
python3 baseline_model.py
```

For the full comparison:

```bash
python3 baseline_model_vs_model_comparison.py
```

Training takes a few minutes on CPU. LSTM and Transformer are the slowest. Results get written to `model_comparison_results.csv`.

## Notes

The baseline script only runs on SPY — that was a deliberate simplification to get a quick sanity check working before scaling to all tickers. The comparison script also operates on SPY only for the same reason; the full multi-ticker pipeline lives in the Transformer_2 and Elastic_Net_Regression directories.

Time splits used throughout: train 2015–2021, validation 2022–2023, test 2024–2025. Scaler is always fit on train only.
