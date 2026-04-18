# Code Directory

Two main scripts here — one for the single linear baseline, one that runs all five models back to back for comparison.

## Files

- `baseline_model.py` — trains the TF linear regression (Model 1) on SPY only, saves the `.keras` file and scatter plot
- `baseline_model_vs_model_comparison.py` — runs Baseline FCN, Random Forest, XGBoost, LSTM, and Transformer, prints a results table, and saves comparison plots

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
