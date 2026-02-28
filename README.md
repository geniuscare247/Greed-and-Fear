# Forecasting Next-Week Realized Volatility for S&P 500 ETFs

**Team:** Greed and Fear  
**Members:** Vinit S. Bhatt, 
             Siddharthen Sridhar, 
             Ronald Ho, 
             Diogo Viveiros

## 📌 Project Overview
Volatility is a core object in investing, affecting position sizing, risk budgeting, and derivative pricing. Unlike short-horizon returns, volatility exhibits persistence and clustering, making it a realistic target for predictive modeling.

This project builds an end-to-end machine learning pipeline to predict **next-week realized volatility** for a panel of S&P 500 index-linked ETFs (market, sector, and industry). We use historical price data and macro/risk indicators from FactSet (2014-2025) to compare strong heuristic baselines against a suite of machine learning and deep learning models.

## 🎯 Prediction Task
**Goal:** Predict the annualized realized volatility for the next 5 trading days.

$$ y_{i,t} = \sqrt{\sum_{j=1}^{5} r_{i,t+j}^2} \times \sqrt{252} $$

**Input Features:** computed using only data available up to time $t$ (no look-ahead bias).

## 📂 Repository Structure

```
Greed-and-Fear/
├── Code/               # Modeling and source code
│   ├── baseline_model.py   # TensorFlow implementation of Baseline & Linear Model
│   ├── baseline_model_vs_model_comparison.py # Comprehensive model comparison script
│   └── README.md           # Instructions for running models
├── data/               # Dataset files
│   ├── volatility_dataset_013026.csv
│   └── volatility_dataset_description_UPDATED_013026.pdf
├── EDA/                # Exploratory Data Analysis
│   ├── eda_script.py       # Script to generate plots
│   └── [figures]           # Generated PNG visualizations
├── research_proposal/  # Documentation
│   └── Final_Project_Proposal_Submission_2026-02-01.pdf
└── README.md           # This file
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `xgboost`

### Installation
Clone the repo and install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
```

### Running the EDA
Generate visualizations for price history, volatility comparisons, and correlations:

```bash
python3 EDA/eda_script.py
```
Output figures will be saved in the `EDA/` directory.

### Running the Model Comparison
Train and evaluate all models (Baseline FCN, Random Forest, XGBoost, LSTM, and Transformer):

```bash
python3 Code/baseline_model_vs_model_comparison.py
```
This script will train all models, print the evaluation results to the console, and save comparison plots and a CSV of the results in the `Code/` directory.

## 📊 Models & Evaluation

We evaluate performance using **MAE** and **RMSE** on a time-based split:
*   **Train:** 2015 – 2021
*   **Validation:** 2022 – 2023
*   **Test:** 2024 – 2025

### Implemented Models

1.  **Baseline: Fully Connected Network (FCN)**: A simple linear regression model implemented in TensorFlow.
    ```
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 1)                 3         
    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________
    ```

2.  **Random Forest**: An ensemble of decision trees to capture non-linear relationships.

3.  **XGBoost**: A gradient boosting framework that has proven to be highly effective in many machine learning competitions.

4.  **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) well-suited for time-series data.
    ```
    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 5, 50)             10600     
     lstm_1 (LSTM)               (None, 50)                20200     
     dense_1 (Dense)             (None, 1)                 51        
    =================================================================
    Total params: 30,851
    Trainable params: 30,851
    Non-trainable params: 0
    _________________________________________________________________
    ```

5.  **Transformer**: A deep learning model that uses self-attention, originally designed for natural language processing, but also effective for time-series forecasting.
    ```
    Model: "functional_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_layer_2 (InputLayer)     [(None, 5, 2)]       0           []                               
     multi_head_attention (MultiHea  (None, 5, 2)        11266       [...]
     ...
     global_average_pooling1d (Glob  (None, 5)           0           [...]
     dense_2 (Dense)                (None, 64)           384         [...]
     dense_3 (Dense)                (None, 1)            65          [...]
    ==================================================================================================
    Total params: 24,281
    Trainable params: 24,281
    Non-trainable params: 0
    __________________________________________________________________________________________________
    ```

### Performance

The following table summarizes the performance of each model on the test set:

| Model             | MAE      | RMSE     |
|-------------------|----------|----------|
| Baseline FCN      | 0.101279 | 0.172551 |
| Random Forest     | 0.123252 | 0.262638 |
| XGBoost           | 0.130733 | 0.265453 |
| LSTM              | 0.130582 | 0.290462 |
| Transformer       | 0.119788 | 0.220441 |

![Model Evaluation](Code/model_performance_comparison.png)

## 📅 Roadmap
*   [x] **Proposal Submission**
*   [x] **Data Collection & Cleaning**
*   [x] **EDA & Feature Engineering**
*   [x] **Baseline Model Implementation**
*   [x] **Advanced Model Development & Comparison**
*   [ ] **Subgroup Evaluation (Sector/Industry)**
*   [ ] **Final Report & Presentation**

## 📚 References
*   Campisi, G., et al. (2024). *A comparison of machine learning methods...*
*   Díaz, J. D., et al. (2024). *Machine-learning stock market volatility...*
*   Filipović, D., & Khalilzadeh, A. (2021). *Machine learning for predicting stock return volatility.*
