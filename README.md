# Forecasting Next-Week Realized Volatility for S&P 500 ETFs

**Team:** Greed and Fear  
**Members:** Vinit S. Bhatt, 
             Siddharthen Sridhar, 
             Ronald Ho, 
             Diogo Viveiros

## 📌 Project Overview
Volatility is a core object in investing, affecting position sizing, risk budgeting, and derivative pricing. Unlike short-horizon returns, volatility exhibits persistence and clustering, making it a realistic target for predictive modeling.

This project builds an end-to-end machine learning pipeline to predict **next-week realized volatility** for a panel of S&P 500 index-linked ETFs (market, sector, and industry). We use historical price data and macro/risk indicators from FactSet (2014-2025) to compare strong heuristic baselines against TensorFlow-based models.

## 🎯 Prediction Task
**Goal:** Predict the annualized realized volatility for the next 5 trading days.

$$ y_{i,t} = \sqrt{\sum_{j=1}^{5} r_{i,t+j}^2} \times \sqrt{252} $$

**Input Features:** computed using only data available up to time $t$ (no look-ahead bias).

## 📂 Repository Structure

```
Greed-and-Fear/
├── Code/               # Modeling and source code
│   ├── baseline_model.py   # TensorFlow implementation of Baseline & Linear Model
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
*   `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`

### Installation
Clone the repo and install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Running the EDA
Generate visualizations for price history, volatility comparisons, and correlations:

```bash
python3 EDA/eda_script.py
```
Output figures will be saved in the `EDA/` directory.

### Running the Baseline Model
Train the Linear Regression model (TensorFlow) and evaluate against the persistence baseline:

```bash
python3 Code/baseline_model.py
```

## 📊 Models & Evaluation

We evaluate performance using **MAE** and **RMSE** on a time-based split:
*   **Train:** 2015 – 2021
*   **Validation:** 2022 – 2023
*   **Test:** 2024 – 2025

**Planned Models:**
1.  **Baseline:** Volatility Persistence (predict next week = last 20 days).
2.  **Model 1:** Linear Regression (TensorFlow).
3.  **Model 2:** Feed-Forward Neural Network (MLP) to capture nonlinear interactions.

## 📅 Roadmap
*   [x] **Proposal Submission**
*   [x] **Data Collection & Cleaning**
*   [x] **EDA & Feature Engineering**
*   [x] **Baseline Model Implementation**
*   [ ] **Advanced Model (MLP) Development**
*   [ ] **Subgroup Evaluation (Sector/Industry)**
*   [ ] **Final Report & Presentation**

## 📚 References
*   Campisi, G., et al. (2024). *A comparison of machine learning methods...*
*   Díaz, J. D., et al. (2024). *Machine-learning stock market volatility...*
*   Filipović, D., & Khalilzadeh, A. (2021). *Machine learning for predicting stock return volatility.*
