Credit Risk Prediction with SHAP Explainability

This project builds a complete end-to-end Credit Risk Prediction system with Explainable AI (XAI) using SHAP.
It covers data preprocessing, model training, hyperparameter tuning, SHAP-based global and local explanations, and exporting interactive visualizations.

## 📌 Project Overview

Financial institutions require transparent and explainable ML models for lending decisions.
This project demonstrates:

Building a credit-risk prediction model

Preprocessing numerical + categorical variables

Hyperparameter optimization

Visualizing feature importance using SHAP (global + local)

Generating interactive SHAP force plots, waterfall plots, and summary plots

Exporting each explanation for documentation

## 📁 Project Structure
├── credit_risk_dataset.csv           # Dataset (500 rows)
├── credit_risk_shap_project.ipynb    # Main notebook with SHAP visualizations
├── shap_outputs/                     # Auto-generated SHAP HTML & images
│   ├── shap_summary_plot.png
│   ├── shap_forceplot_0.html
│   ├── shap_waterfall_0.png
│   └── ...
└── README.md

## 🧠 Machine Learning Workflow
1. Data Loading

Loads credit_risk_dataset.csv

500 records with 9 columns (8 features + 1 target)

2. Preprocessing

Missing values handled using SimpleImputer

Numerical scaling using StandardScaler

Categorical encoding using

OneHotEncoder(handle_unknown="ignore", sparse_output=False)

3. Train–Test Split

80% training

20% testing

4. Model

Gradient Boosting (RandomForest/LightGBM/XGBoost optional)

Hyperparameter tuning via RandomizedSearchCV

5. Evaluation Metrics

Accuracy

Confusion Matrix

Classification Report

ROC–AUC Score

## 🔍 Explainable AI Using SHAP
Global Explanations

SHAP Summary Plot (beeswarm)

Feature Importance Rankings

Mean Absolute SHAP Values

Local Explanations

Force Plot (per customer)

Waterfall Plot (detailed breakdown)

Decision Plot (optional)

Each explanation is saved to:

/shap_outputs/


using:

shap.save_html("shap_forceplot_0.html", force_plot_object)

## 📊 Interactive Visualizations

The notebook includes:

Interactive force plots (JavaScript/HTML)

Interactive decision plots

Popup HTML renders inside Jupyter

All saved artifacts can be shared with:

Managers

Auditors

Lenders

Model governance teams

## 🧪 Key Technologies
Category	Tools
Language	Python 3.x
ML	scikit-learn
Explainability	SHAP
Visualization	Matplotlib, SHAP JS
Notebook	Jupyter
## 🚀 How to Run the Project
1. Install dependencies
pip install numpy pandas scikit-learn shap matplotlib

2. Open the notebook
jupyter notebook credit_risk_shap_project.ipynb

3. Run all cells

This will:

✔ Train the ML model
✔ Generate SHAP values
✔ Save all plots automatically
✔ Produce interactive HTML explanations

## 📈 Results Summary

SHAP clearly identifies credit_score, loan_amount, and income as the strongest predictors.

Local explanations highlight why a specific customer is predicted to default.

The model achieves strong evaluation metrics (varies per run).

