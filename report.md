📝 Credit Risk Prediction & SHAP Interpretability Report
📌 Project Overview
This project builds a complete machine learning pipeline to predict credit risk using synthetic data and explain model decisions using SHAP. It includes preprocessing, model training, evaluation, interpretability, and stakeholder-ready exports.

📊 Dataset Summary
- Source: Synthetic dataset (credit_risk_dataset.csv)
- Size: 500 rows, 8 features + 1 binary target (Risk)
- Features:
- Age, Income, LoanAmount, CreditScore, EmploymentStatus, MaritalStatus, EducationLevel, ResidenceType

⚙️ Pipeline Steps
- Preprocessing
- Imputation: SimpleImputer
- Scaling: StandardScaler
- Encoding: OneHotEncoder(sparse_output=False)
- Pipeline: ColumnTransformer + Pipeline
- Model Training
- Algorithms: LightGBM, XGBoost, RandomForest
- Tuning: GridSearchCV
- Best Model: bestmodel.pkl (LightGBM)
- Evaluation
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrix: TP, FP, FN, TN
- Classification Report: classification_report.txt
- Metrics JSON: metrics_report.json

🔍 SHAP Interpretability
Global Explanations
- Summary Plot: shap_summary_plot.png
- Bar Plot: Feature importance
- Mean SHAP CSV: global_shap_importance.csv
Local Explanations
- Force Plots: TP/FP/FN examples (shap_forceplot_*.html)
- Waterfall Plots: shap_waterfall_*.png
- JSON Contributions: local_shap_explanations.json

📦 Exported Artifacts
- bestmodel.pkl: Trained LightGBM model
- global_shap_importance.csv: Ranked feature importance
- local_shap_explanations.json: SHAP values per instance
- textual_shap_report.txt: Plain-language summary
- underwriting_recommendations.txt: Actionable insights
- deliverables_summary.json: Artifact manifest

📣 Stakeholder Insights
- Top Risk Drivers: Low CreditScore, High LoanAmount, Unstable Employment
- Underwriting Suggestions:
- Flag applicants with CreditScore < 600
- Prioritize stable employment and verified income
- Model Confidence: ROC-AUC > 0.85, F1 > 0.80


