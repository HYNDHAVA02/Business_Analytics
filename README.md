# Strategic Educational Data Mining

## Overview
This project implements a comprehensive data analytics and machine learning pipeline designed to analyze student performance data. It leverages advanced techniques in data engineering, causal discovery, and predictive modeling to uncover insights and predict student outcomes.

## Project Structure
The project is organized as follows:

- **`01_Project_Report.pdf`**: The full project report containing detailed analysis and findings.
- **`02_Code_and_Results/`**: Contains all source code, scripts, and generated results.
    - **Data Pipeline**: Scripts for ingestion (`1_ingest_validate.py`), processing (`2_process_engineer.py`), and EDA (`3_eda_hypothesis.py`).
    - **Modeling**: Predictive modeling scripts (`4_modeling_tuning.py`) and evaluation (`5_evaluation_audit.py`).
    - **Dashboard**: Interactive Streamlit application (`6_app.py`).
    - **Docs**: `project_glossary.md` and other documentation.
- **`03_Dataset_Details/`**: Contains the raw and processed datasets.
- **`04_Paper_Overleaf_Link.txt`**: Link to the Overleaf project for the paper.
- **`05_GitHub_Link.txt`**: Link to the GitHub repository.

## Key Features & Technologies

### Data Engineering
- **MICE (Multivariate Imputation by Chained Equations)**: For intelligent missing value imputation.
- **Yeo-Johnson Transformation**: To normalize skewed data distributions.
- **RobustScaler**: For scaling data while handling outliers.

### Advanced Analytics
- **PC Algorithm**: For causal discovery to understand relationships between variables.
- **UMAP**: For dimensionality reduction and identifying learner personas.
- **Boruta**: For robust feature selection.

### Predictive Modeling
- **CatBoost**: High-performance gradient boosting on decision trees.
- **Random Forest**: Baseline model for comparison.
- **Hybrid Stacking**: Ensemble method combining multiple models for improved accuracy.
- **CoxPH**: Survival analysis to predict time-to-event outcomes.

### Evaluation
- **QWK (Quadratic Weighted Kappa)**: Metric for evaluating ordinal classification.
- **SHAP (SHapley Additive exPlanations)**: For model interpretability.
- **Optuna**: For hyperparameter optimization.

## Usage

### Prerequisites
Ensure you have Python installed. You may need to install dependencies:
```bash
pip install -r requirements.txt
```
*(Note: If `requirements.txt` is missing, please install necessary packages like `pandas`, `numpy`, `scikit-learn`, `catboost`, `shap`, `streamlit`, `optuna` manually.)*

### Running the Pipeline
To execute the entire data processing and modeling pipeline, run:
```bash
python 02_Code_and_Results/run_pipeline.py
```
This script sequentially executes the ingestion, processing, EDA, modeling, and evaluation steps.

### Launching the Dashboard
To interact with the project results via the dashboard, run:
```bash
streamlit run 02_Code_and_Results/6_app.py
```

## Links
- **GitHub Repository**: [Link](https://github.com/HYNDHAVA02/Business_Analytics.git)
- **Overleaf Paper**: [See 04_Paper_Overleaf_Link.txt]

## License
This project is for educational purposes.