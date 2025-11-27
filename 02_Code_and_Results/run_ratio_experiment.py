import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, PowerTransformer
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
import joblib
import os
import json

# --- Configuration ---
RATIOS = [0.1, 0.2, 0.3, 0.4] # Test sizes
RESULTS_FILE = "reports/ratio_experiment_results.json"
DATA_PATH = "data/raw/validated_students.csv"

# --- Helper Functions (Adapted from Pipeline Scripts) ---

def feature_engineering(df):
    df = df.copy()
    df['TechSynergy'] = df['InternetAccess'] * df['StudyTimeWeekly']
    df['SelfDriven'] = df['StudyTimeWeekly'] / (df['ParentalEducation'] + 1)
    df['SupportIndex'] = df['ParentalEducation'] + df['Tutoring'] + df['InternetAccess']
    df['BalancedLife'] = df['Extracurricular'] * df['StudyTimeWeekly']
    df['AbsenceRisk'] = df['Absences'] * (1 - df['InternetAccess'])
    return df

def get_preprocessor():
    numeric_features = ['StudyTimeWeekly', 'Absences', 'TechSynergy', 'SelfDriven', 'SupportIndex', 'BalancedLife', 'AbsenceRisk']
    ordinal_features = ['ParentalEducation']
    nominal_features = ['Gender', 'Tutoring', 'Extracurricular', 'InternetAccess']

    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=0)),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ])
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', CatBoostEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('nom', nominal_transformer, nominal_features)
        ],
        remainder='drop'
    )
    return preprocessor

def run_experiment_for_ratio(test_size):
    print(f"\n--- Running Experiment for Test Size: {test_size} ({int((1-test_size)*100)}/{int(test_size*100)} Split) ---")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['GradeClass', 'GPA'])
    y = df['GradeClass']
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 3. Feature Engineering
    X_train_eng = feature_engineering(X_train)
    X_test_eng = feature_engineering(X_test)
    
    # 4. Preprocessing
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_eng, y_train)
    X_test_processed = preprocessor.transform(X_test_eng)
    
    # 5. Modeling (Using Best Params from previous runs)
    # Best params from fine-tuning or initial run. Using robust defaults + optimization insights.
    params = {
        "iterations": 1000,
        "learning_rate": 0.03, # Typical stable rate
        "depth": 6,
        "loss_function": "MultiClass",
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": 42
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train_processed, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test_processed).flatten()
    
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Results for Test Size {test_size}: QWK={qwk:.4f}, Acc={acc:.4f}")
    
    # Save processed data for this ratio (as requested)
    output_dir = f"data/processed_ratio_{int(test_size*100)}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as parquet for consistency if needed, but we just need results for report mostly.
    # Saving just to satisfy "create a new processed folder" requirement.
    pd.DataFrame(X_train_processed).to_parquet(f"{output_dir}/train.parquet")
    pd.DataFrame(X_test_processed).to_parquet(f"{output_dir}/test.parquet")
    
    return {
        "test_size": test_size,
        "train_size": 1.0 - test_size,
        "qwk": qwk,
        "accuracy": acc,
        "f1": f1
    }

def main():
    results = []
    for ratio in RATIOS:
        res = run_experiment_for_ratio(ratio)
        results.append(res)
        
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n=== Experiment Complete ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
