import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, make_scorer
import os
import json

# --- Baseline Configuration ---
# Represents a "Traditional" approach with Tuning:
# - Mean Imputation
# - Standard Scaling
# - Random Forest (Tuned via GridSearchCV)

def run_baseline():
    print("--- Running Baseline Model (Traditional Approach with Tuning) ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv("data/raw/validated_students.csv")
    except:
        df = pd.read_csv("combined_students_final.csv")

    # 2. Basic Preprocessing
    X = df.drop(columns=['GradeClass', 'GPA'])
    y = df['GradeClass']
    
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Imputation & Scaling
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # 5. Modeling (Random Forest with GridSearchCV)
    print("Starting Hyperparameter Tuning (GridSearchCV)...")
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Define QWK scorer
    qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=qwk_scorer,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    
    # 6. Evaluation
    y_pred = best_rf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Tuned Baseline Results: Accuracy={acc:.4f}, QWK={qwk:.4f}, F1={f1:.4f}")
    
    # Save results
    results = {
        "model": "Random Forest (Tuned)",
        "imputation": "Mean",
        "scaling": "StandardScaler",
        "best_params": best_params,
        "accuracy": acc,
        "qwk": qwk,
        "f1": f1
    }
    
    os.makedirs("baseline_implementation", exist_ok=True)
    with open("baseline_implementation/results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_baseline()
