import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import os
import json

# --- Configuration ---
RATIOS = [0.1, 0.2, 0.3, 0.4]
RESULTS_FILE = "baseline_implementation/ratio_results.json"
CM_PLOT_FILE = "baseline_implementation/baseline_confusion_matrix.png"

# Tuned Params from previous step
BEST_PARAMS = {
    "n_estimators": 100,
    "min_samples_leaf": 4,
    "min_samples_split": 10,
    "max_depth": None,
    "random_state": 42
}

def run_experiment():
    print("--- Running Baseline Ratio Experiment & Confusion Matrix ---")
    
    try:
        df = pd.read_csv("data/raw/validated_students.csv")
    except:
        df = pd.read_csv("combined_students_final.csv")

    X = df.drop(columns=['GradeClass', 'GPA'])
    y = df['GradeClass']
    
    # Label Encode Categoricals
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    results = []
    
    # 1. Ratio Experiment
    for test_size in RATIOS:
        print(f"Testing split: {test_size}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train_processed = imputer.fit_transform(X_train)
        X_test_processed = imputer.transform(X_test)
        
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        
        rf = RandomForestClassifier(**BEST_PARAMS)
        rf.fit(X_train_scaled, y_train)
        
        y_pred = rf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
        
        results.append({
            "test_size": test_size,
            "train_size": 1.0 - test_size,
            "qwk": qwk,
            "accuracy": acc
        })
        
        # 2. Generate Confusion Matrix (only for the standard 0.2 split)
        if test_size == 0.2:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Baseline Confusion Matrix (Random Forest)')
            plt.savefig(CM_PLOT_FILE)
            plt.close()
            print(f"Confusion Matrix saved to {CM_PLOT_FILE}")

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_experiment()
