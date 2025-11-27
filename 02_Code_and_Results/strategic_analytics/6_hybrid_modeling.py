import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"
OUTPUT_MODEL = "strategic_analytics/models/hybrid_stack.pkl"
REPORT_DIR = "strategic_analytics/reports"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)

def run_hybrid_modeling():
    print("--- Module 6: Hybrid Modeling (Stacking Ensemble) ---")
    
    # 1. Load Data
    try:
        df = pd.read_parquet(INPUT_PATH)
        with open(FEATURE_PATH, 'r') as f:
            features_json = json.load(f)
            selected_features = features_json['confirmed_features']
            if 'Persona_Cluster' not in selected_features and 'Persona_Cluster' in df.columns:
                selected_features.append('Persona_Cluster')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X = df[selected_features]
    y = df['GradeClass'].round().astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Define Base Learners
    print("Initializing Base Learners...")
    
    # Learner 1: CatBoost (Gradient Boosting)
    # Using parameters close to what we found in Module 4 (or defaults for stability)
    cb_clf = CatBoostClassifier(
        iterations=500, 
        depth=6, 
        learning_rate=0.05, 
        loss_function='MultiClass',
        verbose=False,
        random_seed=42
    )
    
    # Learner 2: Random Forest (Bagging - Proxy for TabPFN/Robust Tabular)
    # RF is excellent for tabular data and adds diversity to Boosting
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    
    # Learner 3: MLP (Deep Learning - Proxy for TabM/Deep Ensemble)
    # Simple Neural Net to capture different patterns
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    
    estimators = [
        ('catboost', cb_clf),
        ('rf', rf_clf),
        ('mlp', mlp_clf)
    ]
    
    # 3. Meta Learner
    # Logistic Regression to weigh the probabilities from base learners
    meta_learner = LogisticRegression(max_iter=1000)
    
    # 4. Stacking
    print("Training Stacking Classifier (Level 2)...")
    # stack_method='predict_proba' uses class probabilities as input to meta-learner
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        stack_method='predict_proba',
        cv=3, # Internal CV for generating meta-features
        n_jobs=-1
    )
    
    stacking_clf.fit(X_train, y_train)
    
    # 5. Evaluation
    print("Evaluating Hybrid Model...")
    y_pred = stacking_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    
    print(f"Hybrid Model Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"QWK: {qwk:.4f}")
    
    # Compare with single CatBoost (from Module 4, roughly)
    cb_clf.fit(X_train, y_train)
    cb_pred = cb_clf.predict(X_test)
    cb_qwk = cohen_kappa_score(y_test, cb_pred, weights='quadratic')
    print(f"Single CatBoost QWK: {cb_qwk:.4f}")
    
    if qwk > cb_qwk:
        print(">> Success: Hybrid Stacking improved performance!")
    else:
        print(">> Note: Hybrid performance is similar or slightly lower (common with small data/strong base).")
        
    # Save Model
    joblib.dump(stacking_clf, OUTPUT_MODEL)
    print(f"Saved Hybrid Model to {OUTPUT_MODEL}")
    
    print("Module 6 Complete.")

if __name__ == "__main__":
    run_hybrid_modeling()
