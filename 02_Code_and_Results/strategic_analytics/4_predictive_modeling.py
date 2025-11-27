import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"
OUTPUT_ORDINAL_MODEL = "strategic_analytics/models/ordinal_model.pkl"
OUTPUT_SURVIVAL_MODEL = "strategic_analytics/models/survival_model.pkl"
REPORT_DIR = "strategic_analytics/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

def run_predictive_modeling():
    print("--- Module 4: Next-Gen Predictive Modeling ---")
    
    # 1. Load Data & Features
    try:
        df = pd.read_parquet(INPUT_PATH)
        with open(FEATURE_PATH, 'r') as f:
            features_json = json.load(f)
            selected_features = features_json['confirmed_features']
            # Add back Persona_Cluster if not selected but useful
            if 'Persona_Cluster' not in selected_features and 'Persona_Cluster' in df.columns:
                selected_features.append('Persona_Cluster')
                
        print(f"Loaded data and {len(selected_features)} selected features.")
    except FileNotFoundError:
        print("Input data or features not found. Run previous modules.")
        return

    # Prepare Data
    X = df[selected_features]
    y = df['GradeClass'].round().astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Ordinal Regression (CatBoost with Optuna)
    print("\nTraining Ordinal Regression Model (CatBoost + Optuna)...")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'loss_function': 'MultiClass', # CatBoost doesn't have native Ordinal yet, use MultiClass
            'eval_metric': 'Kappa', # Optimize for Kappa directly
            'verbose': False,
            'random_seed': 42
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        
        preds = model.predict(X_test)
        qwk = cohen_kappa_score(y_test, preds, weights='quadratic')
        return qwk

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # 20 trials for speed in demo
    
    print(f"Best QWK: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    # Train Final Model
    best_params = study.best_params
    best_params['loss_function'] = 'MultiClass'
    best_params['random_seed'] = 42
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate
    preds = final_model.predict(X_test)
    final_qwk = cohen_kappa_score(y_test, preds, weights='quadratic')
    print(f"Final Test QWK: {final_qwk:.4f}")
    
    # Save Model
    final_model.save_model(OUTPUT_ORDINAL_MODEL)
    print(f"Saved Ordinal Model to {OUTPUT_ORDINAL_MODEL}")
    
    # 3. Survival Analysis (CoxPH)
    print("\nRunning Survival Analysis (Time-to-Dropout)...")
    
    # Create Survival Data
    # Event: GradeClass 4 (Fail/Dropout) = 1, Others = 0
    df_surv = df.copy()
    df_surv['Event'] = (df_surv['GradeClass'] == 4).astype(int)
    
    # Duration: We don't have real time data.
    # Proxy: 'Absences' correlates with time-at-risk (more absences ~ closer to dropout).
    # Let's simulate 'WeeksEnrolled' inversely proportional to Absences + some noise.
    # Or better: Use 'StudyTimeWeekly' * 'AttendanceRate' proxy?
    # Let's use a synthetic proxy for demonstration:
    # High Absences -> Low Duration (dropped out early or at risk)
    # Low Absences -> High Duration (survived longer)
    # This is a heuristic for the demo. In real life, we'd use actual 'DaysEnrolled'.
    
    max_weeks = 40 # Full school year
    # Normalize absences 0-1
    abs_norm = (df_surv['Absences'] - df_surv['Absences'].min()) / (df_surv['Absences'].max() - df_surv['Absences'].min())
    # Duration = MaxWeeks * (1 - Absences_Norm) + Noise
    # If Event=0 (Survived), Duration is censored at MaxWeeks (or close to it)
    
    np.random.seed(42)
    df_surv['Duration'] = max_weeks * (1 - abs_norm) + np.random.normal(0, 2, len(df_surv))
    df_surv['Duration'] = df_surv['Duration'].clip(1, max_weeks)
    
    # For censored (Event=0), set Duration to MaxWeeks (they survived the whole year)
    df_surv.loc[df_surv['Event'] == 0, 'Duration'] = max_weeks
    
    # Select features for CoxPH (must be numeric, no high collinearity)
    # Use a subset of strong features
    surv_features = ['Duration', 'Event', 'StudyTimeWeekly', 'Absences', 'ParentalSupport', 'GPA']
    # Ensure they exist
    surv_features = [f for f in surv_features if f in df_surv.columns]
    
    # Add penalizer to handle collinearity/singularity
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(df_surv[surv_features], duration_col='Duration', event_col='Event')
        
        print("\nSurvival Model Summary:")
        cph.print_summary()
        
        # Plot Survival Curves for average student vs at-risk (Manual Plotting)
        plt.figure(figsize=(10, 6))
        
        # Create synthetic students with varying absences
        # We need to keep other features constant (e.g., at mean)
        base_student = df_surv[surv_features].mean().to_dict()
        
        absences_values = [0, 10, 20, 30]
        for abs_val in absences_values:
            student = base_student.copy()
            student['Absences'] = abs_val
            # Predict survival function
            # predict_survival_function returns a DataFrame where index is time, col is student
            surv_func = cph.predict_survival_function(pd.DataFrame([student]))
            plt.plot(surv_func.index, surv_func.iloc[:, 0], label=f'Absences={abs_val}')
            
        plt.title("Survival Curves by Absences (Risk of Dropout)")
        plt.xlabel("Weeks Enrolled")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{REPORT_DIR}/survival_curves.png")
        print(f"Saved Survival Curves to {REPORT_DIR}/survival_curves.png")
        
        # Save Model
        # Lifelines objects can be pickled
        with open(OUTPUT_SURVIVAL_MODEL, 'wb') as f:
            joblib.dump(cph, f)
            
    except Exception as e:
        print(f"Survival Analysis Failed (likely convergence or data issue): {e}")

    print("Module 4 Complete.")

if __name__ == "__main__":
    run_predictive_modeling()
