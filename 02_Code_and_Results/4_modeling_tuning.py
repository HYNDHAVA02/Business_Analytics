import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import joblib
import os

def objective(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": False,
        "allow_writing_files": False,
        "loss_function": "MultiClass" 
    }
    
    # Cross-Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    qwk_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=50, verbose=False)
        
        preds = model.predict(X_val_fold)
        qwk = cohen_kappa_score(y_val_fold, preds, weights='quadratic')
        qwk_scores.append(qwk)
        
    return np.mean(qwk_scores)

def main():
    print("Loading processed training data...")
    df = pd.read_parquet("data/processed/train.parquet")
    X = df.drop(columns=['GradeClass', 'GPA'])
    y = df['GradeClass']
    
    # Ensure y is integer
    y = y.astype(int)
    
    print("Starting Optuna Optimization...")
    mlflow.set_experiment("Student_Performance_CatBoost")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50) # Increased to 50 for fine tuning
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Train final model with best params
    print("Training final model...")
    best_params = trial.params
    best_params["loss_function"] = "MultiClass"
    best_params["allow_writing_files"] = False
    
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y, verbose=100)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    final_model.save_model("models/best_catboost_model.cbm")
    print("Model saved to models/best_catboost_model.cbm")
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_qwk", trial.value)
        mlflow.catboost.log_model(final_model, "model")
        print("Logged to MLflow.")

if __name__ == "__main__":
    main()
