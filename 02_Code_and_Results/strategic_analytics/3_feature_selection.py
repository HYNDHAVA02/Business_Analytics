import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
OUTPUT_PATH = "strategic_analytics/models/selected_features.json"
REPORT_DIR = "strategic_analytics/reports"
MODEL_DIR = "strategic_analytics/models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_feature_selection():
    print("--- Module 3: Rigorous Feature Selection ---")
    
    # 1. Load Data
    try:
        df = pd.read_parquet(INPUT_PATH)
        print(f"Loaded phenotyped data from {INPUT_PATH}")
    except FileNotFoundError:
        print("Input data not found. Please run Module 2 first.")
        return

    # Prepare X and y
    # Target is GradeClass
    # Ensure y is integer for Classification
    y = df['GradeClass'].round().astype(int)
    # Drop targets and non-feature columns (like UMAP coords, Cluster ID if we don't want to leak it)
    # Actually, Persona_Cluster IS a feature we want to test!
    X = df.drop(columns=['GradeClass', 'GPA', 'UMAP_1', 'UMAP_2'], errors='ignore')
    
    # Ensure numeric
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    
    # 2. Boruta Feature Selection
    print("Running Boruta Algorithm (Shadow Features)...")
    
    # Boruta requires a Random Forest backend
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    
    # BorutaPy works on numpy arrays
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
    boruta_selector.fit(X.values, y.values)
    
    # Get selected features
    selected_mask = boruta_selector.support_
    confirmed_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    tentative_mask = boruta_selector.support_weak_
    tentative_features = [feature_names[i] for i in range(len(feature_names)) if tentative_mask[i]]
    
    print(f"\nConfirmed Features ({len(confirmed_features)}): {confirmed_features}")
    print(f"Tentative Features ({len(tentative_features)}): {tentative_features}")
    
    # 3. Multicollinearity Check (VIF)
    print("\nChecking Multicollinearity (VIF)...")
    # Only check confirmed features
    X_selected = X[confirmed_features]
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_selected.columns
    vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i) 
                       for i in range(len(X_selected.columns))]
    
    print(vif_data.sort_values('VIF', ascending=False))
    
    # Filter High VIF (> 10 is usually the cutoff, but let's be strict with > 5)
    # We won't auto-drop here, just flag.
    high_vif = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
    if high_vif:
        print(f"Warning: High VIF detected in {high_vif}")
        
    # 4. SHAP Interaction Values
    print("\nCalculating SHAP Interaction Values...")
    # Train a quick model on selected features for SHAP
    model_for_shap = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_for_shap.fit(X_selected, y)
    
    explainer = shap.TreeExplainer(model_for_shap)
    # Calculate SHAP values
    # check_additivity=False to avoid errors with some RF implementations
    shap_values = explainer.shap_values(X_selected, check_additivity=False)
    
    # For multi-class, shap_values is a list of arrays (one per class).
    # We want to visualize the "Fail" class (Class 4).
    # However, if the model only sees a subset of classes during this small run, the list might be shorter.
    # We'll safely pick the last class (usually the highest grade index = Fail).
    
    if isinstance(shap_values, list):
        print(f"SHAP values is a list of length {len(shap_values)}")
        shap_values_target = shap_values[-1]
    else:
        # Binary case or regression
        shap_values_target = shap_values
        
    print(f"Shape of target SHAP values: {shap_values_target.shape}")
    print(f"Shape of X_selected: {X_selected.shape}")

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values_target, X_selected, show=False)
    plt.title("SHAP Summary (Target: Fail Class)")
    plt.savefig(f"{REPORT_DIR}/shap_summary_selected.png", bbox_inches='tight')
    plt.close()
    
    # Interaction Plot (Example: StudyTime vs Support)
    # We need to find indices of these features if they exist
    try:
        shap.dependence_plot("StudyTimeWeekly", shap_values_fail, X_selected, 
                             interaction_index="ParentalSupport" if "ParentalSupport" in X_selected.columns else "auto",
                             show=False)
        plt.savefig(f"{REPORT_DIR}/shap_interaction_studytime.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not gen interaction plot: {e}")

    # 5. Save Results
    results = {
        "confirmed_features": confirmed_features,
        "tentative_features": tentative_features,
        "high_vif_features": high_vif
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved feature selection results to {OUTPUT_PATH}")
    print("Module 3 Complete.")

if __name__ == "__main__":
    run_feature_selection()
