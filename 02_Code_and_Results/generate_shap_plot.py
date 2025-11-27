import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import json
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"
MODEL_PATH = "models/best_catboost_model.cbm"
OUTPUT_IMG = "strategic_analytics/reports/shap_summary_selected.png"

def generate_shap():
    print("Generating High-Quality SHAP Summary Plot...")
    
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
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Load Model
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # 3. Align Features
    print("Aligning features...")
    model_features = model.feature_names_
    print(f"Model expects: {model_features}")
    
    # Ensure all model features are in X_test
    # If missing, fill with 0 (though they should be there if we loaded correct data)
    for feat in model_features:
        if feat not in X_test.columns:
            print(f"Warning: Feature {feat} missing in data. Filling with 0.")
            X_test[feat] = 0
            
    # Reorder X_test to match model
    X_test = X_test[model_features]
    
    # 4. Calculate SHAP Values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 4. Plot
    print("Plotting...")
    plt.figure(figsize=(12, 8), dpi=300) # High DPI for clarity
    
    # For multi-class, shap_values is a list of arrays (one per class).
    # We usually visualize the impact on the "Risk" class (e.g., Class 4 - Fail) or aggregate.
    # Or we can just plot for Class 4 to show what drives failure.
    # Alternatively, summary_plot handles multi-class by stacking or we pick one.
    # Let's try plotting for Class 4 (Fail) as it's most critical for "Risk".
    # Or just pass the whole list if summary_plot supports it (it creates a stacked bar).
    # But dot plot is better for "impact".
    # Let's plot for Class 4 (Fail).
    
    # Check shape
    if isinstance(shap_values, list):
        print(f"SHAP values is a list of length {len(shap_values)} (Classes). Plotting for Class 4 (Fail).")
        # Class 4 is index 4
        shap_vals_target = shap_values[4]
    else:
        shap_vals_target = shap_values

    shap.summary_plot(shap_vals_target, X_test, show=False, plot_size=(14, 10))
    
    # Adjust layout to prevent overlap
    plt.title("SHAP Feature Importance (Impact on Grade 'F' Prediction)", fontsize=18, y=1.05)
    plt.tight_layout()
    # Add extra margin at the top
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Saved SHAP plot to {OUTPUT_IMG}")

if __name__ == "__main__":
    generate_shap()
