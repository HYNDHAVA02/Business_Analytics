import joblib
import pandas as pd
import numpy as np
import json
import os

SCALER_PATH = "strategic_analytics/models/robust_scaler.pkl"
PT_PATH = "strategic_analytics/models/power_transformer.pkl"
MODEL_PATH = "strategic_analytics/models/ordinal_model.pkl"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"

def inspect_artifacts():
    print("--- Inspecting Artifacts ---")
    
    # Load Scalers
    try:
        rs = joblib.load(SCALER_PATH)
        pt = joblib.load(PT_PATH)
        print("Scalers loaded successfully.")
        
        if hasattr(pt, 'feature_names_in_'):
            print(f"PowerTransformer expects features: {pt.feature_names_in_}")
        else:
            print("PowerTransformer does NOT have 'feature_names_in_'.")
            
        if hasattr(rs, 'feature_names_in_'):
            print(f"RobustScaler expects features: {rs.feature_names_in_}")
        else:
            print("RobustScaler does NOT have 'feature_names_in_'.")
            
    except Exception as e:
        print(f"Error loading scalers: {e}")

    # Load Model
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        # Try loading the .cbm model
        MODEL_PATH_NEW = "models/best_catboost_model.cbm"
        model.load_model(MODEL_PATH_NEW)
        print(f"Model loaded from {MODEL_PATH_NEW}")
        print(f"Model feature names: {model.feature_names_}")
        print(f"Model Classes: {model.classes_}")
    except Exception as e:
        print(f"Error loading model: {e}")

    # Test Scaling
    print("\n--- Test Scaling ---")
    # Create a dummy input with known bad values
import joblib
import pandas as pd
import numpy as np
import json
import os

SCALER_PATH = "strategic_analytics/models/robust_scaler.pkl"
PT_PATH = "strategic_analytics/models/power_transformer.pkl"
MODEL_PATH = "strategic_analytics/models/ordinal_model.pkl"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"

def inspect_artifacts():
    print("--- Inspecting Artifacts ---")
    
    # Load Scalers
    try:
        rs = joblib.load(SCALER_PATH)
        pt = joblib.load(PT_PATH)
        print("Scalers loaded successfully.")
        
        if hasattr(pt, 'feature_names_in_'):
            print(f"PowerTransformer expects features: {pt.feature_names_in_}")
        else:
            print("PowerTransformer does NOT have 'feature_names_in_'.")
            
        if hasattr(rs, 'feature_names_in_'):
            print(f"RobustScaler expects features: {rs.feature_names_in_}")
        else:
            print("RobustScaler does NOT have 'feature_names_in_'.")
            
    except Exception as e:
        print(f"Error loading scalers: {e}")

    # Load Model
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        # Try loading the .cbm model
        MODEL_PATH_NEW = "models/best_catboost_model.cbm"
        model.load_model(MODEL_PATH_NEW)
        print(f"Model loaded from {MODEL_PATH_NEW}")
        print(f"Model feature names: {model.feature_names_}")
        print(f"Model Classes: {model.classes_}")
    except Exception as e:
        print(f"Error loading model: {e}")

    # Test Scaling
    print("\n--- Test Scaling ---")
    # Create a dummy input with known bad values
    # We need to guess the columns if feature_names_in_ is missing.
    # Based on the screenshot, the raw input had many columns.
    
    # Let's try to reconstruct the training columns from 1_causal_integrity.py logic if needed.
    # But first, let's see the output of this script.

    # Load Preprocessor
    try:
        PREPROCESSOR_PATH = "models/preprocessor.pkl"
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        return

    print("\n--- Test Full Pipeline ---")
    
    # Feature Engineering Function
    def feature_engineering(df):
        df = df.copy()
        df['TechSynergy'] = df['InternetAccess'] * df['StudyTimeWeekly']
        df['SelfDriven'] = df['StudyTimeWeekly'] / (df['ParentalEducation'] + 1)
        df['SupportIndex'] = df['ParentalEducation'] + df['Tutoring'] + df['InternetAccess']
        df['BalancedLife'] = df['Extracurricular'] * df['StudyTimeWeekly']
        df['AbsenceRisk'] = df['Absences'] * (1 - df['InternetAccess'])
        return df

    # Create Raw Input (Absences=29)
    # We need all columns expected by the preprocessor's transformers
    # Numeric: StudyTimeWeekly, Absences
    # Ordinal: ParentalEducation
    # Nominal: Gender, Tutoring, Extracurricular, InternetAccess
    
    data = {
        'StudyTimeWeekly': 1,
        'Absences': 29,
        'ParentalEducation': 0,
        'Gender': 0,
        'Tutoring': 0,
        'Extracurricular': 0,
        'InternetAccess': 0,
        # Extra cols to avoid errors if preprocessor expects them (though it shouldn't if defined correctly)
        'GradeClass': 2, 
        'GPA': 0.0
    }
    
    df_raw = pd.DataFrame([data])
    print("Raw Input:")
    print(df_raw)
    
    # 1. Feature Engineering
    df_eng = feature_engineering(df_raw)
    print("\nEngineered Features:")
    print(df_eng[['TechSynergy', 'SelfDriven', 'SupportIndex', 'BalancedLife', 'AbsenceRisk']])
    
    # 2. Preprocessing
    try:
        # Transform
        X_processed = preprocessor.transform(df_eng)
        print("\nProcessed Output Shape:", X_processed.shape)
        print("Processed Output (First 5):", X_processed[0][:5])
        
        # 3. Predict
        pred = model.predict(X_processed)
        proba = model.predict_proba(X_processed)
        
        print(f"\nPrediction: {pred[0]}")
        print(f"Probabilities: {proba[0]}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    with open("debug_output_utf8.txt", "w", encoding="utf-8") as f:
        import sys
        sys.stdout = f
        inspect_artifacts()
