import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"
MODEL_PATH = "strategic_analytics/models/hybrid_stack.pkl"
OUTPUT_IMG = "strategic_analytics/reports/hybrid_confusion_matrix.png"

def generate_cm():
    print("Generating Hybrid Confusion Matrix...")
    
    # Load Data
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
    
    # Split (Same seed as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Predict
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['A', 'B', 'C', 'D', 'F'],
                yticklabels=['A', 'B', 'C', 'D', 'F'])
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    plt.title('Confusion Matrix: Hybrid Stacking Ensemble')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Saved confusion matrix to {OUTPUT_IMG}")

if __name__ == "__main__":
    generate_cm()
