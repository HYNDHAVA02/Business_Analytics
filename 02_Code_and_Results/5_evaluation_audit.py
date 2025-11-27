import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix, classification_report
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, count
import joblib
import json
import os

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def perform_fairness_audit(y_true, y_pred, sensitive_features, output_dir):
    # We define "Pass" as GradeClass < 4 (assuming 4 is Fail/lowest, 0 is best)
    # Or let's check the mapping. 
    # In Combining_Data.py: bins=[-1, 2.0, 2.5, 3.0, 3.5, 4.5], labels=[4, 3, 2, 1, 0]
    # So 0 is best (GPA > 3.5), 4 is worst (GPA <= 2.0).
    # Let's define "Positive Outcome" as GradeClass <= 2 (GPA > 2.5).
    
    y_true_binary = (y_true <= 2).astype(int)
    y_pred_binary = (y_pred <= 2).astype(int)
    
    metrics = {
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
        'count': count
    }
    
    audit_results = {}
    
    for col in sensitive_features.columns:
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true_binary,
            y_pred=y_pred_binary,
            sensitive_features=sensitive_features[col]
        )
        
        # Save detailed group metrics
        mf.by_group.to_csv(f'{output_dir}/fairness_metrics_{col}.csv')
        
        # Calculate disparities
        audit_results[col] = {
            'demographic_parity_diff': mf.difference(method='between_groups')['selection_rate'],
            'equalized_odds_diff': mf.difference(method='between_groups')['true_positive_rate']
        }
        
        # Plot
        mf.by_group[['selection_rate', 'true_positive_rate']].plot(kind='bar', figsize=(10, 5))
        plt.title(f'Fairness Metrics by {col}')
        plt.ylabel('Rate')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fairness_plot_{col}.png')
        plt.close()
        
    with open(f'{output_dir}/fairness_audit.json', 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    return audit_results

def main():
    print("Loading test data...")
    df_test = pd.read_parquet("data/processed/test.parquet")
    X_test = df_test.drop(columns=['GradeClass', 'GPA'])
    y_test = df_test['GradeClass'].astype(int)
    
    # Load raw validated data to get sensitive features (Gender, InternetAccess)
    # We need to join by index to ensure alignment
    df_raw = pd.read_csv("data/raw/validated_students.csv")
    sensitive_cols = ['Gender', 'InternetAccess']
    sensitive_features = df_raw.loc[X_test.index, sensitive_cols]
    
    print("Loading model...")
    model = CatBoostClassifier()
    model.load_model("models/best_catboost_model.cbm")
    
    print("Predicting...")
    y_pred = model.predict(X_test).flatten()
    
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    visuals_dir = "visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Metrics
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test QWK: {qwk:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (Weighted): {f1:.4f}")
    
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"QWK: {qwk}\nAccuracy: {acc}\nF1: {f1}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, visuals_dir)
    
    # SHAP Analysis
    print("Performing SHAP Analysis...")
    # try:
    #     explainer = shap.TreeExplainer(model)
    #     shap_values = explainer.shap_values(X_test)
        
    #     # Check if shap_values is a list (multi-class) or array
    #     if isinstance(shap_values, list):
    #         print(f"SHAP values shape: List of {len(shap_values)} arrays, each {shap_values[0].shape}")
    #         # Class 0 (Best Grade)
    #         plt.figure()
    #         shap.summary_plot(shap_values[0], X_test, show=False)
    #         plt.title("SHAP Summary (Class 0: Best Grade)")
    #         plt.tight_layout()
    #         plt.savefig(f'{visuals_dir}/shap_summary_class0.png')
    #         plt.close()
            
    #         # Class 4 (Fail) - Check if index exists
    #         if len(shap_values) > 4:
    #             plt.figure()
    #             shap.summary_plot(shap_values[4], X_test, show=False)
    #             plt.title("SHAP Summary (Class 4: Fail)")
    #             plt.tight_layout()
    #             plt.savefig(f'{visuals_dir}/shap_summary_class4.png')
    #             plt.close()
                
    #         # Dependence Plot (Class 0)
    #         if 'TechSynergy' in X_test.columns:
    #             plt.figure()
    #             shap.dependence_plot("TechSynergy", shap_values[0], X_test, show=False)
    #             plt.title("SHAP Dependence: TechSynergy (Class 0)")
    #             plt.tight_layout()
    #             plt.savefig(f'{visuals_dir}/shap_dependence_techsynergy.png')
    #             plt.close()
    #     else:
    #         # Binary/Regression case
    #         print(f"SHAP values shape: {shap_values.shape}")
    #         plt.figure()
    #         shap.summary_plot(shap_values, X_test, show=False)
    #         plt.savefig(f'{visuals_dir}/shap_summary.png')
    #         plt.close()
            
    # except Exception as e:
    #     print(f"SHAP Error: {e}")
        
    # Fairness Audit
    print("Performing Fairness Audit...")
    audit = perform_fairness_audit(y_test, y_pred, sensitive_features, output_dir)
    print("Fairness Audit Complete.")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
