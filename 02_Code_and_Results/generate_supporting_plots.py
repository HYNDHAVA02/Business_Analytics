import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure visuals directory exists
os.makedirs("visuals", exist_ok=True)

def plot_ratio_comparison():
    print("Generating Ratio Comparison Plot...")
    # Data from Report/Experiments
    ratios = [0.1, 0.2, 0.3, 0.4]
    # Strategic (CatBoost) QWK scores
    strategic_scores = [0.7313, 0.7406, 0.7315, 0.7365] 
    # Baseline (RF) QWK scores (from baseline_implementation/ratio_results.json)
    baseline_scores = [0.7655, 0.7714, 0.7687, 0.7711]

    plt.figure(figsize=(10, 6))
    plt.plot(ratios, strategic_scores, marker='o', label='Strategic (CatBoost)', linewidth=2, markersize=8)
    plt.plot(ratios, baseline_scores, marker='s', label='Baseline (Random Forest)', linewidth=2, markersize=8, linestyle='--')
    
    plt.title('Model Stability: QWK Score across Test Split Ratios', fontsize=14)
    plt.xlabel('Test Set Size (Ratio)', fontsize=12)
    plt.ylabel('Quadratic Weighted Kappa (QWK)', fontsize=12)
    plt.ylim(0.70, 0.80)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    output_path = "visuals/ratio_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

def plot_model_performance_comparison():
    print("Generating Model Performance Comparison Plot...")
    # Data from Report Table
    models = ['Initial Optuna', 'Fine-Tuned', 'Class-Weighted', 'Baseline (RF)']
    qwk_scores = [0.7502, 0.7314, 0.6814, 0.7636]
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3'] # Seaborn muted palette

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, qwk_scores, color=colors, alpha=0.8)
    
    plt.title('Comparative Model Performance (QWK Score)', fontsize=14)
    plt.ylabel('Quadratic Weighted Kappa', fontsize=12)
    plt.ylim(0.60, 0.80)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    output_path = "visuals/model_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    
    plot_ratio_comparison()
    plot_model_performance_comparison()
