import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

def plot_distributions(df, features, output_dir):
    """Plots distributions of features."""
    for feature in features:
        if feature in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[feature], kde=True)
            plt.title(f'Distribution of {feature} (Transformed)')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/dist_{feature}.png')
            plt.close()

def test_hypothesis(df, group_col, target_col, hypothesis_name, output_dir):
    """Performs Mann-Whitney U test and plots boxplot."""
    # Create groups based on median split if continuous
    if df[group_col].nunique() > 2:
        median_val = df[group_col].median()
        df['Group'] = np.where(df[group_col] > median_val, 'High', 'Low')
    else:
        df['Group'] = df[group_col]

    high_group = df[df['Group'] == 'High'][target_col]
    low_group = df[df['Group'] == 'Low'][target_col]

    stat, p_value = mannwhitneyu(high_group, low_group)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Group', y=target_col, data=df)
    plt.title(f'{hypothesis_name}\nMann-Whitney U p-value: {p_value:.4f}')
    plt.savefig(f'{output_dir}/hypothesis_{hypothesis_name}.png')
    plt.close()
    
    return p_value

def plot_interaction(df, x_col, y_col, hue_col, output_dir):
    """Plots interaction effects."""
    plt.figure(figsize=(10, 6))
    # Bin x_col for better visualization if it's continuous
    df['X_Binned'] = pd.qcut(df[x_col], q=5, duplicates='drop')
    
    sns.pointplot(x='X_Binned', y=y_col, hue=hue_col, data=df)
    plt.title(f'Interaction: {x_col} vs {y_col} by {hue_col}')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/interaction_{x_col}_{hue_col}.png')
    plt.close()

def main():
    print("Loading processed training data...")
    df = pd.read_parquet("data/processed/train.parquet")
    
    output_dir = "visuals/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Distribution Plots...")
    # Plot key transformed features
    features_to_plot = ['StudyTimeWeekly', 'Absences', 'TechSynergy', 'SelfDriven', 'SupportIndex']
    plot_distributions(df, features_to_plot, output_dir)
    
    print("Testing Hypotheses...")
    # 1. TechSynergy: Does high synergy lead to better GPA?
    p_tech = test_hypothesis(df, 'TechSynergy', 'GPA', 'TechSynergy_Effect', output_dir)
    print(f"TechSynergy Effect p-value: {p_tech}")
    
    # 2. BalancedLife: Do extracurriculars hurt GPA?
    # Here we check if 'BalancedLife' (Interaction) is better than low.
    # Or simpler: Compare Extracurricular Yes vs No on GPA
    # Note: Extracurricular is likely scaled/encoded. 
    # Let's use the 'BalancedLife' feature directly.
    p_balance = test_hypothesis(df, 'BalancedLife', 'GPA', 'BalancedLife_Effect', output_dir)
    print(f"BalancedLife Effect p-value: {p_balance}")
    
    # 3. AbsenceRisk: Does high risk lead to lower GradeClass?
    # GradeClass is categorical (0-4), but treated as numeric in parquet for convenience?
    # Let's check. In 2_process_engineer.py, we saved GradeClass as is.
    p_risk = test_hypothesis(df, 'AbsenceRisk', 'GradeClass', 'AbsenceRisk_Effect', output_dir)
    print(f"AbsenceRisk Effect p-value: {p_risk}")
    
    print("Generating Interaction Plots...")
    # TechSynergy Interaction: StudyTime vs GPA by InternetAccess
    # We need to reconstruct 'InternetAccess' binary from the processed data if possible.
    # Or use the 'TechSynergy' feature itself.
    # Since we don't have raw 'InternetAccess' easily accessible (it's encoded/scaled),
    # we can try to infer or just use the derived feature.
    # Actually, let's use 'TechSynergy' vs GPA directly as the interaction proxy.
    
    # Better: Load raw validated data to get categorical labels for hue, 
    # join with processed data index.
    df_raw = pd.read_csv("data/raw/validated_students.csv")
    df_joined = df.join(df_raw[['InternetAccess']], rsuffix='_raw')
    
    plot_interaction(df_joined, 'StudyTimeWeekly', 'GPA', 'InternetAccess_raw', output_dir)
    
    print("EDA Complete. Visuals saved to visuals/eda/")

if __name__ == "__main__":
    main()
