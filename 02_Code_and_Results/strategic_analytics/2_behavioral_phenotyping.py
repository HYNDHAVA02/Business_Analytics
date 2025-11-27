import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_PATH = "strategic_analytics/data/causal_balanced_train.parquet"
OUTPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
REPORT_DIR = "strategic_analytics/reports"
MODEL_DIR = "strategic_analytics/models"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_phenotyping():
    print("--- Module 2: Behavioral Phenotyping & Manifold Learning ---")
    
    # 1. Load Data
    try:
        df = pd.read_parquet(INPUT_PATH)
        print(f"Loaded balanced data from {INPUT_PATH}")
    except FileNotFoundError:
        print("Input data not found. Please run Module 1 first.")
        return

    # Prepare features for clustering (exclude target)
    X = df.drop(columns=['GradeClass', 'GPA'], errors='ignore')
    # Ensure all numeric
    X = X.select_dtypes(include=[np.number])
    
    # 2. UMAP Projection (Manifold Learning)
    print("Running UMAP Projection...")
    # UMAP parameters: n_neighbors controls local vs global structure
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    embedding = reducer.fit_transform(X)
    
    df['UMAP_1'] = embedding[:, 0]
    df['UMAP_2'] = embedding[:, 1]
    
    # 3. HDBScan Clustering (Density-Based)
    print("Running HDBScan Clustering...")
    # min_cluster_size: smallest size grouping to consider a cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        gen_min_span_tree=True,
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(X)
    
    df['Persona_Cluster'] = cluster_labels
    
    # -1 indicates noise/outliers in HDBScan
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Identified {n_clusters} clusters and {n_noise} noise points.")
    
    # 4. Stability Analysis (Simplified)
    # We calculate silhouette score for non-noise points
    if n_clusters > 1:
        mask = cluster_labels != -1
        sil_score = silhouette_score(X[mask], cluster_labels[mask])
        print(f"Silhouette Score (valid clusters): {sil_score:.4f}")
    else:
        print("Not enough clusters for silhouette score.")
        
    # 5. Visualization
    print("Generating Phenotype Map...")
    plt.figure(figsize=(12, 8))
    
    # Plot noise as grey
    noise_mask = df['Persona_Cluster'] == -1
    plt.scatter(df.loc[noise_mask, 'UMAP_1'], df.loc[noise_mask, 'UMAP_2'], 
                c='lightgrey', s=10, label='Outliers/Noise')
    
    # Plot clusters
    cluster_mask = ~noise_mask
    scatter = plt.scatter(df.loc[cluster_mask, 'UMAP_1'], df.loc[cluster_mask, 'UMAP_2'], 
                          c=df.loc[cluster_mask, 'Persona_Cluster'], 
                          cmap='viridis', s=20, alpha=0.8)
    
    plt.colorbar(scatter, label='Persona Cluster ID')
    plt.title('Student Behavioral Phenotypes (UMAP + HDBScan)', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    
    plot_path = f"{REPORT_DIR}/phenotypes.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Phenotype map saved to {plot_path}")
    
    # 6. Persona Labeling (Centroid Analysis)
    print("Analyzing Cluster Personas...")
    # Calculate mean values for key features per cluster
    key_features = ['StudyTimeWeekly', 'Absences', 'ParentalEducation', 'InternetAccess']
    # Ensure these exist
    available_features = [f for f in key_features if f in df.columns]
    
    persona_summary = df.groupby('Persona_Cluster')[available_features + ['GradeClass']].mean()
    print("\nPersona Summary (Mean Values):")
    print(persona_summary)
    
    # Save summary
    persona_summary.to_csv(f"{REPORT_DIR}/persona_summary.csv")
    
    # Save Processed Data with Clusters
    df.to_parquet(OUTPUT_PATH)
    print(f"Saved Phenotyped data to {OUTPUT_PATH}")
    
    # Save Models
    joblib.dump(reducer, f"{MODEL_DIR}/umap_reducer.pkl")
    joblib.dump(clusterer, f"{MODEL_DIR}/hdbscan_clusterer.pkl")
    
    print("Module 2 Complete.")

if __name__ == "__main__":
    run_phenotyping()
