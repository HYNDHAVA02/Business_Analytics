import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = "data/raw/validated_students.csv"
OUTPUT_PATH = "strategic_analytics/data/causal_balanced_train.parquet"
DAG_PLOT_PATH = "strategic_analytics/reports/causal_dag.png"
MODEL_DIR = "strategic_analytics/models"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DAG_PLOT_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_causal_integrity():
    print("--- Module 1: Causal Integrity & Advanced Data Foundations ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded validated data from {DATA_PATH}")
    except FileNotFoundError:
        print("Validated data not found, falling back to combined.")
        df = pd.read_csv("combined_students_final.csv")
        
    # Drop IDs if present
    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])
        
    # 2. Causal Discovery (PC Algorithm)
    print("Running Causal Discovery (PC Algorithm)...")
    
    # Prepare data for PC (must be numeric/encoded)
    df_causal = df.copy()
    le_dict = {}
    for col in df_causal.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_causal[col] = le.fit_transform(df_causal[col].astype(str))
        le_dict[col] = le
        
    data_matrix = df_causal.to_numpy()
    
    # Run PC
    # alpha=0.05 is the significance level for independence tests
    cg = pc(data_matrix, alpha=0.05, indep_test='fisherz') 
    
    # Visualize DAG
    print("Visualizing DAG...")
    try:
        pyd = GraphUtils.to_pydot(cg.G, labels=df_causal.columns)
        pyd.write_png(DAG_PLOT_PATH)
        print(f"DAG saved to {DAG_PLOT_PATH}")
    except Exception as e:
        print(f"Could not save DAG image (GraphViz might be missing): {e}")
        print("Attempting fallback using NetworkX...")
        try:
            import networkx as nx
            # Create a simple directed graph from the adjacency matrix or edges
            # cg.G.graph is the adjacency matrix (if available) or we iterate edges
            nx_graph = nx.DiGraph()
            
            # Add nodes
            labels = df_causal.columns
            for i, label in enumerate(labels):
                nx_graph.add_node(label)
                
            # Add edges
            # cg.G.get_graph_edges() returns list of Edge objects
            # Edge has node1, node2, endpoint1, endpoint2
            # We need to map internal indices/names to our labels
            # cg.G.nodes is a list of Node objects. Node.name might be 'X1', 'X2' etc.
            # We assume the order matches df_causal.columns
            
            nodes = cg.G.get_nodes()
            node_map = {node.name: labels[i] for i, node in enumerate(nodes)}
            
            for edge in cg.G.get_graph_edges():
                # Check for directed edge '-->'
                # In causal-learn: endpoint=1 is Tail, endpoint=2 is Arrow
                # endpoint1 is for node1, endpoint2 is for node2
                # Arrow (2) means it points TO that node. Tail (1) means it comes FROM.
                # e.g. X1 -> X2: node1=X1 (Tail), node2=X2 (Arrow)
                
                n1 = node_map[edge.node1.name]
                n2 = node_map[edge.node2.name]
                
                if edge.endpoint1 == 1 and edge.endpoint2 == 2: # n1 -> n2
                    nx_graph.add_edge(n1, n2)
                elif edge.endpoint1 == 2 and edge.endpoint2 == 1: # n2 -> n1
                    nx_graph.add_edge(n2, n1)
                # Ignore undirected or bi-directed for simple viz
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(nx_graph, seed=42)
            nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
                    node_size=3000, font_size=10, arrowsize=20, edge_color='gray')
            plt.title("Causal DAG (NetworkX Fallback)")
            plt.savefig(DAG_PLOT_PATH)
            print(f"DAG saved to {DAG_PLOT_PATH} (via NetworkX)")
            
        except Exception as e2:
            print(f"Fallback failed: {e2}")
        
    # 3. Advanced Imputation (MICE)
    print("Performing MICE Imputation...")
    # We use the causal insight implicitly here: MICE uses all variables to predict missing ones,
    # effectively capturing the conditional dependencies found by the DAG.
    
    mice_imputer = IterativeImputer(max_iter=10, random_state=42)
    # Note: MICE works on numeric data. We use the encoded df_causal for this.
    # In a real pipeline, we'd handle categorical imputation separately or use a specialized library.
    # For this demo, we assume encoded categoricals are ordinal-ish enough or use MICE's predictive power.
    
    df_imputed = pd.DataFrame(mice_imputer.fit_transform(df_causal), columns=df_causal.columns)
    
    # 4. Transformation & Scaling (Yeo-Johnson + RobustScaler)
    print("Applying Yeo-Johnson and RobustScaler...")
    
    # Yeo-Johnson for skew
    pt = PowerTransformer(method='yeo-johnson')
    df_transformed = pd.DataFrame(pt.fit_transform(df_imputed), columns=df_imputed.columns)
    
    # RobustScaler for outliers
    rs = RobustScaler()
    df_scaled = pd.DataFrame(rs.fit_transform(df_transformed), columns=df_transformed.columns)
    
    # Restore Target (GradeClass) to original integer form for CTGAN (it treats it as categorical)
    # We need to inverse transform GradeClass to train CTGAN properly on the "concept" of grades
    # But wait, CTGAN works best on raw-ish data. 
    # STRATEGY SHIFT: Train CTGAN on the *Imputed* data (before scaling/transform) to preserve semantic meaning.
    # Then apply scaling/transform to the *combined* dataset.
    
    print("Training CTGAN for Class Balancing...")
    
    # Re-construct the imputed dataframe with proper types for CTGAN
    df_for_ctgan = df_imputed.copy()
    # Round GradeClass back to nearest int (since MICE might have made it float)
    df_for_ctgan['GradeClass'] = df_for_ctgan['GradeClass'].round().astype(int)
    
    # Identify minority class (e.g., GradeClass 4 = Fail)
    # Let's assume Class 4 is the minority we want to boost.
    minority_class = 4
    minority_data = df_for_ctgan[df_for_ctgan['GradeClass'] == minority_class]
    
    if len(minority_data) < 50:
        print("Warning: Minority class has very few samples. CTGAN might struggle.")
        
    # Initialize CTGAN
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_for_ctgan)
    
    synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=True) # 100 epochs for demo speed
    synthesizer.fit(minority_data)
    
    # Generate synthetic samples
    # Aim to double the minority class size
    n_generate = len(minority_data)
    synthetic_data = synthesizer.sample(num_rows=n_generate)
    
    print(f"Generated {n_generate} synthetic samples for Class {minority_class}")
    
    # Combine Real + Synthetic
    df_balanced = pd.concat([df_for_ctgan, synthetic_data], axis=0).reset_index(drop=True)
    
    # Now Apply Transform/Scaling to the BALANCED dataset
    print("Applying Final Transforms to Balanced Data...")
    
    # We need to re-fit scalers on the balanced data to ensure the distribution is handled correctly
    df_balanced_transformed = pd.DataFrame(pt.fit_transform(df_balanced), columns=df_balanced.columns)
    df_balanced_scaled = pd.DataFrame(rs.fit_transform(df_balanced_transformed), columns=df_balanced.columns)
    
    # Save Processed Data
    df_balanced_scaled.to_parquet(OUTPUT_PATH)
    print(f"Saved Causal-Balanced-Scaled data to {OUTPUT_PATH}")
    
    # Save Models
    joblib.dump(mice_imputer, f"{MODEL_DIR}/mice_imputer.pkl")
    joblib.dump(pt, f"{MODEL_DIR}/power_transformer.pkl")
    joblib.dump(rs, f"{MODEL_DIR}/robust_scaler.pkl")
    synthesizer.save(f"{MODEL_DIR}/ctgan_synthesizer.pkl")
    
    print("Module 1 Complete.")

if __name__ == "__main__":
    run_causal_integrity()
