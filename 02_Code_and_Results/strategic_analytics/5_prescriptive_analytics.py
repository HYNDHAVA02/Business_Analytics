import pandas as pd
import numpy as np
import dice_ml
from catboost import CatBoostClassifier
import joblib
import json
import os
import warnings
import random

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"
MODEL_PATH = "strategic_analytics/models/ordinal_model.pkl"
REPORT_DIR = "strategic_analytics/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

def run_prescriptive_analytics():
    print("--- Module 5: Prescriptive Analytics & Actionable Recourse ---")
    
    # 1. Load Data, Features, Model
    try:
        df = pd.read_parquet(INPUT_PATH)
        with open(FEATURE_PATH, 'r') as f:
            features_json = json.load(f)
            selected_features = features_json['confirmed_features']
            if 'Persona_Cluster' not in selected_features and 'Persona_Cluster' in df.columns:
                selected_features.append('Persona_Cluster')
        
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        print("Loaded data, features, and model.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # Prepare Data for DiCE
    # DiCE needs a dataset with target
    # Target: GradeClass. We want to flip Fail (4) to Pass (0, 1, 2, 3).
    # For simplicity, let's treat it as Binary for DiCE: Fail (4) vs Pass (<4)
    # Or just ask DiCE to change the class to 0 (A) or 2 (C).
    
    # DiCE works best with sklearn-wrapped models. CatBoost is fine.
    # We need to pass the dataframe with features + target
    target_col = 'GradeClass'
    df_dice = df[selected_features + [target_col]].copy()
    df_dice[target_col] = df_dice[target_col].round().astype(int)
    
    # 2. Counterfactual Explanations (DiCE)
    print("\nGenerating Counterfactuals (DiCE)...")
    
    # Define Data object
    # Continuous features
    continuous_features = df_dice.select_dtypes(include=[np.number]).columns.tolist()
    continuous_features.remove(target_col)
    
    d = dice_ml.Data(dataframe=df_dice, continuous_features=continuous_features, outcome_name=target_col)
    
    # Define Model object
    # We need a wrapper because DiCE expects predict() to return just classes, 
    # but CatBoost might return something else depending on version.
    # Actually, dice_ml.Model(model=model, backend='sklearn') usually works for CatBoost.
    m = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')
    
    # Initialize DiCE
    exp = dice_ml.Dice(d, m, method="random")
    
    # Pick an at-risk student (GradeClass = 4)
    at_risk_students = df_dice[df_dice[target_col] == 4]
    if len(at_risk_students) > 0:
        query_instance = at_risk_students.iloc[0:1].drop(columns=[target_col])
        
        print("Query Instance (At-Risk Student):")
        print(query_instance)
        
        # Generate Counterfactuals
        # Desired class: 2 (C grade) - realistic goal
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=3, 
                desired_class=2,
                features_to_vary=['StudyTimeWeekly', 'Absences', 'ParentalEducation'] # Constrain actionable features
            )
            
            print("\nSuggested Interventions (Counterfactuals):")
            dice_exp.visualize_as_dataframe(show_only_changes=True)
            
            # Save to HTML
            with open(f"{REPORT_DIR}/counterfactuals.html", "w") as f:
                f.write(dice_exp.as_html())
            print(f"Saved Counterfactuals to {REPORT_DIR}/counterfactuals.html")
            
        except Exception as e:
            print(f"DiCE generation failed: {e}")
    else:
        print("No at-risk students found for demo.")

    # 3. Contextual Bandits (Simulation)
    print("\nSimulating Contextual Bandits for Intervention Optimization...")
    
    # Actions: [0: No Action, 1: Email Nudge, 2: Tutor, 3: Grant]
    actions = ['No Action', 'Email Nudge', 'Tutor', 'Grant']
    n_actions = len(actions)
    
    # Reward Function (Simulated Environment)
    # We assume:
    # - Tutor works best for 'Struggling' (Cluster X)
    # - Grant works best for 'Low SES' (Cluster Y)
    # - Nudge works best for 'Coaster' (Cluster Z)
    
    def get_reward(student_row, action_idx):
        # Simplified logic based on features
        reward = 0
        
        # Base probability of improvement
        base_prob = 0.1
        
        if action_idx == 0: # No Action
            return 0 if random.random() > 0.05 else 1 # Small chance of self-improvement
            
        if action_idx == 1: # Nudge
            # Good for high support but low study time
            # Proxy: ParentalEducation > 2 (Some College+)
            if 'ParentalEducation' in student_row and student_row['ParentalEducation'] > 2 and student_row['StudyTimeWeekly'] < 5:
                base_prob += 0.3
                
        if action_idx == 2: # Tutor
            # Good for low GPA but high absences (needs catch up)
            if student_row['GPA'] < 2.0:
                base_prob += 0.4
                
        if action_idx == 3: # Grant
            # Good for low SES (proxy: ParentalEducation < 2?)
            if 'ParentalEducation' in student_row and student_row['ParentalEducation'] < 2:
                base_prob += 0.5
                
        return 1 if random.random() < base_prob else 0

    # Epsilon-Greedy Policy
    epsilon = 0.1
    q_values = np.zeros(n_actions)
    action_counts = np.zeros(n_actions)
    
    rewards_history = []
    selected_actions = []
    
    # Simulate for 1000 students
    sim_students = df.sample(1000, replace=True)
    
    for idx, row in sim_students.iterrows():
        # Choose Action
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1) # Explore
        else:
            action = np.argmax(q_values) # Exploit
            
        # Get Reward
        reward = get_reward(row, action)
        
        # Update Q-Values
        action_counts[action] += 1
        q_values[action] += (reward - q_values[action]) / action_counts[action]
        
        rewards_history.append(reward)
        selected_actions.append(action)
        
    print("\nBandit Results (Epsilon-Greedy):")
    for i, act in enumerate(actions):
        print(f"Action: {act}, Est. Value: {q_values[i]:.4f}, Count: {int(action_counts[i])}")
        
    best_action = actions[np.argmax(q_values)]
    print(f"Optimal Global Intervention: {best_action}")
    
    # Save Bandit Report
    with open(f"{REPORT_DIR}/bandit_report.txt", "w") as f:
        f.write("Contextual Bandit Simulation Results\n")
        f.write("====================================\n")
        for i, act in enumerate(actions):
            f.write(f"Action: {act}, Value: {q_values[i]:.4f}, Count: {int(action_counts[i])}\n")
        f.write(f"\nOptimal Global Action: {best_action}\n")
        
    # --- 4. Off-Policy Evaluation (OPE) with IPS ---
    print("\nRunning Off-Policy Evaluation (OPE) with IPS...")
    # Simulate Historical Log (Behavior Policy)
    # Assume historical policy was random (uniform)
    n_samples = 1000
    historical_actions = np.random.randint(0, n_actions, n_samples)
    propensities = np.ones(n_samples) / n_actions # Uniform probability
    
    # Simulate Rewards for historical actions
    historical_rewards = []
    sim_students_ope = df.sample(n_samples, replace=True).reset_index(drop=True)
    for idx, row in sim_students_ope.iterrows():
        historical_rewards.append(get_reward(row, historical_actions[idx]))
    historical_rewards = np.array(historical_rewards)
    
    # Evaluate Target Policy (Our Trained Bandit)
    # For simplicity, let's evaluate the "Greedy" policy derived from our Q-values above
    # Target Policy: Always choose 'best_action' (or argmax Q)
    # Actually, let's use the Q-values we learned to define the target policy
    target_actions = []
    for idx, row in sim_students_ope.iterrows():
        # In a real bandit, we'd use the context to pick action. 
        # Here our Q-values are global (Context-Free Bandit for demo).
        # To make it Contextual, we'd need a model predicting reward per context.
        # Let's assume our "Target Policy" is the "Optimal Global Action" we found.
        target_actions.append(np.argmax(q_values))
    target_actions = np.array(target_actions)
    
    # IPS Estimator
    # V_IPS = mean( (I(pi_target == a) / pi_behavior) * Reward )
    ips_values = []
    for i in range(n_samples):
        if historical_actions[i] == target_actions[i]:
            weight = 1.0 / propensities[i]
            ips_values.append(weight * historical_rewards[i])
        else:
            ips_values.append(0)
            
    v_ips = np.mean(ips_values)
    v_random = np.mean(historical_rewards)
    
    print(f"Historical Random Policy Value: {v_random:.4f}")
    print(f"Target Policy (IPS Estimated) Value: {v_ips:.4f}")
    print(f"Improvement: {(v_ips - v_random) / v_random * 100:.1f}%")
    
    with open(f"{REPORT_DIR}/ope_report.txt", "w") as f:
        f.write("Off-Policy Evaluation (OPE) Results\n")
        f.write("===================================\n")
        f.write(f"Historical Random Policy Value: {v_random:.4f}\n")
        f.write(f"Target Policy (IPS Estimated) Value: {v_ips:.4f}\n")
        f.write(f"Improvement: {(v_ips - v_random) / v_random * 100:.1f}%\n")
        
    print("Module 5 Updated with OPE.")

if __name__ == "__main__":
    run_prescriptive_analytics()
