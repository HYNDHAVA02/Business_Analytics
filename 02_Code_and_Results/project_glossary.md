# Project Glossary: Algorithms & Techniques

This document provides a comprehensive list of all technical methods used in the "Strategic Educational Data Mining" project, including their full forms and simplified explanations of how they work.

---

## 1. Data Preprocessing & Engineering

### **MICE**
*   **Full Form:** Multivariate Imputation by Chained Equations
*   **How it Works:** Instead of filling missing values with a simple average (which distorts data), MICE looks at other columns to make an "educated guess." For example, if a student has a high GPA, MICE infers they likely have high Study Time, preserving the natural relationship between variables.

### **Yeo-Johnson Transformation**
*   **Full Form:** Yeo-Johnson Power Transformation
*   **How it Works:** It is a mathematical function that makes skewed data look more like a "Bell Curve" (Normal Distribution). This is critical for algorithms that assume data is symmetrical. Unlike Box-Cox, it works on zero and negative numbers (e.g., 0 Absences).

### **RobustScaler**
*   **Full Form:** Robust Scaler (based on Interquartile Range)
*   **How it Works:** Standard scaling uses the Mean and Variance, which are easily thrown off by outliers. RobustScaler uses the Median and the IQR (25th-75th percentile), effectively "ignoring" extreme outliers when scaling the data so they don't crush the rest of the distribution.

---

## 2. Advanced Analytics & Discovery

### **PC Algorithm**
*   **Full Form:** Peter-Clark Algorithm (for Causal Discovery)
*   **How it Works:** It starts by assuming everything is connected to everything. Then, it performs a series of statistical tests to "cut" links that are just correlations, leaving behind a Directed Acyclic Graph (DAG) that represents the likely *causal* structure (e.g., Study Time -> GPA, not vice versa).

### **UMAP**
*   **Full Form:** Uniform Manifold Approximation and Projection
*   **How it Works:** A dimensionality reduction technique (like PCA but better for clusters). It imagines the data points as stars in a high-dimensional universe and tries to draw a 2D map that preserves the "local neighborhood" of each star. This allows us to see distinct clusters ("Learner Personas") that linear methods miss.

### **Boruta**
*   **Full Form:** Boruta Feature Selection Algorithm (named after a Slavic forest god)
*   **How it Works:** It creates "Shadow Features" (randomly shuffled copies of real features). It then trains a Random Forest. If a real feature (e.g., Absences) doesn't perform better than its random shadow copy, it is deemed "useless" and removed. This ensures we only keep features with *proven* signal.

---

## 3. Predictive Modeling

### **CatBoost**
*   **Full Form:** Categorical Boosting
*   **How it Works:** A Gradient Boosting algorithm (an ensemble of decision trees). It is unique because it handles categorical data (like "Parental Education: High") natively without needing One-Hot Encoding (which creates messy data). It builds trees sequentially, where each new tree tries to fix the errors of the previous one.

### **Random Forest (Baseline)**
*   **Full Form:** Random Forest Classifier
*   **How it Works:** It builds hundreds of independent Decision Trees, each looking at a random subset of data and features. It then takes a "Majority Vote" from all trees. It is robust and hard to overfit, making it a perfect baseline for comparison.

### **Hybrid Stacking**
*   **Full Form:** Stacking Ensemble Classifier
*   **How it Works:** It trains multiple "Base Learners" (CatBoost, Random Forest, MLP). Then, instead of just averaging them, it trains a "Meta-Learner" (Logistic Regression) to learn *when* to trust which model. For example, it might learn to trust CatBoost for students with high absences but Random Forest for others.

### **CoxPH**
*   **Full Form:** Cox Proportional Hazards Model
*   **How it Works:** Used for Survival Analysis. Instead of predicting "Will they fail?", it predicts "How *fast* will they fail?". It calculates a "Hazard Ratio" – for example, a ratio of 2.0 means a student is dropping out twice as fast as the baseline.

---

## 4. Evaluation & Explainability

### **QWK**
*   **Full Form:** Quadratic Weighted Kappa
*   **How it Works:** An accuracy metric for ordinal data (grades A, B, C, D, F). It penalizes "bad" mistakes more than "small" mistakes. Predicting an 'A' student as 'F' is a huge penalty (squared error), whereas predicting 'A' as 'B' is a small penalty. Standard Accuracy treats both errors the same.

### **SHAP**
*   **Full Form:** SHapley Additive exPlanations
*   **How it Works:** Based on Game Theory. It treats each feature as a "player" in a game (the prediction). It calculates the marginal contribution of each player to the final score. If removing "Absences" changes the prediction from 'F' to 'B', then "Absences" gets a high SHAP value.

### **Optuna**
*   **Full Form:** Optuna (Hyperparameter Optimization Framework)
*   **How it Works:** Instead of trying every random combination of settings (Grid Search), it uses Bayesian Optimization (TPE - Tree-structured Parzen Estimator). It learns from previous trials: "High learning rate was bad, let's try low learning rate." This finds the best model settings much faster.
