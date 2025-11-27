import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import os

# Create visuals directory
if not os.path.exists('visuals'):
    os.makedirs('visuals')

# ==========================================================
# 1. Data Loading & Preprocessing
# ==========================================================
print("Loading data...")
try:
    df = pd.read_csv("combined_students_final.csv")
except FileNotFoundError:
    print("Error: 'combined_students_final.csv' not found. Please run Combining_Data.py first.")
    exit()

print(f"Initial shape: {df.shape}")

# Handling Missing Values (if any remain)
df.fillna(df.mean(numeric_only=True), inplace=True) # Simple imputation for numeric
# For categorical, mode or constant
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Outlier Detection & Handling (IQR Method for GPA)
Q1 = df['GPA'].quantile(0.25)
Q3 = df['GPA'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Cap outliers instead of removing to preserve data
df['GPA'] = np.where(df['GPA'] < lower_bound, lower_bound, df['GPA'])
df['GPA'] = np.where(df['GPA'] > upper_bound, upper_bound, df['GPA'])

# Encoding Categorical Variables
# 'DatasetType' is categorical but already numeric-coded. 
# We might want to drop it for modeling if it's just an identifier, or keep it if it implies source difference.
# Let's drop 'DatasetType' for the actual student performance modeling to avoid bias from source.
df_model = df.drop(columns=['DatasetType'])

# ==========================================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================================
print("Performing EDA...")

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_model.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visuals/correlation_heatmap.png')
plt.close()

# GPA Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['GPA'], kde=True, bins=20)
plt.title('Distribution of GPA')
plt.savefig('visuals/gpa_distribution.png')
plt.close()

# Pairplot for selected features
sns.pairplot(df_model[['GPA', 'StudyTimeWeekly', 'Absences', 'ParentalEducation']])
plt.savefig('visuals/pairplot.png')
plt.close()

# ==========================================================
# 3. Feature Extraction / Data Reduction (PCA)
# ==========================================================
print("Performing PCA...")
features = df_model.drop(columns=['GPA', 'GradeClass'])
target_reg = df_model['GPA']
target_clf = df_model['GradeClass']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA()
pca.fit(features_scaled)

# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('PCA Explained Variance (Scree Plot)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig('visuals/pca_scree_plot.png')
plt.close()

# Select components explaining > 90% variance
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
print(f"Selected {n_components} components explaining >90% variance.")

pca_final = PCA(n_components=n_components)
features_pca = pca_final.fit_transform(features_scaled)

# ==========================================================
# 4. Predictive Modelling
# ==========================================================
print("Training Models...")

# Split Data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(features_pca, target_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(features_pca, target_clf, test_size=0.2, random_state=42)

# --- Regression (Predicting GPA) ---
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

reg_results = []
for name, model in reg_models.items():
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    r2 = r2_score(y_test_reg, y_pred)
    reg_results.append({"Model": name, "RMSE": rmse, "R2": r2})

# --- Classification (Predicting GradeClass) ---
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

clf_results = []
for name, model in clf_models.items():
    model.fit(X_train, y_train_clf)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_clf, y_pred)
    prec = precision_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_clf, y_pred, average='weighted', zero_division=0)
    clf_results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
    
    # Confusion Matrix for RF
    if name == "Random Forest Classifier":
        cm = confusion_matrix(y_test_clf, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('visuals/confusion_matrix_rf.png')
        plt.close()

# ==========================================================
# 5. Results & Evaluation
# ==========================================================
print("\n--- Regression Results ---")
df_reg_results = pd.DataFrame(reg_results)
print(df_reg_results)

print("\n--- Classification Results ---")
df_clf_results = pd.DataFrame(clf_results)
print(df_clf_results)

# Save results to CSV for report
df_reg_results.to_csv("regression_results.csv", index=False)
df_clf_results.to_csv("classification_results.csv", index=False)

# ==========================================================
# 6. Business Insights
# ==========================================================
print("\n--- Business Insights ---")
# Feature Importance from Random Forest Regressor (using original features for interpretability)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(features, target_reg) # Fit on original features, not PCA
importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Predicting GPA)")
plt.bar(range(features.shape[1]), importances[indices], align="center")
plt.xticks(range(features.shape[1]), features.columns[indices], rotation=45)
plt.tight_layout()
plt.savefig('visuals/feature_importance.png')
plt.close()

print("Top 3 Influential Factors for GPA:")
for i in range(3):
    print(f"{i+1}. {features.columns[indices[i]]} ({importances[indices[i]]:.4f})")

print("\nAnalysis Complete. Visuals saved to 'visuals/' folder.")
