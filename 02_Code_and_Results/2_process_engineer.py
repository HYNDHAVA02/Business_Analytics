import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, FunctionTransformer, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import CatBoostEncoder
import joblib
import os

def feature_engineering(df):
    """Creates new features based on domain knowledge."""
    df = df.copy()
    
    # 1. TechSynergy: InternetAccess * StudyTimeWeekly
    df['TechSynergy'] = df['InternetAccess'] * df['StudyTimeWeekly']
    
    # 2. SelfDriven: StudyTimeWeekly / (ParentalEducation + 1)
    # Avoid division by zero by adding 1 (ParentalEducation is 0-4)
    df['SelfDriven'] = df['StudyTimeWeekly'] / (df['ParentalEducation'] + 1)
    
    # 3. SupportIndex: ParentalEducation + Tutoring + InternetAccess
    # Simple additive index as a starting point for PCA later if needed, 
    # but for now we keep it explicit as per plan.
    df['SupportIndex'] = df['ParentalEducation'] + df['Tutoring'] + df['InternetAccess']
    
    # 4. BalancedLife: Extracurricular * StudyTimeWeekly
    df['BalancedLife'] = df['Extracurricular'] * df['StudyTimeWeekly']
    
    # 5. AbsenceRisk: Absences * (1 - InternetAccess)
    df['AbsenceRisk'] = df['Absences'] * (1 - df['InternetAccess'])
    
    return df

def main():
    print("Loading validated data...")
    df = pd.read_csv("data/raw/validated_students.csv")
    
    print("Splitting data (80/20 Stratified)...")
    X = df.drop(columns=['GradeClass', 'GPA']) # Target variables
    y_class = df['GradeClass']
    y_reg = df['GPA']
    
    # Stratify by GradeClass to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Align y_reg with the split
    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]
    
    print("Constructing Pipeline...")
    
    # Feature Groups
    numeric_features = ['StudyTimeWeekly', 'Absences']
    ordinal_features = ['ParentalEducation']
    nominal_features = ['Gender', 'Tutoring', 'Extracurricular', 'InternetAccess']
    
    # Preprocessing Steps
    
    # Numeric: MICE Imputation -> Yeo-Johnson -> RobustScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=0)),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ])
    
    # Ordinal: Simple Imputation -> Ordinal Encoding
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    
    # Nominal: Simple Imputation -> CatBoost Encoding
    # CatBoostEncoder needs target for training, so we handle it carefully.
    # For simplicity in this pipeline, we'll use OneHot if cardinality is low, 
    # but plan asked for CatBoostEncoder.
    # Note: CatBoostEncoder is supervised. We need to pass y to fit.
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', CatBoostEncoder())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('nom', nominal_transformer, nominal_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    
    # Full Pipeline including Feature Engineering
    # Note: FunctionTransformer for feature engineering needs to happen BEFORE scaling/encoding
    # if it relies on raw values. 
    # However, sklearn Pipeline doesn't easily allow "insert column then process it" 
    # inside a ColumnTransformer.
    # Strategy: Apply Feature Engineering FIRST on the dataframe, THEN run the pipeline.
    
    print("Applying Feature Engineering...")
    X_train_eng = feature_engineering(X_train)
    X_test_eng = feature_engineering(X_test)
    
    # Update feature lists to include new features
    numeric_features.extend(['TechSynergy', 'SelfDriven', 'SupportIndex', 'BalancedLife', 'AbsenceRisk'])
    
    # Re-define preprocessor with new features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('nom', nominal_transformer, nominal_features)
        ],
        remainder='drop' # Drop original columns that were not processed if any remain
    )
    
    print("Fitting Pipeline...")
    # Fit on Train
    # CatBoostEncoder needs y, so we pass y_train (GradeClass)
    X_train_processed = preprocessor.fit_transform(X_train_eng, y_train)
    
    # Transform Test
    X_test_processed = preprocessor.transform(X_test_eng)
    
    # Get feature names
    # This can be tricky with Transformers, but we'll try to reconstruct
    try:
        num_names = numeric_features
        ord_names = ordinal_features
        nom_names = nominal_features # CatBoostEncoder keeps same names usually
        feature_names = num_names + ord_names + nom_names
    except:
        feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]

    print(f"Processed Train Shape: {X_train_processed.shape}")
    print(f"Processed Test Shape: {X_test_processed.shape}")
    
    # Save Data
    # Convert back to DataFrame for easier saving/loading
    df_train_proc = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    df_train_proc['GradeClass'] = y_train
    df_train_proc['GPA'] = y_train_reg
    
    df_test_proc = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    df_test_proc['GradeClass'] = y_test
    df_test_proc['GPA'] = y_test_reg
    
    df_train_proc.to_parquet("data/processed/train.parquet")
    df_test_proc.to_parquet("data/processed/test.parquet")
    
    # Save Preprocessor
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    print("Processing complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()
