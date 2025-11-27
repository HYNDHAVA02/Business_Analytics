import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import joblib
import os

# Page Config
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

@st.cache_resource
def load_model_and_preprocessor():
    model = CatBoostClassifier()
    model.load_model("models/best_catboost_model.cbm")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_model_and_preprocessor()
except Exception as e:
    st.error(f"Error loading model/preprocessor: {e}")
    st.stop()

st.title("🎓 Student Performance Predictor")
st.markdown("### Interactive Prediction & Explainability Dashboard")

# Sidebar Inputs
st.sidebar.header("Student Profile")

def user_input_features():
    # We need to match the input schema expected by the preprocessor (before feature engineering)
    # The preprocessor expects raw columns, but our script 2_process_engineer.py applied 
    # feature engineering BEFORE the pipeline.
    # So we need to replicate that feature engineering here.
    
    gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==1 else "Male")
    parental_edu = st.sidebar.slider("Parental Education Level", 0.0, 4.0, 2.0, 0.5)
    study_time = st.sidebar.slider("Weekly Study Time (Hours)", 0.0, 20.0, 5.0, 0.5)
    absences = st.sidebar.number_input("Absences", 0, 100, 2)
    tutoring = st.sidebar.selectbox("Tutoring", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    extracurricular = st.sidebar.selectbox("Extracurricular Activities", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    internet = st.sidebar.selectbox("Internet Access", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    data = {
        'Gender': gender,
        'ParentalEducation': parental_edu,
        'StudyTimeWeekly': study_time,
        'Absences': absences,
        'Tutoring': tutoring,
        'Extracurricular': extracurricular,
        'InternetAccess': internet
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display Input
st.subheader("Student Data")
st.write(input_df)

# Feature Engineering (Must match 2_process_engineer.py)
input_df['TechSynergy'] = input_df['InternetAccess'] * input_df['StudyTimeWeekly']
input_df['SelfDriven'] = input_df['StudyTimeWeekly'] / (input_df['ParentalEducation'] + 1)
input_df['SupportIndex'] = input_df['ParentalEducation'] + input_df['Tutoring'] + input_df['InternetAccess']
input_df['BalancedLife'] = input_df['Extracurricular'] * input_df['StudyTimeWeekly']
input_df['AbsenceRisk'] = input_df['Absences'] * (1 - input_df['InternetAccess'])

# Prediction
if st.button("Predict Grade"):
    # Preprocess
    try:
        # The preprocessor expects y for fit, but for transform it should be fine.
        # However, CatBoostEncoder might be tricky if it wasn't fitted with handle_unknown='value' or similar.
        # Let's hope it works.
        processed_data = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(processed_data)
        proba = model.predict_proba(processed_data)
        
        # Mapping (assuming 0 is best, 4 is worst)
        grade_map = {0: 'A (Excellent)', 1: 'B (Good)', 2: 'C (Average)', 3: 'D (Poor)', 4: 'F (Fail)'}
        result = grade_map.get(prediction[0][0], "Unknown")
        
        st.success(f"Predicted Grade Class: **{result}**")
        
        # Probability Bar Chart
        proba_df = pd.DataFrame(proba, columns=grade_map.values())
        st.bar_chart(proba_df.T)
        
        # SHAP Explanation
        st.subheader("Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_data)
        
        # Force Plot for the predicted class
        predicted_class_idx = int(prediction[0][0])
        
        # We need feature names. 
        # Try to get them from preprocessor or just use generic if fails
        try:
            # This is hard with ColumnTransformer. 
            # Let's just use the processed_data shape
            feature_names = [f"Feature {i}" for i in range(processed_data.shape[1])]
            # If we can map them manually based on pipeline order:
            # Numeric (7) + Ordinal (1) + Nominal (4) = 12 features?
            # Numeric: StudyTime, Absences, TechSynergy, SelfDriven, SupportIndex, BalancedLife, AbsenceRisk (7)
            # Ordinal: ParentalEducation (1)
            # Nominal: Gender, Tutoring, Extra, Internet (4)
            # Total 12.
            feature_names = [
                'StudyTimeWeekly', 'Absences', 'TechSynergy', 'SelfDriven', 'SupportIndex', 'BalancedLife', 'AbsenceRisk',
                'ParentalEducation',
                'Gender', 'Tutoring', 'Extracurricular', 'InternetAccess'
            ]
        except:
            pass
            
        st_shap(shap.force_plot(explainer.expected_value[predicted_class_idx], shap_values[predicted_class_idx][0], processed_data[0], feature_names=feature_names))
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 400)

# What-If Analysis
st.markdown("---")
st.header("What-If Analysis")
st.info("Adjust the sliders in the sidebar to see how the prediction changes dynamically.")
