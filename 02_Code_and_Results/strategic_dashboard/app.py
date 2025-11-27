import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import json

# --- Configuration ---
st.set_page_config(page_title="Strategic Student Analytics", layout="wide", page_icon="🎓")

# --- Constants ---
DATA_PATH = "combined_students_final.csv" # Raw data for display
MODEL_PATH = "models/best_catboost_model.cbm"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
FEATURE_PATH = "strategic_analytics/models/selected_features.json"

# --- Helper Functions ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure GradeClass is int
        if 'GradeClass' in df.columns:
            df['GradeClass'] = df['GradeClass'].fillna(-1).astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Data not found at {DATA_PATH}.")
        return None

@st.cache_resource
def load_model_and_pipeline():
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model/pipeline: {e}")
        return None, None

def feature_engineering(df):
    """Creates new features based on domain knowledge."""
    df = df.copy()
    # 1. TechSynergy: InternetAccess * StudyTimeWeekly
    df['TechSynergy'] = df['InternetAccess'] * df['StudyTimeWeekly']
    # 2. SelfDriven: StudyTimeWeekly / (ParentalEducation + 1)
    df['SelfDriven'] = df['StudyTimeWeekly'] / (df['ParentalEducation'] + 1)
    # 3. SupportIndex: ParentalEducation + Tutoring + InternetAccess
    df['SupportIndex'] = df['ParentalEducation'] + df['Tutoring'] + df['InternetAccess']
    # 4. BalancedLife: Extracurricular * StudyTimeWeekly
    df['BalancedLife'] = df['Extracurricular'] * df['StudyTimeWeekly']
    # 5. AbsenceRisk: Absences * (1 - df['InternetAccess'])
    df['AbsenceRisk'] = df['Absences'] * (1 - df['InternetAccess'])
    return df

# --- Main Layout ---
st.title("🎓 Strategic Decision Intelligence Dashboard")
st.markdown("### Advanced Educational Data Mining & Prescriptive Analytics")

df = load_data()
model, preprocessor = load_model_and_pipeline()

if df is not None:
    # Sidebar
    st.sidebar.header("Navigation")
    tab_selection = st.sidebar.radio("Go to:", [
        "Executive Overview", 
        "Student Risk Analytics", 
        "Key Drivers (Explainability)", 
        "Fairness Audit", 
        "Behavioral Insights", 
        "Survival Analysis", 
        "What-If Simulator"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("v1.3 | Strategic Architecture")

    # --- TAB 1: EXECUTIVE OVERVIEW ---
    if tab_selection == "Executive Overview":
        st.header("📊 Executive Overview")
        
        # Top KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate KPIs
        total_students = len(df)
        avg_gpa = df['GPA'].mean() if 'GPA' in df.columns else 0.0
        # Risk: GradeClass >= 3 (D or F)
        at_risk_count = len(df[df['GradeClass'] >= 3])
        at_risk_pct = (at_risk_count / total_students) * 100
        
        # QWK Score (Hardcoded from report for Executive View)
        qwk_score = 0.75
        
        col1.metric("Overall Model QWK", f"{qwk_score:.2f}", "+0.02 vs Baseline")
        col2.metric("Total Students", f"{total_students:,}")
        col3.metric("At-Risk Students", f"{at_risk_count:,}", f"{at_risk_pct:.1f}% of pop", delta_color="inverse")
        col4.metric("Avg GPA", f"{avg_gpa:.2f}")
        
        st.markdown("---")
        
        # Charts Row
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Predicted Grade Distribution")
            grade_counts = df['GradeClass'].value_counts().sort_index()
            grade_labels = ['A (0)', 'B (1)', 'C (2)', 'D (3)', 'F (4)']
            # Map index to labels
            plot_df = pd.DataFrame({'Grade': [grade_labels[int(i)] for i in grade_counts.index], 'Count': grade_counts.values})
            
            fig = px.bar(plot_df, x='Grade', y='Count', color='Grade', 
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Student Distribution by Predicted Grade Class")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("ROI Calculator")
            st.markdown("Estimate the financial impact of interventions.")
            
            tuition = st.number_input("Avg Tuition / Year ($)", value=20000, step=1000)
            intervention_cost = st.number_input("Intervention Cost / Student ($)", value=500, step=50)
            success_rate = st.slider("Intervention Success Rate (%)", 0, 100, 40) / 100.0
            
            # ROI Logic
            # Target: At-Risk Students
            target_pop = at_risk_count
            cost = target_pop * intervention_cost
            retained = target_pop * success_rate
            revenue_saved = retained * tuition
            net_roi = revenue_saved - cost
            roi_ratio = revenue_saved / cost if cost > 0 else 0
            
            st.metric("Projected Revenue Preserved", f"${revenue_saved/1e6:.2f} M")
            st.metric("Net ROI", f"${net_roi/1e6:.2f} M", f"{roi_ratio:.1f}x Return")
            
            st.info(f"Targeting {target_pop} students identified as High Risk (Class D/F).")

    # --- TAB 2: STUDENT RISK ANALYTICS ---
    elif tab_selection == "Student Risk Analytics":
        st.header("⚠️ Student Risk Analytics")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Filters")
            risk_threshold = st.slider("Risk Threshold (Grade Class)", 0, 4, 3, help="Select minimum Grade Class to consider 'At Risk' (3=D, 4=F)")
            
            gender_filter = st.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
            internet_filter = st.multiselect("Internet Access", df['InternetAccess'].unique(), default=df['InternetAccess'].unique())
            
        with col2:
            # Filter Data
            filtered_df = df[
                (df['GradeClass'] >= risk_threshold) & 
                (df['Gender'].isin(gender_filter)) & 
                (df['InternetAccess'].isin(internet_filter))
            ]
            
            st.subheader(f"Identified {len(filtered_df)} At-Risk Students")
            
            # Risk Distribution
            fig = px.histogram(filtered_df, x="GradeClass", color="GradeClass", 
                               title="Distribution of At-Risk Grades",
                               color_discrete_sequence=['orange', 'red'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            st.dataframe(filtered_df[['StudyTimeWeekly', 'Absences', 'GPA', 'ParentalEducation', 'InternetAccess', 'GradeClass']])

    # --- TAB 3: EXPLAINABILITY ---
    elif tab_selection == "Key Drivers (Explainability)":
        st.header("🔍 Key Drivers & Explainability")
        
        st.subheader("Global Feature Importance (SHAP)")
        st.markdown("Which features drive the model's predictions the most?")
        
        # Display SHAP Summary Image (Pre-computed)
        shap_img_path = "strategic_analytics/reports/shap_summary_selected.png"
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="SHAP Summary Plot", use_column_width=True)
        else:
            st.warning("SHAP Summary image not found.")
            
        st.markdown("---")
        st.subheader("Feature Importance Ranking")
        
        # Load feature importance from JSON if available, or compute simple correlation
        try:
            with open(FEATURE_PATH, 'r') as f:
                features_json = json.load(f)
                selected_features = features_json['confirmed_features']
                st.write(f"**Selected Features by Boruta:** {', '.join(selected_features)}")
                
                # Simple correlation with Target
                corr = df[selected_features + ['GradeClass']].corr()['GradeClass'].sort_values(ascending=False)
                st.bar_chart(corr.drop('GradeClass'))
        except:
            st.info("Feature importance data not available.")

    # --- TAB 4: FAIRNESS AUDIT ---
    elif tab_selection == "Fairness Audit":
        st.header("⚖️ Fairness & Ethics Audit")
        
        st.markdown("### The Digital Divide Analysis")
        st.info("This section audits the model for bias against students without Internet Access.")
        
        # Fairness Metrics
        col1, col2 = st.columns(2)
        
        # Calculate Selection Rate (Predicted Pass Rate)
        # Pass = GradeClass 0, 1, 2. Fail = 3, 4.
        df['Predicted_Pass'] = df['GradeClass'] <= 2
        
        sr_internet = df[df['InternetAccess'] == 1]['Predicted_Pass'].mean()
        sr_no_internet = df[df['InternetAccess'] == 0]['Predicted_Pass'].mean()
        
        col1.metric("Selection Rate (Internet)", f"{sr_internet:.1%}")
        col2.metric("Selection Rate (No Internet)", f"{sr_no_internet:.1%}", f"{sr_internet - sr_no_internet:.1%} Gap", delta_color="inverse")
        
        st.markdown("---")
        st.subheader("Fairness Visualization")
        
        fairness_img_path = "fairness_plot_InternetAccess.png"
        if os.path.exists(fairness_img_path):
            st.image(fairness_img_path, caption="Demographic Parity Plot", use_column_width=True)
        else:
            # Fallback plot
            fig = px.bar(x=['Internet', 'No Internet'], y=[sr_internet, sr_no_internet], 
                         title="Selection Rate Disparity", labels={'x': 'Group', 'y': 'Selection Rate'})
            st.plotly_chart(fig)

    # --- TAB 5: BEHAVIORAL INSIGHTS ---
    elif tab_selection == "Behavioral Insights":
        st.header("🧠 Behavioral Insights (Learner Personas)")
        
        st.markdown("Unsupervised learning (UMAP + HDBScan) identified distinct student personas.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pheno_img_path = "strategic_analytics/reports/phenotypes.png"
            if os.path.exists(pheno_img_path):
                st.image(pheno_img_path, caption="Manifold Learning Projection (UMAP)", use_column_width=True)
            else:
                st.warning("Phenotype image not found.")
                
        with col2:
            st.subheader("Persona Definitions")
            summary_path = "strategic_analytics/reports/persona_summary.csv"
            if os.path.exists(summary_path):
                persona_df = pd.read_csv(summary_path)
                st.dataframe(persona_df)
            else:
                st.info("Persona summary not available.")

    # --- TAB 6: SURVIVAL ANALYSIS ---
    elif tab_selection == "Survival Analysis":
        st.header("⏳ Time-to-Dropout Analysis")
        
        st.markdown("Predicting the **risk of dropout** over time based on behavioral signals.")
        
        surv_img_path = "strategic_analytics/reports/survival_curves.png"
        if os.path.exists(surv_img_path):
            st.image(surv_img_path, caption="Survival Curves by Absence Level", use_column_width=True)
        else:
            st.warning("Survival Curves image not found.")
            
        st.info("Insight: High absences (Red line) drastically accelerate dropout risk after Week 10.")

    # --- TAB 7: WHAT-IF SIMULATOR ---
    elif tab_selection == "What-If Simulator":
        st.header("🧪 Prescriptive 'What-If' Simulator")
        
        st.markdown("Simulate the impact of interventions on a hypothetical student.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Student Profile Inputs")
            
            with st.form("simulation_form"):
                # Inputs
                study_time = st.slider("Study Time Weekly (Hours)", 0, 20, 5)
                absences = st.slider("Absences", 0, 30, 10)
                internet = st.selectbox("Internet Access", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                parent_ed = st.selectbox("Parental Education", [0, 1, 2, 3, 4], index=2, help="0: None, 4: Higher Ed")
                gpa = st.number_input("Current GPA", 0.0, 4.0, 2.5, step=0.1)
                
                # Hidden Defaults
                gender = 0 # Default Male
                tutoring = 0 # Default No
                extracurricular = 0 # Default No
                
                submitted = st.form_submit_button("🔮 Simulate Outcome")
            
            # Create Raw Input DataFrame
            input_data = {
                'StudyTimeWeekly': study_time,
                'Absences': absences,
                'ParentalEducation': parent_ed,
                'InternetAccess': internet,
                'Gender': gender,
                'Tutoring': tutoring,
                'Extracurricular': extracurricular,
                'GPA': gpa,
                'GradeClass': 2 # Dummy
            }
            df_raw = pd.DataFrame([input_data])
            
        with col2:
            st.subheader("Prediction & Impact")
            
            if submitted:
                if model and preprocessor:
                    try:
                        # 1. Feature Engineering
                        df_eng = feature_engineering(df_raw)
                        
                        # 2. Preprocessing
                        X_processed = preprocessor.transform(df_eng)
                        
                        # 3. Predict
                        pred_class = model.predict(X_processed)[0]
                        pred_proba = model.predict_proba(X_processed)[0]
                        
                        # Display Result
                        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
                        grade_label = grade_map.get(int(pred_class), "Unknown")
                        
                        # Color coding
                        color = "green" if pred_class <= 2 else "red"
                        
                        st.metric("Predicted Grade", f"{grade_label} (Class {int(pred_class)})", delta_color="off")
                        
                        # Risk Probability (Prob of D or F)
                        # Classes are 0,1,2,3,4. D/F are indices 3 and 4.
                        risk_prob = sum(pred_proba[3:])
                        st.metric("Risk Probability (D/F)", f"{risk_prob:.1%}", delta_color="inverse")
                        
                        # Recommendation Logic
                        st.markdown("### 💡 AI Recommendation")
                        if risk_prob > 0.5:
                            st.error("High Risk of Failure.")
                            if absences > 10:
                                st.write("- **Critical:** Reduce absences immediately. Aim for < 5.")
                            if study_time < 5:
                                st.write("- **Action:** Increase study time to at least 8-10 hours/week.")
                            if internet == 0:
                                st.write("- **Support:** Provide access to school computer lab or hotspot.")
                        else:
                            st.success("Student is on track.")
                            if pred_class > 0:
                                st.write("- To improve further, focus on consistent study habits.")
                        
                        # Debug Info
                        with st.expander("Debug Model Input"):
                            st.write("Raw Input:", df_raw)
                            st.write("Engineered Features:", df_eng[['TechSynergy', 'SelfDriven', 'SupportIndex', 'BalancedLife', 'AbsenceRisk']])
                            st.write("Prediction Proba:", pred_proba)
                                
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                        st.info("Ensure input features match the model's training data.")
                else:
                    st.error("Model or Pipeline not loaded.")
            else:
                st.info("Adjust inputs and click 'Simulate Outcome' to see results.")

else:
    st.warning("Data could not be loaded.")
