import pandas as pd
import numpy as np

# --------------------------
# Load all three datasets
# --------------------------
df1 = pd.read_csv("student_data.csv") 
df2 = pd.read_csv("Student_performance_data.csv") 
df3 = pd.read_csv("StudentPerformanceFactors.csv")  

# ==========================================================
# 🧾 Dataset 1: student_data.csv
# ==========================================================
df1['Gender'] = df1['sex'].map({'F': 1, 'M': 0})
df1['ParentalEducation'] = (df1['Medu'] + df1['Fedu']) / 2
df1['StudyTimeWeekly'] = df1['studytime'].map({1: 1.5, 2: 3.5, 3: 7.5, 4: 12.0})
df1['Tutoring'] = df1['schoolsup'].map({'yes': 1, 'no': 0})
df1['Extracurricular'] = df1['activities'].map({'yes': 1, 'no': 0})
df1['InternetAccess'] = df1['internet'].map({'yes': 1, 'no': 0})
df1['GPA'] = df1['G3'] / 5

# Safe GradeClass conversion
df1['GradeClass'] = pd.cut(
    df1['GPA'],
    bins=[-1, 2.0, 2.5, 3.0, 3.5, 4.5],
    labels=[4, 3, 2, 1, 0]
)
df1['GradeClass'] = df1['GradeClass'].fillna(4).astype(int)
df1['DatasetType'] = 1

df1_aligned = df1[['Gender', 'ParentalEducation', 'StudyTimeWeekly', 'absences',
                   'Tutoring', 'Extracurricular', 'InternetAccess',
                   'GPA', 'GradeClass', 'DatasetType']].rename(columns={'absences': 'Absences'})

# ==========================================================
# 🧾 Dataset 2: Student_performance_data.csv
# ==========================================================
df2['Gender'] = df2['Gender'].map({1: 1, 0: 0})
df2['ParentalEducation'] = df2['ParentalEducation']
df2['StudyTimeWeekly'] = df2['StudyTimeWeekly']
df2['Tutoring'] = df2['Tutoring']
df2['Extracurricular'] = df2['Extracurricular']
df2['InternetAccess'] = np.nan  
df2['GPA'] = df2['GPA']
df2['GradeClass'] = df2['GradeClass']
df2['DatasetType'] = 2

df2_aligned = df2[['Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
                   'Tutoring', 'Extracurricular', 'InternetAccess',
                   'GPA', 'GradeClass', 'DatasetType']]

# ==========================================================
# 🧾 Dataset 3: StudentPerformanceFactors.csv
# ==========================================================
df3['Gender'] = df3['Gender'].map({'Female': 1, 'Male': 0})
df3['ParentalEducation'] = df3['Parental_Education_Level'].map({
    'None': 0, 'High School': 1, 'College': 2, 'Postgraduate': 3
}).fillna(0)
df3['StudyTimeWeekly'] = df3['Hours_Studied']
df3['Absences'] = 100 - df3['Attendance']
df3['Tutoring'] = df3['Tutoring_Sessions'].apply(lambda x: 1 if x > 0 else 0)
df3['Extracurricular'] = df3['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})
df3['InternetAccess'] = df3['Internet_Access'].map({'Yes': 1, 'No': 0})
df3['GPA'] = df3['Exam_Score'] / 25
df3.loc[df3['GPA'] > 4.0, 'GPA'] = 4.0
df3.loc[df3['GPA'] < 0, 'GPA'] = 0.0

# Safe GradeClass conversion
df3['GradeClass'] = pd.cut(
    df3['GPA'],
    bins=[-1, 2.0, 2.5, 3.0, 3.5, 4.5],
    labels=[4, 3, 2, 1, 0]
)
df3['GradeClass'] = df3['GradeClass'].fillna(4).astype(int)
df3['DatasetType'] = 3

df3_aligned = df3[['Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
                   'Tutoring', 'Extracurricular', 'InternetAccess',
                   'GPA', 'GradeClass', 'DatasetType']]

# ==========================================================
# 🧩 Combine All Datasets
# ==========================================================
combined = pd.concat([df1_aligned, df2_aligned, df3_aligned], ignore_index=True)

# ==========================================================
# 🧼 Handle Missing Values
# ==========================================================
combined['InternetAccess'].fillna(0, inplace=True)
combined.fillna(0, inplace=True)

# ==========================================================
# ✅ Final Verification
# ==========================================================
print("✅ Combined dataset created successfully!")
print("Shape:", combined.shape)
print("\nPreview:")
print(combined.head())

# Save clean combined dataset
combined.to_csv("combined_students_final.csv", index=False)
print("\n📁 Saved as 'combined_students_final.csv'")
