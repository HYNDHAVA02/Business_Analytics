import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_PATH = "strategic_analytics/data/phenotyped_data.parquet"
OUTPUT_REPORT = "strategic_analytics/reports/drift_report.html"
REPORT_DIR = "strategic_analytics/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI) for a single feature."""
    
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    
    if expected.nunique() > buckets:
        breakpoints = np.percentile(expected, breakpoints)
    else:
        # For categorical/low cardinality, use unique values
        breakpoints = np.unique(expected)
        # Add a small epsilon to the last breakpoint to include max
        breakpoints = np.append(breakpoints, breakpoints[-1] + 1e-10)

    # Handle unique values being too few for percentiles
    if len(np.unique(breakpoints)) < len(breakpoints):
        breakpoints = np.unique(breakpoints)

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

def run_model_monitoring():
    print("--- Module 8: Model Monitoring & Lifecycle Management (Custom PSI) ---")
    
    # 1. Load Data
    try:
        df = pd.read_parquet(INPUT_PATH)
        print(f"Loaded data from {INPUT_PATH}")
    except FileNotFoundError:
        print("Input data not found.")
        return

    # 2. Simulate Drift
    mid_point = len(df) // 2
    reference_data = df.iloc[:mid_point].copy()
    current_data = df.iloc[mid_point:].copy()
    
    print("Simulating Data Drift in 'StudyTimeWeekly'...")
    # Reduce StudyTime in current data to simulate "Senioritis"
    current_data['StudyTimeWeekly'] = current_data['StudyTimeWeekly'] * 0.7 # Stronger drift
    
    # 3. Calculate PSI
    print("Calculating Population Stability Index (PSI)...")
    
    numeric_cols = ['StudyTimeWeekly', 'Absences', 'GPA', 'ParentalEducation']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    psi_results = {}
    for col in numeric_cols:
        psi = calculate_psi(reference_data[col], current_data[col])
        psi_results[col] = psi
        print(f"PSI for {col}: {psi:.4f}")
        
    # 4. Generate Report
    html_content = """
    <html>
    <head><title>Model Drift Report (PSI)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 50%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .alert { color: red; font-weight: bold; }
        .stable { color: green; font-weight: bold; }
    </style>
    </head>
    <body>
    <h1>Model Drift Report (Population Stability Index)</h1>
    <table>
        <tr><th>Feature</th><th>PSI Value</th><th>Status</th></tr>
    """
    
    drift_detected = False
    for col, psi in psi_results.items():
        status = "Stable"
        css_class = "stable"
        if psi > 0.2:
            status = "Drift Detected (Critical)"
            css_class = "alert"
            drift_detected = True
        elif psi > 0.1:
            status = "Warning"
            css_class = "alert"
            
        html_content += f"<tr><td>{col}</td><td>{psi:.4f}</td><td class='{css_class}'>{status}</td></tr>"
        
    html_content += """
    </table>
    <br>
    """
    
    if drift_detected:
        html_content += "<h2 class='alert'>ALERT: Significant Model Drift Detected! Retraining Recommended.</h2>"
    else:
        html_content += "<h2 class='stable'>System Status: Stable.</h2>"
        
    html_content += "</body></html>"
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write(html_content)
        
    print(f"Saved Drift Report to {OUTPUT_REPORT}")
    print("Module 8 Complete.")

if __name__ == "__main__":
    run_model_monitoring()
