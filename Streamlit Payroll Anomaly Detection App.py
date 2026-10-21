import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import zipfile
import io
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Payroll Anomaly Detection System - Federal Republic of Nigeria",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nigerian Official Theme and Visual Enhancements
st.markdown("""
    <style>
    /* Main Background Accent */
    .stApp {
        background-color: #f8faf9;
    }
    
    /* Header Banner Styling */
    .nigeria-banner {
        background: linear-gradient(90deg, #008751 0%, #008751 33%, #ffffff 33%, #ffffff 66%, #008751 66%, #008751 100%);
        height: 10px;
        border-radius: 4px;
        margin-bottom: 25px;
    }
    
    .official-header {
        text-align: center;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 12px rgba(0, 135, 81, 0.1);
        margin-bottom: 25px;
    }
    
    .official-title {
        color: #008751;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 26px;
        margin-top: 10px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .official-subtitle {
        color: #2e7d32;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 0px;
    }
    
    /* Card Container Styles */
    .card-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #008751;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Primary Buttons */
    .stButton>button {
        background-color: #008751 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# OFFICIAL NIGERIAN HEADER & EMBLEM DISPLAY
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    st.markdown("""
        <div class="official-header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Coat_of_arms_of_Nigeria.svg/300px-Coat_of_arms_of_Nigeria.svg.png" width="90" alt="Coat of Arms of Nigeria">
            <div class="official-title">Federal Republic of Nigeria</div>
            <div class="official-subtitle">AI Payroll Anomaly Detection & Fraud Audit System</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 1: USER AUTHORIZATION & REGISTRATION (ON MAIN SCREEN)
# ============================================================
st.header("🔑 Step 1: User Authorization & Registration")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Authorization Check")
        authorization = st.radio(
            "Do you have official authorization to access this audit system?",
            options=["Select", "YES", "NO"],
            index=0
        )
        
        if authorization == "NO":
            st.error("⛔ ACCESS DENIED: Please obtain proper authorization before proceeding with the audit.")
            st.stop()
        elif authorization == "YES":
            st.success("✅ Authorization Confirmed.")

    with col2:
        st.subheader("Auditor / User Information")
        full_name = st.text_input("Full Name:", value="Oluwabusayo Juwon AMPITAN")
        address = st.text_input("Official / Residential Address:", value="OG1 Ogba Ogun compound Ejuku Yagba East")
        contact_number = st.text_input("Contact Phone Number:", value="08161999686")

if authorization != "YES":
    st.info("👈 Please confirm authorization above to proceed.")
    st.stop()

if not full_name or not address or not contact_number:
    st.warning("⚠️ Please ensure all registration fields above are completed.")
    st.stop()

st.divider()

# ============================================================
# STEP 2: DATA INPUT SELECTION
# ============================================================
st.header("📂 Step 2: Select Payroll Data Source")

data_choice = st.radio(
    "Choose how to load payroll dataset:",
    ["Generate Sample Payroll Data (200 Records)", "Upload Payroll CSV File"],
    horizontal=True
)

df = None

if "Upload" in data_choice:
    uploaded_file = st.file_uploader("Upload your Payroll CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()
    else:
        st.info("Awaiting CSV upload...")
        st.stop()
else:
    # Generate Sample Payroll Data
    np.random.seed(42)
    number_of_employees = 200
    
    employee_ids = [f"EMP{i:03d}" for i in range(1, number_of_employees + 1)]
    employee_names = [f"Employee_{i:03d}" for i in range(1, number_of_employees + 1)]
    states = np.random.choice(["Kogi", "Kwara", "Ekiti"], number_of_employees)
    lgas = np.random.choice(["Okene", "Kabba", "Lokoja", "Ado"], number_of_employees)
    
    basic_salary = np.maximum(np.random.normal(80000, 12000, number_of_employees), 30000)
    allowances = np.maximum(np.random.normal(15000, 4000, number_of_employees), 0)
    deductions = np.maximum(np.random.normal(5000, 1500, number_of_employees), 0)
    
    net_salary = basic_salary + allowances - deductions
    
    df = pd.DataFrame({
        "EmployeeID": employee_ids,
        "Name": employee_names,
        "State": states,
        "LGA": lgas,
        "BasicSalary": basic_salary.round(2),
        "Allowances": allowances.round(2),
        "Deductions": deductions.round(2),
        "NetSalary": net_salary.round(2)
    })
    
    # Inject test anomalies
    df.loc[10, "BasicSalary"] = 350000.00
    df.loc[10, "NetSalary"] = df.loc[10, "BasicSalary"] + df.loc[10, "Allowances"] - df.loc[10, "Deductions"]
    
    df.loc[25, "Allowances"] = 200000.00
    df.loc[25, "NetSalary"] = df.loc[25, "BasicSalary"] + df.loc[25, "Allowances"] - df.loc[25, "Deductions"]
    
    st.success("Generated standard sample payroll dataset with test anomaly records.")

st.subheader("Preview Loaded Dataset")
st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ============================================================
# STEP 3: DATA CLEANING & PREPARATION
# ============================================================
st.header("🧹 Step 3: Automated Data Cleaning & Feature Extraction")

# Cleaning operations
df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(how="all").copy()

# Numerical conversion
for column in df.columns:
    if df[column].dtype == "object":
        cleaned_values = (
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₦", "", regex=False)
            .str.strip()
        )
        converted_values = pd.to_numeric(cleaned_values, errors="coerce")
        non_empty_count = df[column].notna().sum()
        if non_empty_count > 0:
            conversion_rate = converted_values.notna().sum() / non_empty_count
            if conversion_rate >= 0.80:
                df[column] = converted_values

df = df.drop_duplicates().copy()

# Feature Selection
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
identifier_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]

feature_columns = [
    col for col in numeric_columns
    if col.lower() not in identifier_words
]

X = df[feature_columns].copy()
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
feature_columns = X.columns.tolist()

if len(feature_columns) == 0:
    st.error("No valid numerical features found for machine learning detection.")
    st.stop()

X = X.fillna(X.median())
varying_columns = X.columns[X.nunique() > 1].tolist()
X = X[varying_columns]
feature_columns = X.columns.tolist()

col_clean1, col_clean2, col_clean3 = st.columns(3)
col_clean1.metric("Total Records", len(df))
col_clean2.metric("Clean Features Count", len(feature_columns))
col_clean3.metric("Selected Features", ", ".join(feature_columns[:3]))

st.divider()

# ============================================================
# STEP 4: AI ANOMALY DETECTION (ISOLATION FOREST)
# ============================================================
st.header("🤖 Step 4: AI Model Training & Anomaly Detection")

contamination_rate = st.slider(
    "Expected Anomaly Sensitivity Rate (%):",
    min_value=1,
    max_value=20,
    value=5
) / 100.0

if len(X) < 10:
    st.error("At least 10 records are required to train the machine learning model.")
    st.stop()

X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = IsolationForest(
    n_estimators=200,
    contamination=contamination_rate,
    random_state=42
)
model.fit(X_train_scaled)

test_scores = model.decision_function(X_test_scaled)
test_predictions = model.predict(X_test_scaled)

all_scaled = scaler.transform(X)
all_predictions = model.predict(all_scaled)

results = df.copy()
results["AnomalyStatus"] = np.where(all_predictions == -1, "Flagged for Review", "No Anomaly Flag")
results["AnomalyScore"] = model.decision_function(all_scaled)

flagged_records = results[results["AnomalyStatus"] == "Flagged for Review"].copy()

# Summary Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Analyzed Records", len(results))
m2.metric("Flagged Anomalies", len(flagged_records))
m3.metric("Normal Records", len(results) - len(flagged_records))
m4.metric("Flagged Percentage", f"{round(len(flagged_records) / len(results) * 100, 2)}%")

st.subheader("🚩 Records Flagged for Review")
if len(flagged_records) > 0:
    st.dataframe(flagged_records, use_container_width=True)
else:
    st.success("No anomaly flags detected under the current sensitivity rate.")

st.divider()

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================
st.header("📊 Step 5: Payroll Analysis Analytics")

col_fig1, col_fig2 = st.columns(2)

with col_fig1:
    fig, ax = plt.subplots(figsize=(6, 4))
    status_counts = results["AnomalyStatus"].value_counts().reindex(
        ["No Anomaly Flag", "Flagged for Review"], fill_value=0
    )
    bars = ax.bar(status_counts.index, status_counts.values, color=["#008751", "#d32f2f"])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
        
    ax.set_title("Payroll Classification Summary", fontsize=12, fontweight='bold')
    ax.set_ylabel("Record Count")
    plt.tight_layout()
    st.pyplot(fig)

with col_fig2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(results["AnomalyScore"], bins=20, color="#008751", edgecolor="black")
    ax2.set_title("Anomaly Score Distribution", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Anomaly Score (Lower = More Anomalous)")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()

# ============================================================
# STEP 6: REPORT GENERATION & DOWNLOAD
# ============================================================
st.header("📄 Step 6: Audit Report Generation")

report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

summary_df = pd.DataFrame({
    "ReportItem": [
        "SystemName", "ReportDateTime", "AuditorName", "AuditorContact",
        "TotalRecords", "FlaggedRecords", "PercentageFlagged", "Algorithm"
    ],
    "Value": [
        "AI Payroll Anomaly Detection System (Nigeria)", report_time,
        full_name, contact_number, len(results), len(flagged_records),
        f"{round(len(flagged_records) / len(results) * 100, 2)}%", "Isolation Forest"
    ]
})

zip_buffer = io.BytesIO()

with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
    zip_file.writestr("payroll_summary.csv", summary_df.to_csv(index=False))
    zip_file.writestr("full_payroll_results.csv", results.to_csv(index=False))
    zip_file.writestr("flagged_payroll_anomalies.csv", flagged_records.to_csv(index=False))

st.download_button(
    label="📦 Download Complete Official Audit Package (.ZIP)",
    data=zip_buffer.getvalue(),
    file_name=f"Payroll_Audit_Report_{datetime.now().strftime('%Y%m%d')}.zip",
    mime="application/zip"
)

st.caption("Federal Republic of Nigeria • Official Audit & AI Payroll Fraud Inspection Portal")
