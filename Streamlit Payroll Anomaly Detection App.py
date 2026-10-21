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
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="EFCC AI Payroll Anomaly & Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & EFCC Graphics
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f7f6;
    }
    .nigeria-banner {
        background: linear-gradient(90deg, #008751 0%, #008751 33%, #ffffff 33%, #ffffff 66%, #008751 66%, #008751 100%);
        height: 12px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .official-header {
        text-align: center;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.2);
        margin-bottom: 20px;
    }
    .official-title {
        color: #008751;
        font-family: sans-serif;
        font-weight: 900;
        font-size: 24px;
        margin-top: 10px;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .official-subtitle {
        color: #1b5e20;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .efcc-logo {
        max-width: 80px;
        height: auto;
        display: block;
        margin: 0 auto 10px auto;
    }
    .welcome-card {
        background-color: rgba(255, 255, 255, 0.98);
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #1b5e20;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
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
# HEADER & EMBLEM
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Coat_of_arms_of_Nigeria.svg/300px-Coat_of_arms_of_Nigeria.svg.png"

st.markdown(f"""
    <div class="official-header">
        <img class="efcc-logo" src="{logo_url}" alt="Emblem">
        <div class="official-title">Economic and Financial Crimes Commission (EFCC)</div>
        <div class="official-subtitle">AI Payroll Anomaly Detection & Fraud Forensic System</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 1: AUTHORIZATION & REGISTRATION
# ============================================================
st.header("🔑 Step 1: User Authorization & Registration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Authorization Check")
    authorization = st.radio(
        "Do you have official authorization to access this forensic audit system?",
        options=["Select", "YES", "NO"],
        index=0
    )
    if authorization == "NO":
        st.error("⛔ ACCESS DENIED: Proper authorization is required.")
        st.stop()
    elif authorization == "YES":
        st.success("✅ Authorization Confirmed.")

with col2:
    st.subheader("Investigator / Auditor Details")
    full_name = st.text_input("Full Name:", value="Oluwabusayo Juwon AMPITAN")
    address = st.text_input("Official / Residential Address:", value="OG1 Ogba Ogun compound Ejuku Yagba East")
    contact_number = st.text_input("Contact Phone Number:", value="08161999686")

if authorization != "YES":
    st.info("👈 Please confirm your official authorization above to proceed.")
    st.stop()

if not full_name or not address or not contact_number:
    st.warning("⚠️ Please complete all investigator registration details.")
    st.stop()

# ============================================================
# WELCOME MESSAGE WITH USER INFO
# ============================================================
st.markdown(f"""
    <div class="welcome-card">
        <h3 style="color: #008751; margin-top: 0;">🏛️ WELCOME TO EFCC PAYROLL FORENSIC SYSTEM</h3>
        <p><b>Investigator Name:</b> {full_name}</p>
        <p><b>Official Address:</b> {address}</p>
        <p><b>Contact Number:</b> {contact_number}</p>
        <hr style="margin: 10px 0; border-top: 1px solid #ddd;">
        <p style="margin-bottom: 0; color: #2e7d32; font-weight: 600;">Authorization verified successfully. Your secure forensic session is now active.</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# STEP 2: DATA INPUT SELECTION
# ============================================================
st.header("📂 Step 2: Select Payroll Data Source")

data_choice = st.radio(
    "Choose payroll dataset source:",
    ["Generate Sample Payroll Dataset (200 Records)", "Upload External Payroll CSV File"],
    horizontal=True
)

df = None

if "Upload" in data_choice:
    uploaded_file = st.file_uploader("Upload your Payroll CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()
    else:
        st.info("Awaiting CSV upload...")
        st.stop()
else:
    np.random.seed(42)
    num_emp = 200
    df = pd.DataFrame({
        "EmployeeID": [f"EMP{i:03d}" for i in range(1, num_emp + 1)],
        "Name": [f"Employee_{i:03d}" for i in range(1, num_emp + 1)],
        "State": np.random.choice(["Kogi", "Kwara", "Ekiti"], num_emp),
        "LGA": np.random.choice(["Okene", "Kabba", "Lokoja", "Ado"], num_emp),
        "BasicSalary": np.maximum(np.random.normal(80000, 12000, num_emp), 30000).round(2),
        "Allowances": np.maximum(np.random.normal(15000, 4000, num_emp), 0).round(2),
        "Deductions": np.maximum(np.random.normal(5000, 1500, num_emp), 0).round(2)
    })
    df["NetSalary"] = (df["BasicSalary"] + df["Allowances"] - df["Deductions"]).round(2)
    
    # Inject test anomalies
    df.loc[10, "BasicSalary"] = 350000.00
    df.loc[10, "NetSalary"] = df.loc[10, "BasicSalary"] + df.loc[10, "Allowances"] - df.loc[10, "Deductions"]
    df.loc[25, "Allowances"] = 200000.00
    df.loc[25, "NetSalary"] = df.loc[25, "BasicSalary"] + df.loc[25, "Allowances"] - df.loc[25, "Deductions"]
    
    st.success("Generated sample EFCC payroll dataset with embedded test anomalies.")

st.subheader("Preview Loaded Dataset")
st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ============================================================
# STEP 3: DATA CLEANING
# ============================================================
st.header("🧹 Step 3: Automated Data Cleaning & Feature Extraction")

df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(how="all").copy()

for col in df.columns:
    if df[col].dtype == "object":
        clean_val = df[col].astype(str).str.replace(",", "", regex=False).str.replace("₦", "", regex=False).str.strip()
        conv_val = pd.to_numeric(clean_val, errors="coerce")
        if df[col].notna().sum() > 0 and (conv_val.notna().sum() / df[col].notna().sum()) >= 0.80:
            df[col] = conv_val

df = df.drop_duplicates().copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
ignore_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]
feature_cols = [c for c in numeric_cols if c.lower() not in ignore_words]

X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
feature_cols = X.columns.tolist()

if len(feature_cols) == 0:
    st.error("No valid numerical features found for analysis.")
    st.stop()

X = X.fillna(X.median())
X = X[X.columns[X.nunique() > 1]]

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(df))
c2.metric("Features Count", len(X.columns))
c3.metric("Selected Features", ", ".join(X.columns[:3]))

st.divider()

# ============================================================
# STEP 4: AI MODEL TRAINING & ANOMALY DETECTION
# ============================================================
st.header("🤖 Step 4: AI Model Training & Anomaly Detection")

contamination_rate = st.slider("Expected Anomaly Sensitivity Rate (%):", 1, 20, 5) / 100.0

if len(X) < 10:
    st.error("At least 10 records are required to run machine learning detection.")
    st.stop()

X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = IsolationForest(n_estimators=200, contamination=contamination_rate, random_state=42)
model.fit(X_train_scaled)

all_preds = model.predict(scaler.transform(X))
results = df.copy()
results["AnomalyStatus"] = np.where(all_preds == -1, "Flagged for Review", "No Anomaly Flag")
results["AnomalyScore"] = model.decision_function(scaler.transform(X))

flagged_records = results[results["AnomalyStatus"] == "Flagged for Review"].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Analyzed", len(results))
m2.metric("Flagged", len(flagged_records))
m3.metric("Normal", len(results) - len(flagged_records))
m4.metric("Percentage", f"{round(len(flagged_records) / len(results) * 100, 2)}%")

st.subheader("🚩 Records Flagged for Forensic Audit")
if len(flagged_records) > 0:
    st.dataframe(flagged_records, use_container_width=True)
else:
    st.success("No payroll anomalies flagged under current sensitivity setting.")

st.divider()

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================
st.header("📊 Step 5: Payroll Analysis Graphics")

fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = results["AnomalyStatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=["#008751", "#d32f2f"])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom')
    ax.set_title("Classification Summary", fontweight='bold')
    st.pyplot(fig)

with fig_col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(results["AnomalyScore"], bins=20, color="#008751", edgecolor="black")
    ax2.set_title("Anomaly Score Distribution", fontweight='bold')
    st.pyplot(fig2)

st.divider()

# ============================================================
# STEP 6: REPORT GENERATION
# ============================================================
st.header("📄 Step 6: Forensic Audit Report Generation")

summary_df = pd.DataFrame({
    "ReportItem": ["System", "DateTime", "Investigator", "Contact", "TotalRecords", "FlaggedRecords"],
    "Value": ["EFCC AI Payroll System", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), full_name, contact_number, len(results), len(flagged_records)]
})

zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("efcc_summary.csv", summary_df.to_csv(index=False))
    zf.writestr("full_results.csv", results.to_csv(index=False))
    zf.writestr("flagged_records.csv", flagged_records.to_csv(index=False))

st.download_button(
    label="📦 Download Complete EFCC Forensic Audit Package (.ZIP)",
    data=zip_buffer.getvalue(),
    file_name=f"EFCC_Payroll_Audit_{datetime.now().strftime('%Y%m%d')}.zip",
    mime="application/zip"
)

st.divider()

# ============================================================
# STEP 7: Q & A ASSISTANT
# ============================================================
st.header("❓ Step 7: Project Q & A Assistant")
q_input = st.text_input("Ask a question about the system (e.g. 'What algorithm is used?'):")

if q_input:
    ans = "I do not have a prepared answer for that question. Try asking about the algorithm, training, testing, or report."
    if "algorithm" in q_input.lower() or "isolation" in q_input.lower():
        ans = "The system uses Isolation Forest, an unsupervised machine learning algorithm that identifies unusual patterns in numerical data."
    elif "training" in q_input.lower():
        ans = "Training uses approximately 80% of the dataset to learn standard payroll metrics."
    elif "testing" in q_input.lower():
        ans = "Testing uses the remaining 20% to evaluate model performance on unseen data."
    elif "report" in q_input.lower():
        ans = "The report contains summary statistics, full data results, and flagged anomaly tables packaged into a ZIP file."
    st.info(f"**Assistant Response:** {ans}")

st.divider()

# ============================================================
# STEP 8: KUDOS & FEEDBACK
# ============================================================
st.header("🌟 Step 8: User Kudos & Feedback")
with st.form("kudos_form"):
    k_name = st.text_input("Your Name / Department:", value=full_name)
    k_rating = st.select_slider("Rate System Performance:", options=["Poor", "Fair", "Good", "Very Good", "Excellent (Top Kudos)"], value="Excellent (Top Kudos)")
    k_comment = st.text_area("Your Kudos or Comments:", placeholder="Outstanding design and detection capabilities!")
    submitted = st.form_submit_button("Submit Kudos")
    if submitted:
        st.success(f"🎉 Thank you, {k_name}! Your kudos have been logged successfully.")

st.caption("Economic and Financial Crimes Commission (EFCC) • Official Forensic Audit & AI Payroll Portal")
