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
    page_title="Federal Republic of Nigeria - AI Payroll Forensic Portal",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f9f8;
    }
    .nigeria-banner {
        background: linear-gradient(90deg, #008751 0%, #008751 33%, #ffffff 33%, #ffffff 66%, #008751 66%, #008751 100%);
        height: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .top-title {
        color: #008751;
        font-weight: 900;
        font-size: 24px;
        text-transform: uppercase;
        text-align: center;
        margin: 0;
    }
    .top-sub {
        color: #1b5e20;
        font-size: 14px;
        font-weight: 700;
        text-align: center;
        margin-top: 5px;
    }
    .stButton>button {
        background-color: #008751 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# TOP HEADER WITH NATIVE IMAGES
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

col_logo1, col_title, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Coat_of_arms_of_Nigeria.svg/200px-Coat_of_arms_of_Nigeria.svg.png", width=80)

with col_title:
    st.markdown('<p class="top-title">Federal Republic of Nigeria</p>', unsafe_allow_html=True)
    st.markdown('<p class="top-sub">AI Payroll Anomaly Detection & EFCC Forensic Audit Portal</p>', unsafe_allow_html=True)

with col_logo2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Economic_and_Financial_Crimes_Commission_logo.png/200px-Economic_and_Financial_Crimes_Commission_logo.png", width=80)

st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📌 Navigation Menu")
page = st.sidebar.radio(
    "Select Page:",
    [
        "1. Authorization & Registration",
        "2. Data Input & Cleaning",
        "3. AI Model Analysis",
        "4. Visualizations & Reports",
        "5. Q&A Assistant & Kudos"
    ]
)

if "auth_verified" not in st.session_state:
    st.session_state.auth_verified = True
    st.session_state.full_name = "Oluwabusayo Juwon AMPITAN"
    st.session_state.address = "OG1 Ogba Ogun compound Ejuku Yagba East"
    st.session_state.contact = "08161999686"

# ============================================================
# PAGE 1: AUTHORIZATION & REGISTRATION
# ============================================================
if page == "1. Authorization & Registration":
    st.header("🔑 Investigator Authorization & Registration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Authorization Check")
        auth_choice = st.radio("Do you have official authorization?", ["YES", "NO"], index=0)
        
        if auth_choice == "NO":
            st.session_state.auth_verified = False
            st.error("⛔ ACCESS DENIED: Proper authorization is required.")
        else:
            st.session_state.auth_verified = True
            st.success("✅ Authorization Confirmed.")

    with col2:
        st.subheader("Investigator Details")
        st.session_state.full_name = st.text_input("Full Name:", value=st.session_state.full_name)
        st.session_state.address = st.text_input("Official Address:", value=st.session_state.address)
        st.session_state.contact = st.text_input("Contact Number:", value=st.session_state.contact)

    if st.session_state.auth_verified:
        st.success("🏛️ Session Active: " + st.session_state.full_name)

# ============================================================
# PAGE 2: DATA INPUT & CLEANING
# ============================================================
elif page == "2. Data Input & Cleaning":
    st.header("📂 Payroll Data Input & Cleaning")
    
    if not st.session_state.auth_verified:
        st.warning("⚠️ Please complete Authorization on Page 1 first.")
        st.stop()
        
    data_choice = st.radio(
        "Select Dataset Source:",
        ["Generate Sample Payroll Dataset (200 Records)", "Upload External Payroll CSV File"],
        horizontal=True
    )
    
    df = None
    if "Upload" in data_choice:
        uploaded_file = st.file_uploader("Upload Payroll CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded CSV file.")
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
        df.loc[10, "BasicSalary"] = 350000.00
        df.loc[10, "NetSalary"] = df.loc[10, "BasicSalary"] + df.loc[10, "Allowances"] - df.loc[10, "Deductions"]
        df.loc[25, "Allowances"] = 200000.00
        df.loc[25, "NetSalary"] = df.loc[25, "BasicSalary"] + df.loc[25, "Allowances"] - df.loc[25, "Deductions"]
        st.success("Sample Nigerian payroll dataset generated with test anomalies.")

    st.session_state.df = df
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# PAGE 3: AI MODEL ANALYSIS
# ============================================================
elif page == "3. AI Model Analysis":
    st.header("🤖 AI Model Training & Anomaly Detection")
    
    if not st.session_state.auth_verified or "df" not in st.session_state:
        st.warning("⚠️ Please complete previous pages first.")
        st.stop()
        
    df = st.session_state.df
    contamination_rate = st.slider("Expected Anomaly Sensitivity Rate (%):", 1, 20, 5) / 100.0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ignore_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]
    feature_cols = [c for c in numeric_cols if c.lower() not in ignore_words]
    
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0)
    X = X[X.columns[X.nunique() > 1]]
    
    if len(X) < 10:
        st.error("At least 10 records required for analysis.")
        st.stop()
        
    X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = IsolationForest(n_estimators=200, contamination=contamination_rate, random_state=42)
    model.fit(X_train_scaled)
    
    all_preds = model.predict(scaler.transform(X))
    results = df.copy()
    results["AnomalyStatus"] = np.where(all_preds == -1, "Flagged for Review", "No Anomaly Flag")
    results["AnomalyScore"] = model.decision_function(scaler.transform(X))
    
    st.session_state.results = results
    st.session_state.flagged_records = results[results["AnomalyStatus"] == "Flagged for Review"].copy()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analyzed", len(results))
    m2.metric("Flagged", len(st.session_state.flagged_records))
    m3.metric("Normal", len(results) - len(st.session_state.flagged_records))
    m4.metric("Percentage", f"{round(len(st.session_state.flagged_records) / len(results) * 100, 2)}%")
    
    st.subheader("🚩 Flagged Payroll Anomalies")
    st.dataframe(st.session_state.flagged_records, use_container_width=True)

# ============================================================
# PAGE 4: VISUALIZATIONS & REPORTS
# ============================================================
elif page == "4. Visualizations & Reports":
    st.header("📊 Payroll Analytics & Report Export")
    
    if "results" not in st.session_state:
        st.warning("⚠️ Please run AI Model Analysis on Page 3 first.")
        st.stop()
        
    results = st.session_state.results
    flagged_records = st.session_state.flagged_records
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = results["AnomalyStatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
        bars = ax.bar(counts.index, counts.values, color=["#008751", "#d32f2f"])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom')
        ax.set_title("Classification Summary", fontweight='bold')
        st.pyplot(fig)
        
    with col_v2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(results["AnomalyScore"], bins=20, color="#008751", edgecolor="black")
        ax2.set_title("Anomaly Score Distribution", fontweight='bold')
        st.pyplot(fig2)
        
    st.divider()
    st.subheader("📦 Download Audit Package")
    summary_df = pd.DataFrame({
        "ReportItem": ["System", "DateTime", "Investigator", "Contact", "TotalRecords", "FlaggedRecords"],
        "Value": ["Federal Republic of Nigeria Payroll System", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st.session_state.full_name, st.session_state.contact, len(results), len(flagged_records)]
    })
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nigeria_payroll_summary.csv", summary_df.to_csv(index=False))
        zf.writestr("full_results.csv", results.to_csv(index=False))
        zf.writestr("flagged_records.csv", flagged_records.to_csv(index=False))
        
    st.download_button(
        label="📥 Download Official Audit ZIP Package",
        data=zip_buffer.getvalue(),
        file_name=f"Nigeria_Payroll_Audit_{datetime.now().strftime('%Y%m%d')}.zip",
        mime="application/zip"
    )

# ============================================================
# PAGE 5: Q&A ASSISTANT & KUDOS
# ============================================================
elif page == "5. Q&A Assistant & Kudos":
    st.header("❓ Project Q & A Assistant & User Kudos")
    
    q_input = st.text_input("Ask a question about the system:")
    if q_input:
        ans = "I do not have a prepared answer for that question."
        if "algorithm" in q_input.lower() or "isolation" in q_input.lower():
            ans = "The system uses Isolation Forest, an unsupervised machine learning algorithm."
        elif "training" in q_input.lower():
            ans = "Training uses 80% of the dataset to learn standard payroll patterns."
        elif "report" in q_input.lower():
            ans = "The report contains summary statistics, full data results, and flagged anomaly tables packaged into a ZIP file."
        st.info("Assistant Response: " + ans)
        
    st.divider()
    st.subheader("🌟 Leave Your Kudos")
    with st.form("kudos_form"):
        k_name = st.text_input("Your Name or Department:", value=st.session_state.full_name)
        k_rating = st.select_slider("Rate System Performance:", options=["Poor", "Fair", "Good", "Very Good", "Excellent"], value="Excellent")
        k_comment = st.text_area("Your Kudos or Comments:", value="Great application!")
        submitted = st.form_submit_button("Submit Kudos")
        if submitted:
            st.success("🎉 Thank you! Your kudos have been logged successfully.")

st.sidebar.caption("Federal Republic of Nigeria • Official Portal")
