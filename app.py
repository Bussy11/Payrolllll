import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import zipfile
import io
import os
import base64
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Federal Republic of Nigeria & IBS - Forensic Audit Portal",
    page_icon="🇳🇬",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING & MOBILE FLEX HEADER
# ============================================================
st.markdown("""
<style>
.stApp {
    background-color: #f7f9f8;
}
.custom-header-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    width: 100%;
    background-color: #ffffff;
    padding: 10px;
    border-radius: 12px;
    border: 2px solid #008751;
    box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.15);
}
.header-img {
    width: 50px;
    height: auto;
    object-fit: contain;
}
.header-title-box {
    text-align: center;
    flex-grow: 1;
}
.top-title {
    color: #008751;
    font-weight: 900;
    font-size: 13px;
    text-transform: uppercase;
    margin: 2px 0;
}
.top-sub {
    color: #1b5e20;
    font-size: 9px;
    font-weight: 700;
    margin: 0;
}
.stButton>button {
    background-color: #008751 !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}
/* Red News Button Styling */
.news-btn-container a {
    background-color: #d32f2f !important;
    color: white !important;
    padding: 6px 14px;
    border-radius: 6px;
    text-decoration: none;
    font-weight: bold;
    font-size: 12px;
    display: inline-block;
    box-shadow: 0px 2px 6px rgba(211, 47, 47, 0.3);
}
.news-btn-container a:hover {
    background-color: #b71c1c !important;
}
/* Bordered & Bold Green Authorization Header */
.auth-header-box {
    border: 2px solid #008751;
    background-color: #e8f5e9;
    padding: 12px 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.auth-header-text {
    color: #008751;
    font-weight: 900;
    font-size: 1.25rem;
    margin: 0;
}
/* Custom styling for Previous (Green) and Next (Red) compact buttons */
div[data-testid="column"] button[key*="prev_btn"] {
    background-color: #008751 !important;
}
div[data-testid="column"] button[key*="next_btn"] {
    background-color: #d32f2f !important;
}
</style>
""", unsafe_allow_html=True)

# Helper function to convert local images to base64 for mobile flex layout
def get_image_base64(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        ext = filepath.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        return f"data:{mime};base64,{encoded}"
    return ""

# ============================================================
# CHART FUNCTIONS DEFINITION (Responsive Matplotlib Figures)
# ============================================================
def chart_pca(d):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    n = d[~d["flagged"]]
    f = d[d["flagged"]]
    ax.scatter(n["pc1"], n["pc2"], s=12, c="#008751", label="Normal", alpha=0.7)
    ax.scatter(f["pc1"], f["pc2"], s=35, c="#d32f2f", marker="x", label="Flagged")
    ax.set_title("Payroll Records (PCA View)", fontweight='bold', fontsize=10)
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def chart_lga(d):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if "lga" in d.columns:
        c = d[d["flagged"]]["lga"].value_counts()
        if not c.empty:
            ax.barh(c.index, c.values, color="#1b5e20")
            ax.set_title("Flagged Staff per LGA", fontweight='bold', fontsize=10)
            ax.set_xlabel("Flagged Count", fontsize=9)
            ax.set_ylabel("LGA", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No LGA Flagged Records", ha='center', va='center')
            ax.set_title("Flagged Staff per LGA", fontweight='bold', fontsize=10)
    else:
        ax.text(0.5, 0.5, "LGA Column Missing", ha='center', va='center')
    plt.tight_layout()
    return fig

def chart_status_bar(d):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = d["anomalystatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=["#008751", "#d32f2f"])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom', fontsize=9)
    ax.set_title("Anomaly Status Count", fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=10, labelsize=8)
    plt.tight_layout()
    return fig

def chart_status_pie(d):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    pie_counts = d["anomalystatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
    colors = ["#008751", "#d32f2f"]
    ax.pie(
        pie_counts.values, 
        labels=pie_counts.index, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90,
        textprops={'fontsize': 8}
    )
    ax.set_title("Anomaly Share (%)", fontweight='bold', fontsize=10)
    plt.tight_layout()
    return fig

def chart_salary_dist(d):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if "netsalary" in d.columns:
        ax.hist(d["netsalary"], bins=15, color="#008751", edgecolor="white", alpha=0.85)
        ax.set_title("Net Salary Distribution", fontweight='bold', fontsize=10)
        ax.set_xlabel("Net Salary (₦)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NetSalary Column Missing", ha='center', va='center')
    plt.tight_layout()
    return fig

def chart_allowance_scatter(d):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if "allowances" in d.columns and "netsalary" in d.columns:
        n = d[~d["flagged"]]
        f = d[d["flagged"]]
        ax.scatter(n["allowances"], n["netsalary"], color="#008751", label="Normal", alpha=0.6, s=25)
        ax.scatter(f["allowances"], f["netsalary"], color="#d32f2f", label="Flagged", marker="X", s=70)
        ax.set_title("Allowances vs Net Salary Outliers", fontweight='bold', fontsize=10)
        ax.set_xlabel("Allowances (₦)", fontsize=9)
        ax.set_ylabel("Net Salary (₦)", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Required Financial Columns Missing", ha='center', va='center')
    plt.tight_layout()
    return fig

# ============================================================
# TOP HEADER SECTION (Forced Side-by-Side on Mobile via Flexbox)
# ============================================================
coat_arm_path = "coat_of_arms.jpeg" if os.path.exists("coat_of_arms.jpeg") else ("coat_of_arms.jpg" if os.path.exists("coat_of_arms.jpg") else "")
coat_arm_base64 = get_image_base64(coat_arm_path)

school_logo_path = ""
for filename in ["school_logo.jpeg", "school_logo.jpg", "school_logo.png", "logo.jpeg", "logo.jpg", "logo.png"]:
    if os.path.exists(filename):
        school_logo_path = filename
        break
school_logo_base64 = get_image_base64(school_logo_path)

st.markdown(f"""
<div style="background-color: #008751; height: 8px; border-radius: 4px; margin-bottom: 8px;"></div>
<div class="custom-header-container">
    <div>
        {"<img src='" + coat_arm_base64 + "' class='header-img'>" if coat_arm_base64 else "🇳🇬"}
    </div>
    <div class="header-title-box">
        <p class="top-title">International Business School</p>
        <p class="top-sub">AI Payroll Anomaly Detection & Forensic Audit Portal</p>
    </div>
    <div>
        {"<img src='" + school_logo_base64 + "' class='header-img'>" if school_logo_base64 else "🏫"}
    </div>
</div>
<div style="background-color: #008751; height: 8px; border-radius: 4px; margin-top: 8px; margin-bottom: 15px;"></div>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATES
# ============================================================
if "auth_verified" not in st.session_state:
    st.session_state.auth_verified = False
    st.session_state.full_name = ""
    st.session_state.address = ""
    st.session_state.contact = ""

if "step" not in st.session_state:
    st.session_state.step = 1

if "dev_index" not in st.session_state:
    st.session_state.dev_index = 0

# ============================================================
# NAVIGATION MENU + RED NEWS BUTTON (Linked to nairaland.com)
# ============================================================
step_names = {
    1: "1. Authorization",
    2: "2. Data Input",
    3: "3. AI Analysis",
    4: "4. Visualizations",
    5: "5. Q&A & Kudos",
    6: "6. Developers"
}

col_nav, col_news = st.columns([5, 1])

with col_nav:
    selected_step_name = st.selectbox(
        "📱 Quick Navigation", 
        options=list(step_names.values()),
        index=st.session_state.step - 1,
        label_visibility="collapsed"
    )
    selected_step = int(selected_step_name.split(".")[0])

    if selected_step != st.session_state.step:
        if selected_step > 1 and (not st.session_state.auth_verified or not st.session_state.full_name.strip()):
            st.sidebar.warning("⛔ Complete Authorization on Step 1 first.")
        elif selected_step > 2 and "df" not in st.session_state:
            st.sidebar.warning("⛔ Load data on Step 2 first.")
        elif selected_step > 3 and "results" not in st.session_state:
            st.sidebar.warning("⛔ Run AI Analysis on Step 3 first.")
        else:
            st.session_state.step = selected_step
            st.rerun()

with col_news:
    st.markdown("""
        <div class="news-btn-container" style="text-align: right; padding-top: 5px;">
            <a href="https://www.nairaland.com" target="_blank">📰 NEWS</a>
        </div>
    """, unsafe_allow_html=True)

st.markdown(f"**Current Step:** `{step_names[st.session_state.step]}`")
st.divider()

# ============================================================
# PAGE 1: AUTHORIZATION & REGISTRATION
# ============================================================
if st.session_state.step == 1:
    st.markdown("""
        <div class="auth-header-box">
            <h2 class="auth-header-text">🔑 Investigator Authorization & Registration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    auth_choice = st.radio(
        "Do you have official authorization?", 
        ["-- Please Select --", "YES", "NO"], 
        index=0
    )
    
    if auth_choice == "-- Please Select --":
        st.session_state.auth_verified = False
        st.info("ℹ️ Please select your authorization status to proceed.")
    elif auth_choice == "NO":
        st.session_state.auth_verified = False
        st.error("⛔ ACCESS DENIED: Proper authorization is required.")
    else:
        st.session_state.auth_verified = True
        st.success("✅ Authorization Confirmed.")

    st.subheader("Investigator Details")
    st.session_state.full_name = st.text_input("Full Name:", value=st.session_state.full_name)
    st.session_state.address = st.text_input("Official Address:", value=st.session_state.address)
    st.session_state.contact = st.text_input("Contact Number:", value=st.session_state.contact)

    if st.session_state.auth_verified and st.session_state.full_name.strip() != "":
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #008751; margin-top: 15px;">
                <h4 style="color: #008751; margin: 0 0 5px 0;">Official Session Active</h4>
                <p style="margin: 0; color: #1b5e20; font-weight: 600;">
                    Welcome, <b>{st.session_state.full_name}</b>. Your credentials have been logged for this forensic session.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")
    if st.button("Continue to Data Input & Cleaning ➡️", use_container_width=True):
        if not st.session_state.auth_verified:
            st.warning("⚠️ Please complete valid authorization first.")
        elif not st.session_state.full_name.strip():
            st.warning("⚠️ Please enter your full name to proceed.")
        else:
            st.session_state.step = 2
            st.rerun()

# ============================================================
# PAGE 2: DATA INPUT & CLEANING
# ============================================================
if st.session_state.step == 2:
    if not st.session_state.auth_verified or not st.session_state.full_name.strip():
        st.error("⛔ ACCESS LOCKED: Complete official authorization on Step 1 first.")
        if st.button("⬅️ Back to Step 1: Authorization"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("📂 Payroll Data Input & Cleaning")
    
    data_choice = st.radio(
        "Select Dataset Source:",
        ["— Please Select —", "Generate Sample Payroll Dataset (500 Records)", "Upload External Payroll CSV File"],
        index=0,
        key="dataset_radio_choice"
    )
    
    df = None
    if data_choice == "— Please Select —":
        if "df" in st.session_state:
            del st.session_state.df
        st.info("ℹ️ Select a dataset source above to proceed.")
        
    elif "Generate" in data_choice:
        np.random.seed(42)
        num_emp = 500
        df = pd.DataFrame({
            "employeeid": [f"EMP{i:03d}" for i in range(1, num_emp + 1)],
            "name": [f"Employee_{i:03d}" for i in range(1, num_emp + 1)],
            "state": np.random.choice(["Kogi State", "Kwara State", "Ekiti State"], num_emp),
            "lga": np.random.choice(["Okene", "Kabba", "Lokoja", "Ado-Ekiti", "Ijumu", "Yagba East"], num_emp),
            "basicsalary": np.maximum(np.random.normal(85000, 15000, num_emp), 30000).round(2),
            "allowances": np.maximum(np.random.normal(18000, 5000, num_emp), 0).round(2),
            "deductions": np.maximum(np.random.normal(6000, 1800, num_emp), 0).round(2)
        })
        df["netsalary"] = (df["basicsalary"] + df["allowances"] - df["deductions"]).round(2)
        df.loc[10, "basicsalary"] = 450000.00
        df.loc[10, "netsalary"] = df.loc[10, "basicsalary"] + df.loc[10, "allowances"] - df.loc[10, "deductions"]
        df.loc[25, "allowances"] = 300000.00
        df.loc[25, "netsalary"] = df.loc[25, "basicsalary"] + df.loc[25, "allowances"] - df.loc[25, "deductions"]
        df.loc[50, "deductions"] = 0.00
        st.success("Sample Nigerian payroll dataset generated.")
        
    elif "Upload" in data_choice:
        uploaded_file = st.file_uploader("Upload Payroll CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower().strip() for c in df.columns]
            st.success("Loaded uploaded CSV file successfully.")
        else:
            if "df" in st.session_state:
                del st.session_state.df
            st.info("Awaiting CSV upload...")

    if df is not None:
        st.session_state.df = df
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Authorization", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_b2:
        can_proceed = "df" in st.session_state and st.session_state.df is not None
        if st.button("Continue to AI Model Analysis ➡️", use_container_width=True, disabled=not can_proceed):
            if can_proceed:
                st.session_state.step = 3
                st.rerun()

# ============================================================
# PAGE 3: AI MODEL ANALYSIS
# ============================================================
if st.session_state.step == 3:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: Complete authorization on Step 1 first.")
        if st.button("⬅️ Go Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
        st.stop()
    if "df" not in st.session_state:
        st.error("⛔ ACCESS LOCKED: Load or generate data on Step 2 first.")
        if st.button("⬅️ Go Back to Step 2"):
            st.session_state.step = 2
            st.rerun()
        st.stop()

    st.header("🤖 AI Model Training & Anomaly Detection")
    
    df = st.session_state.df
    contamination_rate = st.slider("Expected Anomaly Sensitivity Rate (%):", 1, 20, 7) / 100.0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ignore_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]
    feature_cols = [c for c in numeric_cols if c.lower() not in ignore_words]
    
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0)
    X = X[X.columns[X.nunique() > 1]]
    
    if len(X) < 10:
        st.error("At least 10 records required for analysis.")
    else:
        X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = IsolationForest(n_estimators=200, contamination=contamination_rate, random_state=42)
        model.fit(X_train_scaled)
        
        all_preds = model.predict(scaler.transform(X))
        results = df.copy()
        results["anomalystatus"] = np.where(all_preds == -1, "Flagged for Review", "No Anomaly Flag")
        results["flagged"] = all_preds == -1
        results["anomalyscore"] = model.decision_function(scaler.transform(X))
        
        pca = PCA(n_components=2, random_state=42)
        scaled_full = scaler.transform(X)
        pcs = pca.fit_transform(scaled_full)
        results["pc1"] = pcs[:, 0]
        results["pc2"] = pcs[:, 1]
        
        st.session_state.results = results
        st.session_state.flagged_records = results[results["flagged"]].copy()
        
        m1, m2 = st.columns(2)
        m1.metric("Analyzed", len(results))
        m2.metric("Flagged", len(st.session_state.flagged_records))
        
        m3, m4 = st.columns(2)
        m3.metric("Normal", len(results) - len(st.session_state.flagged_records))
        m4.metric("Percentage", f"{round(len(st.session_state.flagged_records) / len(results) * 100, 2)}%")
        
        st.subheader("🚩 Flagged Payroll Anomalies")
        st.dataframe(st.session_state.flagged_records, use_container_width=True)

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Data Input", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_b2:
        if st.button("Continue to Visualizations ➡️", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ============================================================
# PAGE 4: VISUALIZATIONS & REPORTS
# ============================================================
if st.session_state.step == 4:
    if not st.session_state.auth_verified or "results" not in st.session_state:
        st.error("⛔ ACCESS LOCKED: Please complete earlier steps first.")
        st.stop()

    st.header("📊 Advanced Payroll Analytics")
    results = st.session_state.results
    flagged_records = st.session_state.flagged_records
    
    st.subheader("1. Dimensional Reduction & Regional Risk")
    st.pyplot(chart_pca(results), use_container_width=True)
    st.pyplot(chart_lga(results), use_container_width=True)

    st.divider()

    st.subheader("2. Anomaly Counts & Proportion Share")
    st.pyplot(chart_status_bar(results), use_container_width=True)
    st.pyplot(chart_status_pie(results), use_container_width=True)

    st.divider()

    st.subheader("3. Depth & Financial Correlation Analysis")
    st.pyplot(chart_salary_dist(results), use_container_width=True)
    st.pyplot(chart_allowance_scatter(results), use_container_width=True)
        
    st.divider()
    st.subheader("📦 Download Audit Package")
    summary_df = pd.DataFrame({
        "ReportItem": ["System", "DateTime", "Investigator", "Contact", "TotalRecords", "FlaggedRecords"],
        "Value": ["Federal Republic of Nigeria Payroll System", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st.session
