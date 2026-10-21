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
        height: 12px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    .official-header {
        text-align: center;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 12px rgba(0, 135, 81, 0.15);
        margin-bottom: 20px;
    }
    
    .official-title {
        color: #008751;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 24px;
        margin-top: 10px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .official-subtitle {
        color: #2e7d32;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 0px;
    }
    
    .coat-img {
        max-width: 120px;
        height: auto;
        display: block;
        margin: 0 auto;
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
# OFFICIAL NIGERIAN HEADER & COAT OF ARMS
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# Reliable Coat of Arms Image URL with alternative source
coat_of_arms_url = "https://raw.githubusercontent.com/datasets/geo-boundaries/master/data/AGO.png" # Standard fallback
official_coat_url = "https://upload.wikimedia.org/wikipedia/commons/b/bc/Coat_of_arms_of_Nigeria.svg"

st.markdown(f"""
    <div class="official-header">
        <img class="coat-img" src="{official_coat_url}" 
             onerror="this.onerror=null; this.src='https://cdn-icons-png.flaticon.com/512/197/197398.png';" 
             alt="Coat of Arms of Nigeria">
        <div class="official-title">Federal Republic of Nigeria</div>
        <div class="official-subtitle">AI Payroll Anomaly Detection & Fraud Audit System</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# ============================================================
# STEP 1: USER AUTHORIZATION & REGISTRATION (MAIN SCREEN)
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
            conversion_rate = converted_
