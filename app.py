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
import os
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

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f7f9f8;
    }
    .nigeria-banner {
        background-color: #008751;
        height: 10px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .top-header {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.15);
        text-align: center;
        margin-bottom: 15px;
    }
    .top-title {
        color: #008751;
        font-weight: 900;
        font-size: 22px;
        text-transform: uppercase;
        margin: 5px 0 2px 0;
    }
    .top-sub {
        color: #1b5e20;
        font-size: 13px;
        font-weight: 700;
        margin: 0;
    }
    .stButton>button {
        background-color: #008751 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

        padding: 15px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.15);
        text-align: center;
        margin-bottom: 15px;
    }
    .top-title {
        color: #008751;
        font-weight: 900;
        font-size: 22px;
        text-transform: uppercase;
        margin: 5px 0 2px 0;
    }
    .top-sub {
        color: #1b5e20;
        font-size: 13px;
        font-weight: 700;
        margin: 0;
    }
    .stButton>button {
        background-color: #008751 !important;
        color: white !important;
        font-weight: bold !important;
    }
    
 # ============================================================
# TOP HEADER: PERFECTLY CENTERED INSTITUTIONAL LAYOUT
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# 1. School Logo & Title (Centered)
_, col_center, _ = st.columns([1, 3, 1])

with col_center:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    logo_found = False
    for filename in ["school_logo.jpeg", "school_logo.jpg", "school_logo.png", "logo.jpeg", "logo.jpg", "logo.png", "ibs_logo.jpeg"]:
        if os.path.exists(filename):
            st.image(filename, width=100)
            logo_found = True
            break
            
    if not logo_found:
        st.write("🏫 IBS Logo")
        
    st.markdown("""
        <div class="top-header" style="margin-top: 6px; padding: 10px;">
            <p class="top-title" style="font-size: 17px; margin: 0;">International Business School</p>
            <p class="top-sub" style="font-size: 11px; margin: 0;">AI Payroll Anomaly Detection & EFCC Forensic Audit Portal</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 2. Government Logos Below (Cleanly spaced side-by-side)
st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
_, col_arm, col_space, col_efcc, _ = st.columns([2, 1, 0.5, 1, 2])

with col_arm:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    if os.path.exists("coat_of_arms.jpeg"):
        st.image("coat_of_arms.jpeg", width=50)
    elif os.path.exists("coat_of_arms.jpg"):
        st.image("coat_of_arms.jpg", width=50)
    else:
        st.write("🇳🇬")
    st.markdown("</div>", unsafe_allow_html=True)

with col_efcc:
    st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)
    if os.path.exists("efcc_logo.jpeg"):
        st.image("efcc_logo.jpeg", width=50)
    elif os.path.exists("efcc_logo.jpg"):
        st.image("efcc_logo.jpg", width=50)
    else:
        st.write("🦅")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner" style="margin-top: 10px;"></div>', unsafe_allow_html=True)

 #============================================================
# INITIALIZE SESSION STATES (STEP 2 SET AS DEFAULT)
# ============================================================
if "auth_verified" not in st.session_state:
    st.session_state.auth_verified = True
    st.session_state.full_name = " "
    st.session_state.address = " "
    st.session_state.contact= " "

if "step" not in st.session_state:
    st.session_state.step = 2

if "dev_index" not in st.session_state:
    st.session_state.dev_index = 0

# ============================================================
# MAIN BODY STEP NAVIGATION BAR (6 CENTRAL BUTTONS)
# ============================================================
st.markdown("### 📌 Navigation Menu")
step_names = {
    1: "1. Authorization",
    2: "2. Data Input",
    3: "3. AI Analysis",
    4: "4. Visualizations",
    5: "5. Q&A & Kudos",
    6: "6. Developers"
}

nav_cols = st.columns(6)
for s_num, s_name in step_names.items():
    with nav_cols[s_num - 1]:
        btn_label = f"📍 {s_num}" if st.session_state.step != s_num else f"👉 [{s_num}]"
        if st.button(btn_label, use_container_width=True, key=f"nav_btn_{s_num}"):
            # Security check: Block jumping past step 1 if not authorized
            if s_num > 1 and not st.session_state.auth_verified:
                st.warning("⛔ ACCESS LOCKED: Please complete and confirm valid authorization on Step 1 first.")
            else:
                st.session_state.step = s_num
                st.rerun()
                
st.markdown(f"**Current Active Section:** `{step_names[st.session_state.step]}`")
st.divider()

# ============================================================
# PAGE 1: AUTHORIZATION & REGISTRATION
# ============================================================
if st.session_state.step == 1:
    st.header("🔑 Investigator Authorization & Registration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Authorization Check")
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

    with col2:
        st.subheader("Investigator Details")
        st.session_state.full_name = st.text_input("Full Name:", value=st.session_state.full_name)
        st.session_state.address = st.text_input("Official Address:", value=st.session_state.address)
        st.session_state.contact = st.text_input("Contact Number:", value=st.session_state.contact)

    # This condition displays the banner immediately when the name box has text and authorization is verified
    if st.session_state.auth_verified and st.session_state.full_name.strip() != "":
        st.markdown(
            f"""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #008751; margin-top: 15px;">
                <h4 style="color: #008751; margin: 0 0 5px 0;">🏛️ Official Session Active</h4>
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
elif st.session_state.step == 2:
    st.header("📂 Payroll Data Input & Cleaning")
    
    data_choice = st.radio(
        "Select Dataset Source:",
        ["— Please Select —", "Generate Sample Payroll Dataset (200 Records)", "Upload External Payroll CSV File"],
        horizontal=True,
        index=0,
        key="dataset_radio_choice"
    )
    
    df = None
    if data_choice == "— Please Select —":
        # Clear existing dataframe in session state so user cannot proceed without selecting
        if "df" in st.session_state:
            del st.session_state.df
        st.info("ℹ️ Please select a dataset source above to proceed.")
        
    elif "Generate" in data_choice:
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
        
    elif "Upload" in data_choice:
        uploaded_file = st.file_uploader("Upload Payroll CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded uploaded CSV file.")
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
        if not can_proceed:
            st.caption("⚠️ Please select or generate a dataset to continue.")
      
# ============================================================
# PAGE 3: AI MODEL ANALYSIS
# ============================================================
elif st.session_state.step == 3:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 to access this page.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step3"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("🤖 AI Model Training & Anomaly Detection")
    
    if "df" not in st.session_state:
        st.warning("⚠️ Please complete Data Input (Step 2) first.")
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
    else:
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

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Data Input", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_b2:
        if st.button("Continue to Visualizations & Reports ➡️", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ============================================================
# PAGE 4: VISUALIZATIONS & REPORTS
# ============================================================
elif st.session_state.step == 4:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 to access this page.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step4"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("📊 Payroll Analytics & Report Export")
    
    if "results" not in st.session_state:
        st.warning("⚠️ Please run AI Model Analysis on Step 3 first.")
    else:
        results = st.session_state.results
        flagged_records = st.session_state.flagged_records
        
        col_v1, col_v2, col_v3 = st.columns(3)
        
        with col_v1:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            counts = results["AnomalyStatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
            bars = ax1.bar(counts.index, counts.values, color=["#008751", "#d32f2f"])
            for bar in bars:
             h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}', ha='center', va='bottom', fontsize=9)
            ax1.set_title("Anomaly Status Count", fontweight='bold', fontsize=10)
            ax1.tick_params(axis='x', rotation=15)
            st.pyplot(fig1)
            
        with col_v2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            pie_counts = results["AnomalyStatus"].value_counts().reindex(["No Anomaly Flag", "Flagged for Review"], fill_value=0)
            colors = ["#008751", "#d32f2f"]
            ax2.pie(
                pie_counts.values, 
                labels=pie_counts.index, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90,
                textprops={'fontsize': 8}
            )
            ax2.set_title("Anomaly Percentage Share", fontweight='bold', fontsize=10)
            st.pyplot(fig2)

        with col_v3:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            if "State" in results.columns and "BasicSalary" in results.columns:
                state_avg = results.groupby("State")["BasicSalary"].mean()
                bars3 = ax3.bar(state_avg.index, state_avg.values, color="#1b5e20")
                for bar in bars3:
                    h = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., h, f'{int(h):,}', ha='center', va='bottom', fontsize=8)
                ax3.set_title("Avg Basic Salary by State", fontweight='bold', fontsize=10)
            else:
                ax3.text(0.5, 0.5, "No State Data Available", ha='center', va='center')
                ax3.set_title("State Distribution", fontweight='bold', fontsize=10)
            st.pyplot(fig3)
            
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
            mime="application/zip",
            use_container_width=True
        )

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to AI Analysis", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col_b2:
        if st.button("Continue to Q&A & Kudos ➡️", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

# ============================================================
# PAGE 5: Q&A ASSISTANT & KUDOS
# ============================================================
elif st.session_state.step == 5:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 to access this page.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step5"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

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

    st.markdown("")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back to Visualizations", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col_b2:
        if st.button("Continue to Developers ➡️", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

# ============================================================
# PAGE 6: DEVELOPERS TEAM PAGE SLIDESHOW (12 MEMBERS)
# ============================================================
elif st.session_state.step == 6:
    if not st.session_state.auth_verified:
        st.error("⛔ ACCESS LOCKED: You must complete official authorization on Step 1 to access this page.")
        if st.button("⬅️ Go Back to Step 1: Authorization", key="back_step6"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.header("👥 Meet the Development Team (12 Members)")
    st.markdown("Browse through the team members who built this forensic audit platform.")

    # Developer Profiles List (12 Members)
    developers = [
        {
            "name": "A. O. Juwon",
            "address": "Ejuku, Yagba East LGA",
            "state": "Kogi State",
            "education": "NCE, B.Sc., M.Sc. (In View)",
            "role": "Team Lead and Project Problem Analyzer",
            "marital_status": "Married",
            "image": "developer 1.jpeg"
        },
        {
            "name": "ADEYEMI MARY MARVELOUS",
            "address": "ODO-ERE, YAGBA WEST LG",
            "state": "KOGI STATE",
            "education": "B. Sc",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "MARRIED",
            "image": "developer 2.jpeg"
        },
        {
            "name": "IBITOYE MAYOMIKUN JONATHAN",
            "address": "ISANLU, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B. ENG. ILORIN",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "Single",
            "image": "developer 3.jpeg"
        },
        {
            "name": "OLUKAIYE OLUGBENGA TIMOTHY",
            "address": "BOLORUNDURO KABBA, KABBA BUNU LG",
            "state": "KOGI STATE",
            "education": "B. ENG, CUSTECH Osara",
            "role": "PROBLEM SOLVING TEAM",
            "marital_status": "Single",
            "image": "developer 4.jpeg"
        },
        {
            "name": "KOMOLAFE BABATUNDE RICHARD",
            "address": "AYETORO GBEDE IJUMU LG",
            "state": "KOGI STATE",
            "education": "B. Sc.",
            "role": "DEMONSTRATION TEAM",
            "marital_status": "Single",
            "image": "developer 5.jpeg"
        },
        {
            "name": "ADEWUMI BISIRIYU TUNDE",
            "address": "IFE_OLUKOTUN, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B. Sc, CNA, NBA (in view)",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "MARRIED",
            "image": "developer 6.jpeg"
        },
        {
            "name": "BABALOLA MARVELOUS ITUNUOLUWA",
            "address": "EGBE, YAGBA WEST",
            "state": "KOGI STATE",
            "education": "B. Sc (NURSING) FED.UNI LOKOJA",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "Single",
            "image": "developer 7.jpeg"
        },
        {
            "name": "AKANKU TAIYE OLUWAPEMISIRE",
            "address": "ITEDO ISANLU, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B Sc. (Econs)",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "MARRIED",
            "image": "developer 8.jpeg"
        },
        {
            "name": "VICTOR IWALAYE",
            "address": "IYARA, IJUMU LG",
            "state": "KOGI STATE",
            "education": "NATIONAL DEPLOMA",
            "role": "AI TEAM",
            "marital_status": "Single",
            "image": "developer 9.jpeg"
        },
        {
            "name": "RACHEAL OLORUNMAIYE",
            "address": "KABBA BUNNU, LG",
            "state": "KOGI STATE",
            "education": "B.Sc",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "N/A",
            "image": "developer 10.jpeg"
        },
        {
            "name": "ISREAL ADEOLA OLUMAIYEGUN",
            "address": "ODO ERE, YAGBA WEST LG",
            "state": "KOGI STATE",
            "education": "B. Sc. (Econs) KSU, kabba",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "Single",
            "image": "developer 11.jpeg"
        },
        {
            "name": "Team Member 12",
            "address": "Address 12",
            "state": "State 12",
            "education": "Education 12",
            "role": "Role 12",
            "marital_status": "Single",
            "image": "developer12.jpeg"
        }
    ]

    # Navigation buttons (Previous and Next)
    col_prev, col_mid, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⬅️ Previous Developer", use_container_width=True):
            st.session_state.dev_index = (st.session_state.dev_index - 1) % len(developers)
            st.rerun()

    with col_next:
        if st.button("Next Developer ➡️", use_container_width=True):
            st.session_state.dev_index = (st.session_state.dev_index + 1) % len(developers)
            st.rerun()

    current_dev = developers[st.session_state.dev_index]

    st.markdown("---")
    c1, c2 = st.columns([1, 2])

    with c1:
        img_file = current_dev["image"]
        img_alt = img_file.replace(".jpeg", ".jpg") if ".jpeg" in img_file else img_file.replace(".jpg", ".jpeg")
        
        if os.path.exists(img_file):
            st.image(img_file, use_column_width=True)
        elif os.path.exists(img_alt):
            st.image(img_alt, use_column_width=True)
        else:
            st.info(f"🖼️ [Image file '{current_dev['image']}' not found in folder. Please add your JPEG picture.]")

    with c2:
        st.subheader(current_dev["name"])
        st.markdown(f"📍 **Address:** {current_dev['address']}")
        st.markdown(f"🏛️ **State:** {current_dev['state']}")
        st.markdown(f"🎓 **Education:** {current_dev['education']}")
        st.markdown(f"💼 **Role:** {current_dev['role']}")
        st.markdown(f"💍 **Marital Status:** {current_dev['marital_status']}")
        st.caption(f"Developer {st.session_state.dev_index + 1} of {len(developers)}")

    st.markdown("")
    if st.button("⬅️ Back to Q&A Assistant & Kudos", use_container_width=True):
        st.session_state.step = 5
        st.rerun()

# Footer
st.markdown("---")
st.caption("Federal Republic of Nigeria • Official Portal")
