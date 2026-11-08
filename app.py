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
    layout="centered",
)

# ============================================================
# GLOBAL STYLING (compact, phone-friendly, professional)
# ============================================================
st.markdown("""
<style>
:root {
    --green: #0b6b46;
    --green-dark: #084d33;
    --green-soft: #eaf3ee;
    --red: #c62828;
    --blue: #1565c0;
    --ink: #1f2933;
    --muted: #6b7780;
    --line: #d9e1dd;
    --bg: #f4f6f5;
}

.stApp { background-color: var(--bg); color: var(--ink); }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 760px;
    padding-top: 2.2rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
}
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }
hr { margin: 0.8rem 0 !important; }

/* Hidden marker elements (used only to target rows in CSS) */
div[data-testid="stElementContainer"]:has(.mk),
div.element-container:has(.mk) {
    display: none !important;
}

/* ---------- Header banner ---------- */
.custom-header-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    background-color: #ffffff;
    padding: 8px 12px;
    border-radius: 10px;
    border: 1px solid var(--line);
    border-top: 4px solid var(--green);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}
.header-img { width: 42px; height: 42px; object-fit: contain; }
.header-emoji { font-size: 30px; line-height: 1; }
.header-title-box { text-align: center; flex-grow: 1; }
.top-title {
    color: var(--green-dark);
    font-weight: 800;
    font-size: 14px;
    margin: 0;
    line-height: 1.25;
}
.top-sub {
    color: var(--muted);
    font-size: 10.5px;
    font-weight: 500;
    margin: 2px 0 0 0;
    line-height: 1.3;
}

/* ---------- Page title bar ---------- */
.page-title {
    border-left: 4px solid var(--green);
    background-color: #ffffff;
    border-top: 1px solid var(--line);
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 9px 12px;
    border-radius: 6px;
    margin: 4px 0 6px 0;
    color: var(--green-dark);
    font-weight: 700;
    font-size: 1.05rem;
}

/* ---------- Default buttons (green) ---------- */
.stButton > button, .stDownloadButton > button {
    background-color: var(--green) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    min-height: 2.6rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: var(--green-dark) !important;
}
.stButton > button:disabled {
    background-color: #b9c4bf !important;
    color: #ffffff !important;
}

/* ---------- News link ---------- */
.news-btn-container a {
    background-color: var(--red);
    color: #ffffff !important;
    padding: 9px 10px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    font-size: 12px;
    display: block;
    text-align: center;
    white-space: nowrap;
}

/* ---------- Top navigation row: always ONE line ---------- */
div[data-testid="stHorizontalBlock"]:has(.mk-top) {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    align-items: center !important;
}
div[data-testid="stHorizontalBlock"]:has(.mk-top) > div {
    min-width: 0 !important;
}

/* ---------- Welcome box ---------- */
.session-box {
    background-color: var(--green-soft);
    padding: 10px 12px;
    border-radius: 8px;
    border-left: 4px solid var(--green);
    margin-top: 10px;
    color: var(--green-dark);
    font-size: 0.92rem;
}

/* ---------- Developer card ---------- */
div[data-testid="stImage"] img {
    width: 100% !important;
    max-height: 32vh;
    object-fit: cover;
    object-position: center top;
    border-radius: 10px;
}
.dev-card {
    background-color: #ffffff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px;
}
.dev-name {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--ink);
    line-height: 1.25;
}
.dev-role {
    display: inline-block;
    background-color: var(--green-soft);
    color: var(--green-dark);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    margin: 6px 0 8px 0;
}
.dev-grid {
    display: grid;
    grid-template-columns: 90px 1fr;
    row-gap: 5px;
    column-gap: 8px;
    font-size: 0.86rem;
    line-height: 1.35;
}
.dev-grid .k { color: var(--muted); font-weight: 600; }
.dev-grid .v { color: var(--ink); font-weight: 500; }
.dev-counter {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin: 0;
}

/* ---------- Previous (BLUE) and Next (RED): ONE line ---------- */
div[data-testid="stHorizontalBlock"]:has(.mk-dev) {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
}
div[data-testid="stHorizontalBlock"]:has(.mk-dev) > div {
    width: 50% !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
div[data-testid="stHorizontalBlock"]:has(.mk-dev) > div button {
    width: 100% !important;
}
div[data-testid="stHorizontalBlock"]:has(.mk-dev) > div:nth-child(1) button {
    background-color: var(--blue) !important;
    color: #ffffff !important;
}
div[data-testid="stHorizontalBlock"]:has(.mk-dev) > div:nth-child(2) button {
    background-color: var(--red) !important;
    color: #ffffff !important;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 12px;
}
</style>
""", unsafe_allow_html=True)


def get_image_base64(filepath):
    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        ext = filepath.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        return f"data:{mime};base64,{encoded}"
    return ""


def page_title(text):
    st.markdown(f'<div class="page-title">{text}</div>', unsafe_allow_html=True)


# ============================================================
# CHART FUNCTIONS
# ============================================================
GREEN = "#0b6b46"
DARK_GREEN = "#1b5e20"
RED = "#c62828"


def chart_pca(d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    n = d[~d["flagged"]]
    f = d[d["flagged"]]
    ax.scatter(n["pc1"], n["pc2"], s=12, c=GREEN, label="Normal", alpha=0.7)
    ax.scatter(f["pc1"], f["pc2"], s=35, c=RED, marker="x", label="Flagged")
    ax.set_title("Payroll Records (PCA View)", fontweight="bold", fontsize=10)
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def chart_lga(d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if "lga" in d.columns:
        c = d[d["flagged"]]["lga"].value_counts()
        if not c.empty:
            ax.barh(c.index, c.values, color=DARK_GREEN)
            ax.set_title("Flagged Staff per LGA", fontweight="bold", fontsize=10)
            ax.set_xlabel("Flagged Count", fontsize=9)
            ax.set_ylabel("LGA", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No LGA Flagged Records", ha="center", va="center")
            ax.set_title("Flagged Staff per LGA", fontweight="bold", fontsize=10)
    else:
        ax.text(0.5, 0.5, "LGA Column Missing", ha="center", va="center")
    plt.tight_layout()
    return fig


def chart_status_bar(d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    counts = d["anomalystatus"].value_counts().reindex(
        ["No Anomaly Flag", "Flagged for Review"], fill_value=0
    )
    bars = ax.bar(counts.index, counts.values, color=[GREEN, RED])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{int(h)}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Anomaly Status Count", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=10, labelsize=8)
    plt.tight_layout()
    return fig


def chart_status_pie(d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    pie_counts = d["anomalystatus"].value_counts().reindex(
        ["No Anomaly Flag", "Flagged for Review"], fill_value=0
    )
    ax.pie(
        pie_counts.values,
        labels=pie_counts.index,
        colors=[GREEN, RED],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax.set_title("Anomaly Share (%)", fontweight="bold", fontsize=10)
    plt.tight_layout()
    return fig


def chart_salary_dist(d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if "netsalary" in d.columns:
        ax.hist(d["netsalary"], bins=15, color=GREEN, edgecolor="white", alpha=0.85)
        ax.set_title("Net Salary Distribution", fontweight="bold", fontsize=10)
        ax.set_xlabel("Net Salary (₦)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NetSalary Column Missing", ha="center", va="center")
    plt.tight_layout()
    return fig


def chart_allowance_scatter(d):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if "allowances" in d.columns and "netsalary" in d.columns:
        n = d[~d["flagged"]]
        f = d[d["flagged"]]
        ax.scatter(n["allowances"], n["netsalary"], color=GREEN, label="Normal", alpha=0.6, s=25)
        ax.scatter(f["allowances"], f["netsalary"], color=RED, label="Flagged", marker="X", s=70)
        ax.set_title("Allowances vs Net Salary Outliers", fontweight="bold", fontsize=10)
        ax.set_xlabel("Allowances (₦)", fontsize=9)
        ax.set_ylabel("Net Salary (₦)", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Required Financial Columns Missing", ha="center", va="center")
    plt.tight_layout()
    return fig


# ============================================================
# TOP HEADER
# ============================================================
coat_arm_path = ""
for filename in ["coat_of_arms.jpeg", "coat_of_arms.jpg", "coat_of_arms.png"]:
    if os.path.exists(filename):
        coat_arm_path = filename
        break
coat_arm_base64 = get_image_base64(coat_arm_path)

school_logo_path = ""
for filename in ["school_logo.jpeg", "school_logo.jpg", "school_logo.png",
                 "logo.jpeg", "logo.jpg", "logo.png"]:
    if os.path.exists(filename):
        school_logo_path = filename
        break
school_logo_base64 = get_image_base64(school_logo_path)

if coat_arm_base64:
    left_img = "<img src='" + coat_arm_base64 + "' class='header-img'>"
else:
    left_img = "<span class='header-emoji'>🇳🇬</span>"

if school_logo_base64:
    right_img = "<img src='" + school_logo_base64 + "' class='header-img'>"
else:
    right_img = "<span class='header-emoji'>🏫</span>"

st.markdown(
    "<div class='custom-header-container'>"
    "<div>" + left_img + "</div>"
    "<div class='header-title-box'>"
    "<p class='top-title'>International Business School</p>"
    "<p class='top-sub'>AI Payroll Anomaly Detection &amp; Forensic Audit Portal</p>"
    "</div>"
    "<div>" + right_img + "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
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
# NAVIGATION MENU + NEWS BUTTON (one line)
# ============================================================
step_names = {
    1: "1. Authorization",
    2: "2. Data Input",
    3: "3. AI Analysis",
    4: "4. Visualizations",
    5: "5. Q&A & Kudos",
    6: "6. Developers",
}

col_nav, col_news = st.columns([4, 1.4])

with col_nav:
    st.markdown("<span class='mk mk-top'></span>", unsafe_allow_html=True)
    selected_step_name = st.selectbox(
        "Quick Navigation",
        options=list(step_names.values()),
        index=st.session_state.step - 1,
        label_visibility="collapsed",
    )
    selected_step = int(selected_step_name.split(".")[0])

with col_news:
    st.markdown(
        "<div class='news-btn-container'>"
        "<a href='https://www.nairaland.com' target='_blank'>📰 News</a>"
        "</div>",
        unsafe_allow_html=True,
    )

if selected_step != st.session_state.step:
    if selected_step > 1 and (not st.session_state.auth_verified or not st.session_state.full_name.strip()):
        st.warning("⛔ Complete Authorization on Step 1 first.")
    elif selected_step > 2 and "df" not in st.session_state:
        st.warning("⛔ Load data on Step 2 first.")
    elif selected_step > 3 and "results" not in st.session_state:
        st.warning("⛔ Run AI Analysis on Step 3 first.")
    else:
        st.session_state.step = selected_step
        st.rerun()

st.progress(st.session_state.step / len(step_names))
current_label = step_names[st.session_state.step].split(". ", 1)[1]
st.caption("Step " + str(st.session_state.step) + " of " + str(len(step_names)) + ": " + current_label)

# ============================================================
# PAGE 1: AUTHORIZATION
# ============================================================
if st.session_state.step == 1:
    page_title("🔑 Investigator Authorization & Registration")

    auth_choice = st.radio(
        "Do you have official authorization?",
        ["-- Please Select --", "YES", "NO"],
        index=1 if st.session_state.auth_verified else 0,
        horizontal=True,
    )

    if auth_choice == "-- Please Select --":
        st.session_state.auth_verified = False
        st.info("Select your authorization status to proceed.")
    elif auth_choice == "NO":
        st.session_state.auth_verified = False
        st.error("⛔ Access denied: proper authorization is required.")
    else:
        st.session_state.auth_verified = True
        st.success("✅ Authorization confirmed.")

    st.subheader("Investigator Details")
    st.session_state.full_name = st.text_input("Full Name", value=st.session_state.full_name)
    st.session_state.address = st.text_input("Official Address", value=st.session_state.address)
    st.session_state.contact = st.text_input("Contact Number", value=st.session_state.contact)

    if st.session_state.auth_verified and st.session_state.full_name.strip() != "":
        st.markdown(
            "<div class='session-box'><b>Official session active.</b><br>"
            "Welcome, <b>" + st.session_state.full_name + "</b>. "
            "Your credentials have been logged for this forensic session.</div>",
            unsafe_allow_html=True,
        )

    if st.button("Continue to Data Input ➡️", use_container_width=True):
        if not st.session_state.auth_verified:
            st.warning("⚠️ Please complete valid authorization first.")
        elif not st.session_state.full_name.strip():
            st.warning("⚠️ Please enter your full name to proceed.")
        else:
            st.session_state.step = 2
            st.rerun()

# ============================================================
# PAGE 2: DATA INPUT
# ============================================================
if st.session_state.step == 2:
    if not st.session_state.auth_verified or not st.session_state.full_name.strip():
        st.error("⛔ Access locked: complete official authorization on Step 1 first.")
        if st.button("⬅️ Back to Step 1: Authorization"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    page_title("📂 Payroll Data Input & Cleaning")

    data_choice = st.radio(
        "Select dataset source:",
        ["— Please Select —",
         "Generate Sample Payroll Dataset (500 Records)",
         "Upload External Payroll CSV File"],
        index=0,
        key="dataset_radio_choice",
    )

    df = None
    if data_choice == "— Please Select —":
        if "df" in st.session_state:
            del st.session_state.df
        st.info("Select a dataset source above to proceed.")

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
            "deductions": np.maximum(np.random.normal(6000, 1800, num_emp), 0).round(2),
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
            st.success("Uploaded CSV file loaded successfully.")
        else:
            if "df" in st.session_state:
                del st.session_state.df
            st.info("Awaiting CSV upload...")

    if df is not None:
        st.session_state.df = df
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back", use_container_width=True, key="p2_back"):
            st.session_state.step = 1
            st.rerun()
    with col_b2:
        can_proceed = "df" in st.session_state and st.session_state.df is not None
        if st.button("AI Analysis ➡️", use_container_width=True, disabled=not can_proceed, key="p2_next"):
            if can_proceed:
                st.session_state.step = 3
                st.rerun()

# ============================================================
# PAGE 3: AI MODEL ANALYSIS
# ============================================================
if st.session_state.step == 3:
    if not st.session_state.auth_verified:
        st.error("⛔ Access locked: complete authorization on Step 1 first.")
        if st.button("⬅️ Go Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
        st.stop()
    if "df" not in st.session_state:
        st.error("⛔ Access locked: load or generate data on Step 2 first.")
        if st.button("⬅️ Go Back to Step 2"):
            st.session_state.step = 2
            st.rerun()
        st.stop()

    page_title("🤖 AI Model Training & Anomaly Detection")

    df = st.session_state.df
    contamination_rate = st.slider("Expected anomaly sensitivity rate (%)", 1, 20, 7) / 100.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ignore_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]
    feature_cols = [c for c in numeric_cols if c.lower() not in ignore_words]

    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0)
    X = X[X.columns[X.nunique() > 1]]

    if len(X) < 10:
        st.error("At least 10 records are required for analysis.")
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

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back", use_container_width=True, key="p3_back"):
            st.session_state.step = 2
            st.rerun()
    with col_b2:
        if st.button("Visualizations ➡️", use_container_width=True, key="p3_next"):
            st.session_state.step = 4
            st.rerun()

# ============================================================
# PAGE 4: VISUALIZATIONS & REPORTS
# ============================================================
if st.session_state.step == 4:
    if not st.session_state.auth_verified or "results" not in st.session_state:
        st.error("⛔ Access locked: please complete earlier steps first.")
        st.stop()

    page_title("📊 Advanced Payroll Analytics")
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
        "Value": [
            "Federal Republic of Nigeria Payroll System",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state.full_name,
            st.session_state.contact,
            len(results),
            len(flagged_records),
        ],
    })

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nigeria_payroll_summary.csv", summary_df.to_csv(index=False))
        zf.writestr("full_results.csv", results.to_csv(index=False))
        zf.writestr("flagged_records.csv", flagged_records.to_csv(index=False))

    st.download_button(
        label="📥 Download Official Audit ZIP Package",
        data=zip_buffer.getvalue(),
        file_name="Nigeria_Payroll_Audit_" + datetime.now().strftime("%Y%m%d") + ".zip",
        mime="application/zip",
        use_container_width=True,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back", use_container_width=True, key="p4_back"):
            st.session_state.step = 3
            st.rerun()
    with col_b2:
        if st.button("Q&A & Kudos ➡️", use_container_width=True, key="p4_next"):
            st.session_state.step = 5
            st.rerun()

# ============================================================
# PAGE 5: Q&A ASSISTANT & KUDOS
# ============================================================
if st.session_state.step == 5:
    if not st.session_state.auth_verified or "results" not in st.session_state:
        st.error("⛔ Access locked: complete prior steps first.")
        st.stop()

    page_title("❓ Project Q&A Assistant & User Kudos")

    q_input = st.text_input("Ask a question about the system")
    if q_input:
        ans = "I do not have a prepared answer for that question."
        if "algorithm" in q_input.lower() or "isolation" in q_input.lower():
            ans = "The system uses Isolation Forest, an unsupervised machine learning algorithm."
        elif "training" in q_input.lower():
            ans = "Training uses 80% of the dataset to learn standard payroll patterns."
        elif "report" in q_input.lower():
            ans = "The report contains summary statistics, full data results, and flagged anomaly tables packaged into a ZIP file."
        st.info("Assistant: " + ans)

    st.divider()
    st.subheader("🌟 Leave Your Kudos")
    with st.form("kudos_form"):
        k_name = st.text_input("Your name or department", value=st.session_state.full_name)
        k_rating = st.select_slider(
            "Rate system performance",
            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            value="Excellent",
        )
        k_comment = st.text_area("Your kudos or comments", value="Great application!", height=90)
        submitted = st.form_submit_button("Submit Kudos")
        if submitted:
            st.success("🎉 Thank you! Your kudos have been logged successfully.")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⬅️ Back", use_container_width=True, key="p5_back"):
            st.session_state.step = 4
            st.rerun()
    with col_b2:
        if st.button("Developers ➡️", use_container_width=True, key="p5_next"):
            st.session_state.step = 6
            st.rerun()

# ============================================================
# PAGE 6: DEVELOPERS TEAM
# ============================================================
if st.session_state.step == 6:
    if not st.session_state.auth_verified or "results" not in st.session_state:
        st.error("⛔ Access locked: complete prior steps first.")
        st.stop()

    page_title("👥 Meet the Development Team (12 Members)")

    developers = [
        {"name": "A. O. JUWON", "address": "Ejuku, Yagba East LGA", "state": "Kogi State", "education": "NCE, B.Sc., M.Sc. (In View)", "role": "Team Lead and Project Problem Analyzer", "marital_status": "Married", "image": "developer 1.jpeg"},
        {"name": "ADEYEMI MARY MARVELOUS", "address": "ODO-ERE, YAGBA WEST LG", "state": "KOGI STATE", "education": "B. Sc", "role": "AI TEAM", "marital_status": "MARRIED", "image": "developer 2.jpeg"},
        {"name": "IBITOYE MAYOMIKUN JONATHAN", "address": "ISANLU, YAGBA EAST LG", "state": "KOGI STATE", "education": "B.Sc (ENG) ILORIN", "role": "DATA COLLECTION TEAM", "marital_status": "Single", "image": "developer 3.jpeg"},
        {"name": "OLUKAIYE OLUGBENGA TIMOTHY", "address": "BOLORUNDURO KABBA, KABBA BUNU LG", "state": "KOGI STATE", "education": "B.Sc. (ENGINEERING), CUSTECH Osara", "role": "PROBLEM SOLVING TEAM", "marital_status": "Single", "image": "developer 4.jpeg"},
        {"name": "KOMOLAFE BABATUNDE RICHARD", "address": "AYETORO GBEDE IJUMU LG", "state": "KOGI STATE", "education": "B. Sc.", "role": "DEMONSTRATION TEAM", "marital_status": "Single", "image": "developer 5.jpeg"},
        {"name": "ADEWUMI BISIRIYU TUNDE", "address": "IFE_OLUKOTUN, YAGBA EAST LG", "state": "KOGI STATE", "education": "B. Sc, CNA, MBA (in view)", "role": "DATA COLLECTION TEAM", "marital_status": "MARRIED", "image": "developer 6.jpeg"},
        {"name": "BABALOLA MARVELOUS ITUNUOLUWA", "address": "EGBE, YAGBA WEST", "state": "KOGI STATE", "education": "B. Sc (NURSING) FED.UNI LOKOJA", "role": "DEMONSTRATION TEAM MEMBER", "marital_status": "Single", "image": "developer 7.jpeg"},
        {"name": "AKANKU TAIYE OLUWAPEMISIRE", "address": "ITEDO ISANLU, YAGBA EAST LG", "state": "KOGI STATE", "education": "B Sc. (Econs)", "role": "DEMONSTRATION TEAM MEMBER", "marital_status": "MARRIED", "image": "developer 8.jpeg"},
        {"name": "VICTOR IWALAYE", "address": "IYARA, IJUMU LG", "state": "KOGI STATE", "education": "NATIONAL DEPLOMA", "role": "AI TEAM", "marital_status": "Single", "image": "developer 9.jpeg"},
        {"name": "RACHEAL OLORUNMAIYE (Shellie)", "address": "KABBA BUNNU, LG", "state": "KOGI STATE", "education": "B.Sc", "role": "DATA COLLECTION TEAM", "marital_status": "N/A", "image": "developer 10.jpeg"},
        {"name": "ISREAL ADEOLA OLUMAIYEGUN", "address": "ODO ERE, YAGBA WEST LG", "state": "KOGI STATE", "education": "B. Sc. (Econs) KSU, kabba", "role": "DEMONSTRATION TEAM MEMBER", "marital_status": "Single", "image": "developer 11.jpeg"},
        {"name": "OLUWATOSIN MOSES O", "address": "ODO-ERE, YAGBA WEST", "state": "KOGI STATE", "education": "B.Sc., (Sociology)", "role": "AI TEAM", "marital_status": "MARRIED", "image": "developer 12.jpeg"},
    ]

    current_dev = developers[st.session_state.dev_index]

    # Photo
    img_file = current_dev["image"]
    if ".jpeg" in img_file:
        img_alt = img_file.replace(".jpeg", ".jpg")
    else:
        img_alt = img_file.replace(".jpg", ".jpeg")

    if os.path.exists(img_file):
        st.image(img_file, use_container_width=True)
    elif os.path.exists(img_alt):
        st.image(img_alt, use_container_width=True)
    else:
        st.info("Photo not found: " + current_dev["image"])

    # Details card
    st.markdown(
        "<div class='dev-card'>"
        "<div class='dev-name'>" + current_dev["name"] + "</div>"
        "<span class='dev-role'>" + current_dev["role"] + "</span>"
        "<div class='dev-grid'>"
        "<div class='k'>Address</div><div class='v'>" + current_dev["address"] + "</div>"
        "<div class='k'>State</div><div class='v'>" + current_dev["state"] + "</div>"
        "<div class='k'>Education</div><div class='v'>" + current_dev["education"] + "</div>"
        "<div class='k'>Marital status</div><div class='v'>" + current_dev["marital_status"] + "</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='dev-counter'>Developer " + str(st.session_state.dev_index + 1)
        + " of " + str(len(developers)) + "</p>",
        unsafe_allow_html=True,
    )

    # Previous (BLUE) and Next (RED) on the same line
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("<span class='mk mk-dev'></span>", unsafe_allow_html=True)
        if st.button("⬅️ Previous", key="dev_prev_btn"):
            st.session_state.dev_index = (st.session_state.dev_index - 1) % len(developers)
            st.rerun()
    with b_col2:
        if st.button("Next ➡️", key="dev_next_btn"):
            st.session_state.dev_index = (st.session_state.dev_index + 1) % len(developers)
            st.rerun()

    if st.button("⬅️ Back to Q&A Assistant", use_container_width=True, key="p6_back"):
        st.session_state.step = 5
        st.rerun()

st.caption("Federal Republic of Nigeria • Official Portal")
