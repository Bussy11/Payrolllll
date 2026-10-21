import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import warnings

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="AI Payroll Anomaly Detection System",
    page_icon="🔍",
    layout="wide"
)

# App Title & Description
st.title("🔍 AI Payroll Anomaly Detection System")
st.markdown("""
This application uses Machine Learning (**Isolation Forest**) to identify unusual payroll records 
that may require human investigation and auditing.
""")

st.markdown("---")

# Sidebar - User Authorization and Registration
st.sidebar.header("🔐 User Authorization")
auth_status = st.sidebar.radio("Do you have authorization to use this system?", ["Select", "YES", "NO"])

if auth_status == "NO":
    st.error("⛔ ACCESS DENIED: Please obtain authorization before proceeding.")
    st.stop()

elif auth_status == "YES":
    st.sidebar.success("Authorization Confirmed")
    st.sidebar.subheader("👤 Registration Details")
    
    full_name = st.sidebar.text_input("Full Name")
    address = st.sidebar.text_input("Address")
    contact_number = st.sidebar.text_input("Contact Number")

    if not full_name or not address or not contact_number:
        st.info("👈 Please complete your registration details in the sidebar to begin.")
        st.stop()
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**User:** {full_name}")

    # Main Area - Data Selection
    st.header("1. 📂 Data Input")
    data_option = st.radio(
        "Choose payroll data source:",
        ["Generate Sample Payroll Dataset", "Upload Payroll CSV File"]
    )

    df = None

    if data_option == "Upload Payroll CSV File":
        uploaded_file = st.file_uploader("Upload your payroll CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.stop()
        else:
            st.info("Awaiting CSV file upload.")
            st.stop()

    elif data_option == "Generate Sample Payroll Dataset":
        if st.button("Generate Sample Dataset"):
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

            # Synthetic anomalies
            df.loc[10, "BasicSalary"] = 350000
            df.loc[10, "NetSalary"] = df.loc[10, "BasicSalary"] + df.loc[10, "Allowances"] - df.loc[10, "Deductions"]

            df.loc[25, "Allowances"] = 200000
            df.loc[25, "NetSalary"] = df.loc[25, "BasicSalary"] + df.loc[25, "Allowances"] - df.loc[25, "Deductions"]

            st.session_state['df'] = df

        if 'df' in st.session_state and data_option == "Generate Sample Payroll Dataset":
            df = st.session_state['df']

    if df is not None:
        # Display Initial Data Summary
        st.subheader("Raw Data Preview")
        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(df))
        col2.metric("Total Columns", len(df.columns))
        st.dataframe(df.head(), use_container_width=True)

        # ------------------------------------------------------------
        # DATA CLEANING AND PREPARATION
        # ------------------------------------------------------------
        st.header("2. 🧹 Data Cleaning & Feature Preparation")

        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.dropna(how="all").copy()

        # Clean numeric formatting (commas, currency symbols)
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

        # Select Features for Model
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        identifier_words = ["id", "employeeid", "staffid", "recordnumber", "bank_account"]

        feature_columns = [
            col for col in numeric_columns
            if col.lower() not in identifier_words and "anomaly" not in col.lower()
        ]

        if not feature_columns:
            st.error("No valid numerical features available for machine learning after filtering.")
            st.stop()

        X = df[feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        X = X.fillna(X.median())
        varying_columns = X.columns[X.nunique() > 1].tolist()
        X = X[varying_columns]
        feature_columns = X.columns.tolist()

        st.write("**Selected Features for Anomaly Detection:**", feature_columns)

        if len(X) < 10:
            st.error("At least 10 records are required to perform anomaly detection.")
            st.stop()

        # ------------------------------------------------------------
        # MODEL TRAINING AND ANOMALY DETECTION
        # ------------------------------------------------------------
        st.header("3. 🤖 AI Anomaly Detection Model")

        contamination_rate = st.slider("Target Anomaly Contamination Rate (%)", 1, 15, 5) / 100.0

        if st.button("Run Anomaly Detection Analysis"):
            with st.spinner("Training Isolation Forest Model and analyzing records..."):
                # Split & Scale Data
                X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train Isolation Forest Model
                model = IsolationForest(
                    n_estimators=200,
                    contamination=contamination_rate,
                    random_state=42
                )
                model.fit(X_train_scaled)

                # Predict on Full Dataset
                all_scaled = scaler.transform(X)
                all_predictions = model.predict(all_scaled)
                all_scores = model.decision_function(all_scaled)

                # Add results to DataFrame
                results = df.copy()
                results["AnomalyStatus"] = np.where(all_predictions == -1, "Flagged for Review", "No Anomaly Flag")
                results["AnomalyScore"] = all_scores.round(4)

                flagged_records = results[results["AnomalyStatus"] == "Flagged for Review"]

                # Results Summary Metrics
                st.subheader("📊 Analysis Summary")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Total Records Analyzed", len(results))
                res_col2.metric("Flagged Records ⚠️", len(flagged_records))
                res_col3.metric("Normal Records ✅", len(results) - len(flagged_records))

                st.warning("⚠️ **Note:** Flagged records require human investigation and auditing. They are not automatically fraudulent.")

                # Display Flagged Records
                st.subheader("🚨 Flagged Anomaly Records")
                if not flagged_records.empty:
                    st.dataframe(flagged_records, use_container_width=True)
                else:
                    st.success("No anomalies detected at the current threshold.")

                # Visualizations
                st.subheader("📉 Anomaly Score Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(results["AnomalyScore"], bins=30, color="skyblue", edgecolor="black")
                ax.axvline(x=0, color="red", linestyle="--", label="Anomaly Threshold Boundary")
                ax.set_title("Distribution of Decision Scores (Lower = More Anomalous)")
                ax.set_xlabel("Anomaly Score")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)

                # Export Results
                st.subheader("📥 Export Analysis Results")
                csv_buffer = io.BytesIO()
                results.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                st.download_button(
                    label="Download Full Results CSV",
                    data=csv_buffer,
                    file_name="payroll_anomaly_results.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Select your authorization status in the sidebar to proceed.")