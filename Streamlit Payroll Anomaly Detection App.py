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
# PAGE CONFIGURATION & WATERMARK STYLING
# ============================================================
st.set_page_config(
    page_title="Federal Republic of Nigeria - AI Payroll Forensic Portal",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background with EFCC & Nigeria Watermark */
    .stApp {
        background-color: #f7f9f8;
        background-image: linear-gradient(rgba(247, 249, 248, 0.95), rgba(247, 249, 248, 0.95)), 
                          url('https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Coat_of_arms_of_Nigeria.svg/500px-Coat_of_arms_of_Nigeria.svg.png');
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
        background-size: 35% auto;
    }
    
    /* Green-White-Green National Banner */
    .nigeria-banner {
        background: linear-gradient(90deg, #008751 0%, #008751 33%, #ffffff 33%, #ffffff 66%, #008751 66%, #008751 100%);
        height: 10px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    /* Top Official Header */
    .top-header {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.15);
        text-align: center;
        margin-bottom: 20px;
    }
    .top-header img {
        width: 70px;
        height: auto;
        margin-bottom: 8px;
    }
    .top-header-title {
        color: #008751;
        font-weight: 900;
        font-size: 22px;
        text-transform: uppercase;
        margin: 0;
    }
    .top-header-sub {
        color: #1b5e20;
        font-size: 14px;
        font-weight: 700;
        margin: 0;
    }
    
    .content-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #008751;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
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
# TOP HEADER WITH NIGERIAN COAT OF ARMS
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

coat_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Coat_of_arms_of_Nigeria.svg/300px-Coat_of_arms_of_Nigeria.svg.png"

st.markdown(f"""
    <div class="top-header">
        <img src="{coat_url}" alt="Coat of Arms of Nigeria">
        <p class="top-header-title">Federal Republic of Nigeria</p>
        <p class="top-header-sub">AI Payroll Anomaly Detection & EFCC Forensic Audit Portal</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# ============================================================
# MULTI-PAGE NAVIGATION SETUP VIA SIDEBAR
# ============================================================
st.sidebar.title("📌 Navigation Menu")
page = st.sidebar.radio(
    "Select Portal Page:",
    [
        "1. Authorization & Registration",
        "2. Data Input & Cleaning",
        "3. AI Model Analysis",
        "4. Visualizations & Reports",
        "5. Q&A Assistant & Kudos"
    ]
)

# Initialize Session State
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
        auth_choice = st.radio(
            "Do you have official authorization to access this forensic system?",
            options=["YES", "NO"],
            index=0
        )
        
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
        st.session_state.contact = st.text_input("Contact Phone Number:", value=st.session_state.contact)

    if st.session_state.auth_verified:
        st.markdown(f"""
            <div class="content-card">
                <h3 style="color: #008751; margin-top: 0;">🏛️ Session Active</h3>
                <p><b>Investigator:</b> {st.session_state.full_name}</p>
                <p><b>Address:</b> {st.session_state.address}</p>
                <p><b>Contact:</b> {st.session_state.
