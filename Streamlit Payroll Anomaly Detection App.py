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

# SVG Graphic Fallbacks for Guaranteed Mobile Rendering
SVG_COAT_OF_ARMS = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="70" height="70">
  <rect width="100" height="100" rx="15" fill="#008751"/>
  <text x="50" y="42" font-size="28" text-anchor="middle" fill="#ffffff">🇳🇬</text>
  <path d="M 30 55 Q 50 45 70 55 Q 50 80 30 55" fill="#ffffff"/>
  <circle cx="50" cy="60" r="6" fill="#008751"/>
  <text x="50" y="88" font-size="8" font-weight="bold" text-anchor="middle" fill="#ffffff">COAT OF ARMS</text>
</svg>
"""

SVG_EFCC_LOGO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="70" height="70">
  <circle cx="50" cy="50" r="45" fill="#d32f2f" stroke="#008751" stroke-width="5"/>
  <text x="50" y="45" font-size="20" font-weight="900" text-anchor="middle" fill="#ffffff">EFCC</text>
  <text x="50" y="65" font-size="8" font-weight="bold" text-anchor="middle" fill="#ffffff">FORENSIC</text>
</svg>
"""

st.markdown("""
    <style>
    /* Main Background with Subtle Dot Pattern */
    .stApp {
        background-color: #f7f9f8;
        background-image: radial-gradient(#008751 0.75px, transparent 0.75px), radial-gradient(#008751 0.75px, #f7f9f8 0.75px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
    }
    .nigeria-banner {
        background: linear-gradient(90deg, #008751 0%, #008751 33%, #ffffff 33%, #ffffff 66%, #008751 66%, #008751 100%);
        height: 10px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .top-header {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0px 4px 15px rgba(0, 135, 81, 0.15);
        text-align: center;
        margin-bottom: 20px;
    }
    .top-header-title {
        color: #008751;
        font-weight: 900;
        font-size: 22px;
        text-transform: uppercase;
        margin: 10px 0 2px 0;
    }
    .top-header-sub {
        color: #1b5e20;
        font-size: 14px;
        font-weight: 700;
        margin: 0;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 25px;
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
# TOP HEADER WITH BOTH COAT OF ARMS & EFCC LOGOS
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

header_html = f"""
<div class="top-header">
    <div class="logo-container">
        <!-- Nigerian Coat of Arms -->
        <div>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Coat_of_arms_of_Nigeria.svg/250px-Coat_of_arms_of_Nigeria.svg.png" 
                 width="70" alt="Coat of Arms" 
