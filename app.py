# app.py — Streamlit Frontend for AutoMind DS-Agent
# Production-Stable Version (Python 3.11)

import streamlit as st
import requests
import pandas as pd
import os
import json

# ---------------- UI CONFIG ---------------- #
st.set_page_config(
    page_title="AutoMind DS-Agent",
    layout="wide",
    page_icon="🤖",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🤖 AutoMind DS-Agent")
st.sidebar.write("Advanced Autonomous ML + EDA + Explainability Agent")

# Backend Health Check
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False

status = check_backend()
if status:
    st.sidebar.success("🟢 Backend Connected")
else:
    st.sidebar.error("🔴 Backend Offline")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
autodetect = st.sidebar.checkbox("Auto Detect Target", value=True)
manual_target = st.sidebar.text_input("Target Column (optional)")
tune = st.sidebar.slider("Tuning Iterations", 5, 50, 20)
advanced_fe = st.sidebar.checkbox("Enable Advanced Feature Engineering", value=False)
sample_frac = st.sidebar.slider("Sample Fraction (1.0 = full dataset)", 0.1, 1.0, 1.0)


# ---------------- MAIN CONTENT ---------------- #
st.title("AutoMind DS-Agent — End-to-End Autonomous ML Pipeline")

st.write(
    "Upload your dataset and let the AI Agent perform: EDA, preprocessing, feature engineering, "
    "model training, explainability, reporting, and notebook generation."
)

# ---------------- RUN PIPELINE ---------------- #
if st.button("🚀 Run AutoMind Agent"):
    if uploaded is None:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Running full AutoMind Agent pipeline... this may take a few minutes"):

            # Correct safe file handling
            files = {
                "file": (uploaded.name, uploaded.getvalue(), "text/csv")
            }

            data = {
                "autodetect": str(autodetect),
                "target": manual_target,
                "tune": str(tune),
                "advanced_fe": str(advanced_fe),
                "sample_frac": str(sample_frac)
            }

            try:
                r = requests.post(
                    f"{BACKEND_URL}/run_agent",
                    files=files,
                    data=data,
                    timeout=2000
                )

                if r.status_code != 200:
                    st.error(f"Backend Error:\n\n{r.text}")
                else:
                    res = r.json()

                    st.success("🎉 Pipeline Completed Successfully!")
                    st.json(res)

                    run_id = res.get("run_id")

                    if run_id:
                        st.subheader("📄 Reports & Downloads")
                        st.markdown(f"[📘 HTML Report]({BACKEND_URL}/reports/{run_id})")
                        st.markdown(f"[📦 Download Artifact ZIP]({BACKEND_URL}/artifact/{run_id})")

            except Exception as e:
                st.error(f"❌ Request failed: {e}")


# ---------------- RUN HISTORY ---------------- #
st.markdown("---")
st.header("📊 Run History")

try:
    r = requests.get(f"{BACKEND_URL}/runs", timeout=5)
    if r.status_code == 200:
        runs_data = r.json().get("runs", [])

        # Ensure always a list
        if not isinstance(runs_data, list):
            runs_data = []

        if len(runs_data) == 0:
            st.info("No previous runs found.")
        else:
            df = pd.DataFrame(runs_data)
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Could not load run history (backend error).")

except Exception:
    st.warning("Could not connect to backend for run history.")
