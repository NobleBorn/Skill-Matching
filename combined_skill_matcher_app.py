import streamlit as st
# ---------- Page config ----------
st.set_page_config(page_title="AI Skill Matcher", layout="wide")
st.info("🔒 **Privacy Notice**: We do not store or log any uploaded CVs or user input. All data is processed in-memory and discarded after your session.")

from job_ads_gpt import run_job_analysis
from cv_gpt_4_parser import run_cv_analysis
import os
from openai import OpenAI

# ---------- Environment ----------
gpt_key = os.getenv("OPENAI_API_KEY")
theirstack_key = os.getenv("THEIRSTACK_API_KEY")

client = OpenAI(api_key=gpt_key, base_url="https://anast.ita.chalmers.se:4000")

# ---------- Tabs ----------
CV_tab, job_ads_tab, comparison_tab = st.tabs(["📄 Skill Matching with AI", "📊 Job Market Analysis", "🔍 Final Skill Comparison Results"])

# --- CV Tab ---
with CV_tab:
    uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])
    if uploaded_file:
        run_cv_analysis(uploaded_file, gpt_key)


# --- Job Ad Tab ---
with job_ads_tab:
    job_title = st.text_input("💼 Enter a job title to analyze", placeholder="e.g. software developer")
    if st.button("🔍 Analyze Job Market for This Role"):
        run_job_analysis(job_title, gpt_key, theirstack_key)

with comparison_tab: 
    st.header("🧮 Compare Your Skills With Market Demands")

    user_skills = set(s.lower() for s in st.session_state.get("cv_skills", []))
    core_skills_raw = [s["skill"] for s in st.session_state.get("core_skills", [])]

    if user_skills and core_skills_raw:
        core_skills = set([s.lower() for s in core_skills_raw])

        matched_skills = sorted(user_skills & core_skills)
        missing_skills = sorted(core_skills - user_skills)

        if matched_skills:
            st.success("✅ You have the following core skills that match the required skills for: {}".format(job_title))
            for skill in matched_skills:
                st.markdown(f"- ✅ **{skill}**")
        if missing_skills:
            st.warning("🚨 You're missing some core skills commonly required for: {}".format(job_title))
            for skill in missing_skills:
                st.markdown(f"- ❌ **{skill}**")
        else:
            st.success("✅ Great! You already have all the core skills listed in the job ads.")
    else:
        st.info("ℹ️ Please analyze both your CV and job ads before comparing.")