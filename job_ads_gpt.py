# Skill Core-ness Analyzer: Identifying must-have skills for a profession (via TheirStack)


import requests
import json
import streamlit as st
from collections import Counter
import os
from openai import OpenAI
from requests.exceptions import HTTPError
from cv_gpt_4_parser import standardize_skills
from cv_gpt_4_parser import match_skills_to_esco_specific

# --- User input and config ---
#st.set_page_config(layout="wide")
#st.title("📊 Core vs Optional Skill Analyzer (TheirStack)")

#gpt_key = os.getenv("OPENAI_API_KEY")
#theirstack_key = os.getenv("THEIRSTACK_API_KEY")

#job_title = st.text_input("💼 Enter a job title to analyze", placeholder="e.g. software developer")
limit = 10  # Number of job ads to fetch
threshold = 2  # Minimum frequency for skills to be considered relevant

# --- Step 1: Fetch Job Ads from TheirStack ---
def fetch_theirstack_jobs(keyword: str, limit: int = 10, theirstack_key: str = "") -> list[str]:
    url = "https://api.theirstack.com/v1/jobs/search"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {theirstack_key}"
    }
    payload = {
        "page": 0,
        "limit": limit,
        "job_title_or": [
            keyword,
        ],
        "posted_at_max_age_days": 15,
        "blur_company_data": False,
        "order_by": [
            {
                "desc": True,
                "field": "date_posted"
            }
        ],  
        "job_country_code_or": ["SE"],
        "include_total_results": False,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        full_json = resp.json()
        
        jobs = full_json.get("data", [])
        descriptions = []

        for job in jobs:
            description = job.get("description")
            if isinstance(description, str) and description.strip():
                descriptions.append(description.strip())
        
        return descriptions
    except HTTPError as http_err:
        if resp.status_code == 402:
            st.error("🚫 The TheirStack API credits have been exhausted or the free plan has expired.")
        else:
            st.error(f"❌ HTTP error occurred: {http_err}")
        return None



# --- Step 2: Extract skills using OpenAI ---
def extract_skills(texts: list[str],  gpt_key: str) -> list[list[str]]:
    client = OpenAI(
        api_key=gpt_key,
        base_url="https://anast.ita.chalmers.se:4000"  # Custom endpoint for Chalmers deployment
    )
    prompt = f"""
    You are an AI assistant specialized in analyzing job advertisements.

    Your task is to extract relevant **skills, tools, technologies, and job-related competencies** mentioned in the following job descriptions. Focus on specific, actionable skill names (e.g., "Python", "Git", "CI/CD", "Project Management"), not broad categories (e.g., "Software Development").

    Return the result as a **JSON list of lists**, where each inner list contains the skills found in one job description.

    Descriptions: {texts} ...

    Return format example:
    [
      ["Python", "Docker", "Agile"],
      ["JavaScript", "React", "Git"],
      ["SQL", "Scrum", "Team leadership"]
    ]

    Return only valid JSON — no markdown, no triple backticks, no explanation.
    """
    response = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        extracted = json.loads(response.choices[0].message.content)
        return extracted
    except Exception as e:
        st.error(f"❌ Failed to parse skills: {e}")
        return []


# --- Step 3: GPT Core Classification ---
def classify_skills(skills: list[str], gpt_key: str) -> dict[str, list[str]]:
    client = OpenAI(
        api_key=gpt_key,
        base_url="https://anast.ita.chalmers.se:4000"  # Custom endpoint for Chalmers deployment
    )
    prompt = f"""
    Classify the following skills into core and optional for a modern {skills} in Sweden.
    - Core = Needed in almost all roles.
    - Optional = Only needed in some roles or projects.

    Take only the technical and soft skills into account, not languages required.
    Don't include any skills that is the same as the job title.
    Don't include duplicates.

    Skills: {skills}
    Return JSON: {{"core": [...], "optional": [...]}}
    Return only valid JSON — no markdown code blocks, no triple backticks, and no text outside the JSON.
    """
    response = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Failed to classify skills: {e}")
        return {"core": [], "optional": []}
    
def validate_job_title_with_gpt(title: str, api_key: str) -> bool:
    client = OpenAI(api_key=api_key, base_url="https://anast.ita.chalmers.se:4000")
    prompt = f"""
    Decide if the following string is a real and meaningful job title in a professional context. Examples of valid titles include "software engineer", "data analyst", or "project manager".

    Title: "{title}"
    Respond with "yes" if valid, otherwise "no".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return "yes" in response.choices[0].message.content.strip().lower()
    except Exception:
        return False

# --- Step 5: Logic ---
def run_job_analysis(job_title: str, gpt_key: str, theirstack_key: str):
    #if st.button("🔍 Analyze Job Market for This Role"):
    if validate_job_title_with_gpt(job_title, gpt_key):
        if job_title.strip():
            with st.spinner("🔍 Fetching job ads and analyzing..."):
                try:
                    job_ads = fetch_theirstack_jobs(job_title, limit, theirstack_key)

                    if job_ads is not None:

                        skill_lists = extract_skills(job_ads, gpt_key)

                        flat_skills = [s for sub in skill_lists for s in sub] 

                        counts = Counter(flat_skills)

                        total = sum(counts.values())

                        skill_info = [
                            {"skill": s, "count": c, "pct": round(c / total * 100, 1)}
                            for s, c in counts.items() if c >= threshold
                        ]
                        skill_info.sort(key=lambda x: x["count"], reverse=True)      

                        skill_names = [s["skill"] for s in skill_info]

                        skill_names = standardize_skills(skill_names, gpt_key)

                        classification = classify_skills(skill_names, gpt_key)

                        def attach_pct(skills_in_category):
                            return [
                                {"skill": s, "pct": next((i["pct"] for i in skill_info if i["skill"] == s), 0)}
                                for s in skills_in_category
                            ]

                        core_skills = attach_pct(classification["core"])
                        opt_skills = attach_pct(classification["optional"])

                        st.markdown("""
                        ### ℹ️ How to Interpret the Results

                        The lists below show **core** and **optional** skills for the role of **_{}_.**  
                        These were extracted from **recent job postings in Sweden** using AI and matched across multiple employers.

                        - **Percentage (%)** next to each skill shows how often that skill appeared relative to other skills across the job ads analyzed.
                        - **Core Skills** are common across most job postings — they're **must-haves** for this profession.
                        - **Optional Skills** appear less frequently and are often **role- or company-specific**.
                        """.format(job_title.capitalize()))

                        st.subheader("📈 Core and Must-Have Skills in this Role")
                        for skill in core_skills:
                            st.markdown(f"- **{skill['skill']}** ({skill['pct']}%)")
                            st.session_state["core_skills"] = core_skills

                        st.subheader("🧩 Very Good and Nice-to-Have Skills in this Role but that vary company to company")
                        for skill in opt_skills:
                            st.markdown(f"- {skill['skill']} ({skill['pct']}%)")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    else:
        st.error("Invalid job title please provide a valid one.")
