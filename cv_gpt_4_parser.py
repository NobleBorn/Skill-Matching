from openai import OpenAI
from PyPDF2 import PdfReader
import streamlit as st
import os
import json

# Streamlit UI
#st.set_page_config(layout="wide")
#st.title("📄 Skill Matching with AI")
#st.markdown("Upload your CV in PDF format and get structured insights using OpenAI GPT-4.")

# OpenAI API key (should be set as an environment variable for safety)
#api_key = os.getenv("OPENAI_API_KEY")
#if not api_key:
#    st.warning("⚠️ Please set your OPENAI_API_KEY environment variable.")

# Upload PDF file
#uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Call OpenAI API to extract structured CV data
def parse_cv_with_gpt(cv_text, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://anast.ita.chalmers.se:4000"  # Custom endpoint for Chalmers deployment
    )
    prompt = f"""
    You are an AI that extracts structured information from resumes (CVs). The input will be raw, unstructured text parsed from a PDF or DOCX CV.

    Your task is to return a JSON object with the following **fixed and consistently named** top-level fields:

    1. "work_experience": A list of work roles with:
    - "job_title"
    - "organization"
    - "period"
    - "responsibilities" (as a list of bullet points)

    2. "education": A list of education entries with:
    - "degree"
    - "university"
    - "graduation_year"

    3. "programming_languages_and_technical_skills": A list of skills (e.g. Python, SQL, Git)

    4. "projects": A list of past projects with:
    - "name"
    - "technologies_used" (as a list)
    - "outcomes" (short description)

    5. "certifications_and_languages":
    - "certifications": a list of:
        - "name"
        - "institution"
        - "year"
    - "languages_spoken": a list of:
        - "language"
        - "proficiency"

    Use only the keys as shown above. Do not use any other names or formats.
    Return only valid JSON — no markdown code blocks, no triple backticks, and no text outside the JSON.

    CV TEXT:
    {cv_text}
    """


    try:
        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from CVs."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# Render a friendly UI view of extracted info
def render_structured_output(json_text):
    try:
        data = json.loads(json_text)
    except Exception as e:
        st.error("⚠️ Could not parse AI response as JSON. Showing raw response:")
        st.code(json_text, language="json")
        return

    col1, = st.columns([1])

    with col1:
        st.subheader("🧠 Skills")
        skills = data.get("programming_languages_and_technical_skills", [])
        if skills:
            st.code(", ".join(skills))
        return skills


def standardize_skills(skills: list[str], api_key: str) -> list[str]:
    client = OpenAI(
        api_key=api_key,
        base_url="https://anast.ita.chalmers.se:4000"
    )

    prompt = f"""
    Normalize and expand the following list of technical skills. For each skill:
    - Expand abbreviations (e.g., JS → JavaScript, Py → Python)
    - Standardize synonyms (e.g., coding → programming)
    - Return only canonical, full-length, lowercase names where possible
    - Do not separate multi-word skills into individual words (e.g., "CI/CD" should remain as "CI/CD" or "Continuous Integration / Continuous Deployment")

    Skills:
    {skills}

    Return the result as a JSON list of clean skill names, no duplicates.
    Return only valid JSON — no markdown code blocks, no triple backticks, and no text outside the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return [f"Error: {e}"]


def match_skills_to_esco_specific(skills: list[str], api_key: str) -> dict:
    client = OpenAI(
        api_key=api_key,
        base_url="https://anast.ita.chalmers.se:4000"
    )

    prompt = f"""
    You are a skill normalization assistant aligned with the ESCO and O*NET databases.

    For each skill below:
    - Match it to the most **specific and semantically correct** skill label in ESCO or O*NET.
    - Do NOT generalize. For example, match \"programming\" as \"Programming\" — not \"Software Development\".
    - Always return the most precise **canonical skill name** as it would appear in ESCO/O*NET.
    Return a JSON dictionary that maps each input skill to its most precise **canonical name** as found in ESCO or O*NET.
    - Do not add verbs or modifiers like \"use\", \"develop in\", \"working with\".
    - Only return the **skill label** as it appears in ESCO/O*NET.
    - For example if the exact match is \"Java\", return \"Java\", not \"use Java\" or \"Java programming\".
    - If no match exists, return the **skill label** as it appears in the input.

    Input skills:
    {json.dumps(skills)}

    Return only a JSON dictionary mapping each input skill to its best standardized equivalent.
    Return only valid JSON — no markdown code blocks, no triple backticks, and no text outside the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# Render combined skill output
def render_skills_view(skills):
    st.subheader("🧠 Combined Standardized Skills")
    if skills:
        for skill in sorted(skills):
            st.code(f"- {skill}")
    else:
        st.info("No skills to show.")

# This function uses the OpenAI API to suggest skills based on a user-provided description
def suggest_skills_from_description(description: str, api_key: str) -> list[str]:
    client = OpenAI(api_key=api_key, base_url="https://anast.ita.chalmers.se:4000")

    prompt = f"""
    The user will describe what they do in a natural, human way (e.g. 'I work on cloud infrastructure, mostly with AWS and CI/CD').

    Your task is to extract the **key technical or soft skills** they are describing — especially those that are relevant to job matching, as listed in ESCO/O*NET.

    Normalize the names:
    - Expand abbreviations (e.g., CI/CD → Continuous Integration / Continuous Deployment) but do not separate them into multiple skills.
    - Map skills to **standard labels** (e.g., 'coding in Java' → 'Java')
    - Avoid verbs or modifiers like 'working with' or 'use'
    - Return only a JSON list of clean skill names, like:
    ["Java", "Git", "Continuous Deployment"]
    - If possible, use skill names that are one word or a well-known phrase. 

    USER DESCRIPTION:
    {description}

    Return only valid JSON — no markdown code blocks, no triple backticks, and no text outside the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return [f"Error: {e}"]

# Validate skills using GPT-4 to ensure they are real and relevant
def validate_skills_with_gpt(skills: list[str], api_key: str) -> list[str]:
    client = OpenAI(api_key=api_key, base_url="https://anast.ita.chalmers.se:4000")

    prompt = f"""
    Given the following list of items, return only those that are valid skills, technologies, tools, or programming concepts.
    Ignore any that are:
    - Not real words (e.g., 'baba', 'sdfsdf')
    - Not known domains in tech, business, science, design, etc.
    - Accept valid but niche names like 'CATIA', 'SAS', 'Blender'.

    Input:
    {json.dumps(skills)}

    Return only valid JSON — no markdown code blocks, no triple backticks, and no text outside the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.5-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return [f"Error validating skills: {e}"]

# Main logic
def run_cv_analysis(uploaded_file, api_key):
    if uploaded_file and api_key:
        with st.spinner("🔍 Extracting text from CV..."):
            raw_text = extract_text_from_pdf(uploaded_file)

        st.subheader("📝 Extracted Raw Text")
        st.text_area("Preview", raw_text, height=300)

        if st.button("Analyze CV with GPT-4"):
            with st.spinner("🤖 Contacting OpenAI API for structured parsing..."):
                result = parse_cv_with_gpt(raw_text, api_key)
                st.session_state["parsed_result"] = result

    if "parsed_result" in st.session_state:
        st.subheader("📊 GPT-4 Structured Output")
        result = st.session_state.get("parsed_result")
        st.code(result, language="json")
        extracted_skills = render_structured_output(result)
        st.session_state["extracted_skills"] = extracted_skills


        st.markdown("### ➕ Add additional skills you have (comma-separated):")
        manual_skills = st.text_input(label="Manual skill input", placeholder="e.g. React, Spring Boot, GraphQL", label_visibility="collapsed")

        st.markdown("### 💬 Don’t know what skill name to type, ask our AI?")
        user_description = st.text_area("Describe what you’ve done or can do (e.g., 'I built REST APIs using Node.js and deployed on AWS')")

        if st.button("📝 Suggest Skills from My Description"):
            with st.spinner("🤖 Thinking..."):
                suggested_skills = suggest_skills_from_description(user_description, api_key)
                st.session_state["suggested_skills"] = suggested_skills

                if suggested_skills:
                    st.success("💡 Suggested Skill Names:")
                    for skill in st.session_state.get("suggested_skills", []):
                        st.markdown(f"- {skill}")
                else:
                    st.error("⚠️ No skills found in your description. Please try again with more details.")
        
        if st.button("✅ Normalize and Combine All Skills"):
            with st.spinner("🤖 Normalizing and Combining skills with OpenAI API..."):
                if manual_skills:
                    additional_skills = [s.strip() for s in manual_skills.split(",") if s.strip()]
                else:
                    additional_skills = []
                
                valid_skills = validate_skills_with_gpt(additional_skills, api_key)
                invalid_skills = list(set(additional_skills) - set(valid_skills))
                
                if invalid_skills:
                    st.error(f"⚠️ Please check the spelling or try different terms. These are not valid skills: {', '.join(invalid_skills)}.")

                if valid_skills:
                    st.success(f"✅ Added: {', '.join(valid_skills)}")
                else:
                    st.warning("No valid skills were added, but we will still proceed using the CV-only skills.")
                    
                if "extracted_skills" in st.session_state:
                    extracted_skills = st.session_state.get("extracted_skills", [])
                    combined_skills = extracted_skills + valid_skills

                    standardized_skills = standardize_skills(combined_skills, api_key)
                    st.subheader("🔍 Standardized Skills")
                    st.code(standardized_skills, language="json")

                    skill_matches = match_skills_to_esco_specific(standardized_skills, api_key)
                    st.subheader("📚 ESCO/O*NET Skill Matches")
                    st.code(skill_matches, language="json")

                    st.session_state["cv_skills"] = skill_matches

                    if "cv_skills" in st.session_state:
                        render_skills_view(st.session_state["cv_skills"])
                    else:
                        render_skills_view(skill_matches.values())
                

