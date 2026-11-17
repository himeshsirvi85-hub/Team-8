import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a standard dictionary of technical skills to look for
DEFAULT_SKILLS_DB = [
    "python",
    "java",
    "c++",
    "javascript",
    "typescript",
    "react",
    "node.js",
    "sql",
    "nosql",
    "mongodb",
    "postgresql",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "machine learning",
    "deep learning",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "fastapi",
    "flask",
    "django",
    "linux",
    "agile",
    "scrum",
]


def extract_text_from_pdf(uploaded_file):
  try:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
      extracted = page.extract_text()
      if extracted:
        text += extracted + "\n"
    return text
  except Exception as e:
    st.error(f"Error reading PDF: {e}")
    return ""


def extract_skills_from_text(text, skills_db):
  text_lower = text.lower()
  found_skills = set()
  for skill in skills_db:
    pattern = r"\b" + re.escape(skill) + r"\b"
    if re.search(pattern, text_lower):
      found_skills.add(skill)
  return found_skills


def calculate_match_score(resume_text, jd_text):
  if not resume_text.strip() or not jd_text.strip():
    return 0.0
  vectorizer = TfidfVectorizer(stop_words="english")
  tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
  similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
  return round(float(similarity[0][0]) * 100, 2)


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Skill-Gap Analyzer", layout="centered"
)

st.title("Career Skill Gap Analyzer")
# st.write(
#     "Bridge the gap between your resume profile and real-world job requirements"
# )

col1, col2 = st.columns(2)

with col1:
  st.subheader("1. Upload Resume")
  uploaded_file = st.file_uploader(
      "Upload Resume (PDF format)", type=["pdf"]
  )

with col2:
  st.subheader("2. Job Description")
  jd_input = st.text_area(
      "Paste Job Description text here...",
      height=150,
      placeholder="Paste requirements, tech stack, and responsibilities...",
  )

if st.button("Analyze Skill Gap", type="primary"):
  if not uploaded_file:
    st.warning("Please upload a resume PDF first.")
  elif not jd_input.strip():
    st.warning("Please paste a target job description.")
  else:
    with st.spinner("Analyzing profile alignment and detecting gaps..."):
      # Extract text
      resume_text = extract_text_from_pdf(uploaded_file)

      if not resume_text.strip():
        st.error(
            "Could not extract text from the uploaded PDF. Ensure it is a"
            " text-based PDF."
        )
      else:
        # 1. Calculate Semantic Similarity Score
        semantic_score = calculate_match_score(resume_text, jd_input)

        # 2. Extract Skills
        resume_skills = extract_skills_from_text(resume_text, DEFAULT_SKILLS_DB)
        jd_skills = extract_skills_from_text(jd_input, DEFAULT_SKILLS_DB)

        # 3. Compute Gap Analysis
        matched_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills - resume_skills

        # Skill Coverage Ratio
        skill_coverage = (
            (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
        )

        # Overall Weighted Score (60% Skill Coverage + 40% Semantic Match)
        overall_score = round(
            (0.60 * skill_coverage) + (0.40 * semantic_score), 1
        )

        # --- Display Results Dashboard ---
        st.divider()
        st.header("Analysis Results")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Overall Match Score", f"{overall_score}%")
        metric_col2.metric("Skill Coverage", f"{round(skill_coverage, 1)}%")
        metric_col3.metric("Semantic Similarity", f"{semantic_score}%")

        st.subheader("Matched Skills Found")
        if matched_skills:
          st.success(", ".join([s.title() for s in sorted(matched_skills)]))
        else:
          st.info(
              "No direct skill overlaps found from the standard library. Try"
              " adding more keyword contexts."
          )

        st.subheader("Missing Skills (Skill Gaps)")
        if missing_skills:
          st.warning(
              "Consider adding projects or experience covering these skills:"
              f" **{', '.join([s.title() for s in sorted(missing_skills)])}**"
          )
        else:
          st.success(
              "Fantastic! Your resume covers all key terms detected in the job"
              " description."
          )