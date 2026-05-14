import streamlit as st
import pandas as pd
import pickle
import re
import docx
import PyPDF2
import numpy as np
import os
import zipfile
import requests
from PIL import Image
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Utils (defined BEFORE train_local_model) ------------------- #
def clean_resume(txt):
    txt = re.sub(r"http\S+|www\S+|https\S+", '', txt)
    txt = re.sub(r'@\w+|#', '', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def handle_file_upload(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return "".join([p.extract_text() or "" for p in PyPDF2.PdfReader(file).pages])
    elif ext == 'docx':
        return "\n".join([p.text for p in docx.Document(file).paragraphs])
    elif ext == 'txt':
        try:
            return file.read().decode("utf-8")
        except UnicodeDecodeError:
            return file.read().decode("latin-1")
    return ""

# ------------------- Train / Load Local Model ------------------- #
@st.cache_resource
def train_local_model():
    try:
        df = pd.read_csv("UpdatedResumeDataSet.xls")
    except Exception:
        df = pd.read_excel("UpdatedResumeDataSet.xls", engine="xlrd")

    text_col = "Resume"
    label_col = "Category"

    df[text_col] = df[text_col].astype(str).apply(clean_resume)

    tfidf = TfidfVectorizer(stop_words="english", max_features=7000)
    X = tfidf.fit_transform(df[text_col])

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    clf = OneVsRestClassifier(LinearSVC())
    clf.fit(X, y)

    return tfidf, le, clf

tfidf, le, clf = train_local_model()

# ------------------- Sidebar ------------------- #
st.set_page_config(page_title="AI Internship Resume Screening System", layout="wide")
st.sidebar.title("🤖 AI Internship Matcher")

menu = st.sidebar.radio("Go to", [
    "🏠 Home", "📄 Upload Resumes", "⚖️ Compare with JD", "📖 About", "💬 Feedback"
])

# ------------------- Home ------------------- #
if menu == "🏠 Home":
    st.title("AI-Powered Resume Classifier")
    st.write("Upload your resume to get predicted industry categories using NLP and Machine Learning.")
    st.markdown("This tool helps job seekers and recruiters match resumes to domains using AI.")

# ------------------- Upload Resumes ------------------- #
elif menu == "📄 Upload Resumes":
    st.title("📄 Upload & Analyze Resumes")
    show_raw = st.checkbox("📄 Show Extracted Text")
    uploaded_files = st.file_uploader(
        "Upload your resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    resume_data = []
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"🔍 Analyzing `{file.name}`..."):
                text = handle_file_upload(file)
                cleaned = clean_resume(text)

                if not cleaned.strip():
                    st.warning(f"{file.name} appears empty or unreadable.")
                    continue

                vector = tfidf.transform([cleaned])
                prediction = clf.predict(vector)
                category = le.inverse_transform(prediction)[0]

                resume_data.append({
                    "Filename": file.name,
                    "Predicted Category": category,
                    "Match Level": "Category predicted successfully",
                    "Raw Text": text[:300] + "..." if show_raw else "Hidden"
                })

        if resume_data:
            df = pd.DataFrame(resume_data)
            st.success(f"✅ Processed {len(df)} resume(s)")
            st.dataframe(df, use_container_width=True)

            st.markdown("### 📊 Category Distribution")
            fig = px.histogram(df, x="Predicted Category", title="Resume Categories")
            st.plotly_chart(fig)

# ------------------- Compare with JD ------------------- #
elif menu == "⚖️ Compare with JD":
    st.title("📝 Resume vs Job Description")
    jd_text = st.text_area("Paste Job Description")
    resume_file = st.file_uploader("Upload a single resume", type=["pdf", "docx", "txt"])

    if jd_text and resume_file:
        resume_txt = handle_file_upload(resume_file)
        resume_cleaned = clean_resume(resume_txt)
        jd_cleaned = clean_resume(jd_text)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_cleaned, jd_cleaned])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        st.success(f"Match Score: {round(score * 100, 2)}%")
        if score >= 0.7:
            st.info("✅ Great match!")
        elif score >= 0.5:
            st.warning("⚠️ Moderate match. Some tweaking needed.")
        else:
            st.error("❌ Weak match. Consider rephrasing your resume.")

# ------------------- About ------------------- #
elif menu == "📖 About":
    st.title("👩‍💻 About This App")
    st.markdown("""
This AI-powered resume classifier helps:

* 🔍 Categorize resumes into industries.
* 📊 Rank multiple resumes by confidence.
* 🧠 Compare resumes with job descriptions.

**Built with:** Python, Streamlit, Scikit-learn, Plotly  
""")

# ------------------- Feedback ------------------- #
elif menu == "💬 Feedback":
    st.title("💬 We Value Your Feedback")
    name = st.text_input("Name")
    comment = st.text_area("What's on your mind?")
    if st.button("Submit"):
        st.success("✅ Thanks for your feedback!")