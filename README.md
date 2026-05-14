# 📚 AI Resume Screening App

![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-blue?logo=streamlit)
![Python](https://img.shields.io/badge/Language-Python-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Streamlit-based NLP-powered resume screening tool that classifies resumes into job categories and ranks them by relevance. It also compares resumes against job descriptions using cosine similarity.

---

## 🚀 Features

- 📄 Extracts and cleans resume text from `.pdf`, `.docx`, and `.txt`
- 🧠 Predicts job categories using a trained ML model
- 📊 Ranks multiple resumes by prediction confidence
- 📝 Compares resume content with pasted job descriptions
- 💬 Collects user feedback

---

## 🔍 Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- Plotly
- NumPy, pandas, re, PyPDF2, python-docx

---

## 📦 Folder Structure
resume-screening-app/
│
├── app.py # Streamlit app interface
├── clf.pkl # Trained classifier (multiclass)
├── tfidf.pkl # TF-IDF vectorizer
├── encoder.pkl # Label encoder for categories
├── requirements.txt # All dependencies
├── UpdatedResumeDataSet.xls # Resume dataset used for training
├── README.md # You're reading this!

---


---

## ✅ Requirements

```bash
pip install -r requirements.txt
```
---
## 🚀 Run the App

```bash
streamlit run app.py
```
---

##🧠 Model Training (optional)
The classifier was trained using Scikit-learn on a labeled resume dataset with a TF-IDF vectorizer and a Multinomial Naive Bayes classifier.
To reproduce the training pipeline, see the notebook:
Resume Screening with Python.ipynb

---

## 📌 Example Predictions
- ✅ Filename: resume1.pdf

- ✅ Predicted Category: Data Science

- ✅ Confidence: 92.5%

- ✅ JD Similarity Score: 88.7%

---

## 🌟 Contribution
- Pull requests and enhancement ideas are welcome!
- Feel free to fork the repo and add new features or suggest improvements.

---


## 🏷️ Tags
#NLP #Streamlit #MachineLearning #ResumeClassifier #AIRecruitment #PortfolioProject


---

### ✅ How to Use This
1. Save this file as `README.md` in the root directory of your project.
2. Push it to GitHub — it will render beautifully and make your repo more attractive.

Let me know if you want me to generate the matching `requirements.txt` or zip the full folder for upload.

---

## ✅ 2. `requirements.txt`

Generate this automatically (recommended):

```bash
pip freeze > requirements.txt
```

