# 🛡️ HireGuard AI — Recruitment Fraud Detection

> An end-to-end Machine Learning and NLP system for detecting potentially fraudulent online job advertisements.

HireGuard AI analyzes the textual content of job postings and predicts whether an advertisement is **Likely Legitimate** or **Potentially Fraudulent**.

The project goes beyond model training by integrating the trained Machine Learning pipeline with a **FastAPI REST API** and an interactive **HTML/CSS/JavaScript frontend**, creating a complete deployable ML application.

---

## 🚀 Project Overview

Online recruitment platforms have made job searching easier, but they have also created opportunities for fraudulent job advertisements involving fake companies, unrealistic salaries, registration fees, misleading work-from-home offers, and attempts to collect personal information.

**HireGuard AI** explores whether Natural Language Processing and Machine Learning can identify textual patterns associated with fraudulent recruitment advertisements.

The system performs:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Class imbalance analysis
- Natural Language Processing
- TF-IDF feature extraction
- Machine Learning model training
- Multi-model evaluation
- Hyperparameter tuning
- Model serialization
- Real-time prediction
- REST API development
- Interactive frontend integration

---

## 🎯 Problem Statement

Build an end-to-end Data Science system capable of classifying online job advertisements as:

- ✅ **Likely Legitimate**
- 🚨 **Potentially Fraudulent**

The project also aims to demonstrate how a Machine Learning model developed inside a Data Science notebook can be converted into a usable application through an API and web interface.

---

## 📊 Dataset

The project uses a dataset containing real and fraudulent job advertisements with information such as:

- Job title
- Location
- Department
- Company profile
- Job description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Fraudulent label

### Dataset after preprocessing

| Metric | Value |
|---|---:|
| Total Job Postings | 17,632 |
| Legitimate Postings | 16,776 |
| Fraudulent Postings | 856 |
| Fraud Rate | 4.85% |

The dataset is therefore **highly imbalanced**, making accuracy alone an insufficient metric for evaluating fraud detection performance.

---

## 🔍 Exploratory Data Analysis

The notebook investigates the dataset before model development.

Major analysis includes:

- Dataset dimensions and structure
- Missing-value analysis
- Target class distribution
- Fraud percentage
- Text-length characteristics
- Common terms in fraudulent advertisements
- Common terms in legitimate advertisements
- Word-frequency analysis
- Visualization of recruitment fraud patterns

The EDA revealed a strong imbalance between legitimate and fraudulent advertisements.

Because only approximately **4.85%** of the processed observations are fraudulent, special attention was given to:

- Precision
- Recall
- F1-score

rather than relying only on overall accuracy.

---

## 🧹 Data Preprocessing

Textual fields were combined to create a unified representation of each job advertisement.

The preprocessing pipeline includes operations such as:

1. Handling missing values
2. Removing unnecessary columns
3. Combining relevant textual attributes
4. Converting text to lowercase
5. Removing URLs and unwanted characters
6. Removing punctuation
7. Removing extra whitespace
8. Preparing cleaned text for vectorization

The processed dataset is stored separately from the original dataset to preserve reproducibility.

---

## 🧠 NLP Pipeline

Machine Learning algorithms cannot directly process raw textual job advertisements.

HireGuard AI therefore converts the cleaned advertisements into numerical representations using:

### TF-IDF — Term Frequency-Inverse Document Frequency

TF-IDF assigns importance to terms based on their relevance within an advertisement relative to the complete collection of job postings.

The resulting sparse feature matrix is then supplied to the classification algorithms.

### Pipeline

```text
Raw Job Advertisement
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Combined Text Features
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Classifier
        ↓
Fraud Risk Prediction
```

---

## 🤖 Machine Learning Models

Multiple classification algorithms were evaluated rather than selecting a model based on a single experiment.

The project compares:

### 1. Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 97.59% |
| Precision | 70.87% |
| Recall | 85.38% |
| F1-Score | 77.45% |

Logistic Regression achieved strong fraud recall but produced more false-positive predictions.

### 2. Multinomial Naive Bayes

| Metric | Score |
|---|---:|
| Accuracy | 96.34% |
| Precision | 65.67% |
| Recall | 51.46% |
| F1-Score | 57.70% |

Although computationally efficient for text classification, Naive Bayes detected substantially fewer fraudulent advertisements.

### 3. Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 98.44% |
| Precision | 96.77% |
| Recall | 70.18% |
| F1-Score | 81.36% |

Random Forest produced excellent precision but lower fraud recall.

### 4. Linear Support Vector Machine

The Linear SVM provided the strongest overall balance between fraud detection precision and recall.

---

## 🏆 Final Model — Linear SVM

After comparing the candidate models, **Linear SVM** was selected as the final classifier.

Hyperparameter tuning identified:

```text
Best C = 1.0
```

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **98.50%** |
| Fraud Precision | **86.42%** |
| Fraud Recall | **81.87%** |
| Fraud F1-Score | **84.08%** |

### Classification Report

```text
              precision    recall    f1-score    support

Legitimate       0.99       0.99       0.99        3356
Fraudulent       0.86       0.82       0.84         171

Accuracy                               0.98        3527
Macro Avg        0.93       0.91       0.92        3527
Weighted Avg     0.98       0.98       0.98        3527
```

The final model correctly classified the majority of both legitimate and fraudulent advertisements while maintaining a strong balance between precision and recall.

---

## 📈 Why F1-Score Matters

The dataset contains significantly fewer fraudulent advertisements than legitimate ones.

A classifier predicting almost every advertisement as legitimate could therefore obtain deceptively high accuracy.

For this reason, HireGuard AI evaluates the minority fraud class using:

**Precision**

Measures how many advertisements predicted as fraudulent were actually fraudulent.

**Recall**

Measures how many actual fraudulent advertisements were successfully detected.

**F1-Score**

Provides a harmonic balance between precision and recall.

The final fraud-class **F1-score of 84.08%** therefore provides a more informative measure of fraud detection performance than accuracy alone.

---

## 🔢 Confusion Matrix — Linear SVM

The Linear SVM confusion matrix on the test set produced:

| | Predicted Legitimate | Predicted Fraudulent |
|---|---:|---:|
| **Actual Legitimate** | 3334 | 22 |
| **Actual Fraudulent** | 31 | 140 |

This corresponds to:

- **3334** legitimate advertisements correctly identified
- **140** fraudulent advertisements correctly detected
- **22** legitimate advertisements incorrectly flagged
- **31** fraudulent advertisements missed

---

## 💾 Model Serialization

After model selection, the trained model and TF-IDF vectorizer are serialized using `joblib`.

```text
models/
├── hireguard_svm_model.pkl
└── hireguard_tfidf_vectorizer.pkl
```

Saving both components ensures that new job advertisements undergo the same feature transformation used during training.

---

# 🏗️ System Architecture

HireGuard AI follows a simple full-stack Machine Learning architecture:

```text
┌─────────────────────────────┐
│        User / Browser       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     HTML / CSS / JavaScript │
│          Frontend           │
└──────────────┬──────────────┘
               │ HTTP POST
               ▼
┌─────────────────────────────┐
│         FastAPI API         │
│       /predict endpoint     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Text Preprocessing    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      TF-IDF Vectorizer      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         Linear SVM          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Prediction + Decision Score │
└─────────────────────────────┘
```

---

# ⚡ FastAPI Backend

The trained ML pipeline is exposed through a REST API built with **FastAPI**.

The backend:

- Loads the serialized Linear SVM model
- Loads the saved TF-IDF vectorizer
- Accepts job advertisement text
- Cleans incoming text
- Vectorizes the advertisement
- Performs inference
- Calculates the SVM decision score
- Returns a structured JSON response

### API Endpoints

```text
GET  /
GET  /health
POST /predict
```

When the backend is running, interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Example API Response — Suspicious Posting

```json
{
  "prediction": "Potentially Fraudulent",
  "decision_score": 0.3793,
  "model": "Linear SVM",
  "disclaimer": "Prediction is a risk indicator and not definitive proof of fraud."
}
```

## Example API Response — Legitimate Posting

```json
{
  "prediction": "Likely Legitimate",
  "decision_score": -0.9572,
  "model": "Linear SVM",
  "disclaimer": "Prediction is a risk indicator and not definitive proof of fraud."
}
```

---

# 💻 Interactive Frontend

HireGuard AI includes a custom frontend built using:

- HTML5
- CSS3
- JavaScript
- Fetch API

Users can paste a job advertisement into the application and receive an ML-generated risk assessment.

The interface displays:

- Prediction
- Decision score
- Classification model
- Risk explanation
- Responsible-use disclaimer
- Backend status

---

## 🛡️ Responsible Prediction Design

HireGuard intentionally uses the labels:

```text
Potentially Fraudulent
```

and

```text
Likely Legitimate
```

rather than presenting predictions as absolute facts.

Machine Learning predictions can be incorrect, and a classification result should not be treated as proof that a company or advertisement is fraudulent.

Users should independently verify:

- Employer identity
- Official company website
- Recruiter identity
- Application source
- Requests for payment
- Requests for sensitive personal information

---

# 🗂️ Project Structure

```text
HireGuard-AI-Recruitment-Fraud-Detection/
│
├── backend/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── fake_job_postings.csv
│   └── processed/
│       └── cleaned_job_postings.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── images/
│
├── models/
│   ├── hireguard_svm_model.pkl
│   └── hireguard_tfidf_vectorizer.pkl
│
├── notebooks/
│   └── HireGuard_AI_Final_Project.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

### Data Science

- Python
- Pandas
- NumPy
- Matplotlib
- NLTK
- WordCloud

### Machine Learning

- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest
- GridSearchCV
- Joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- HTML
- CSS
- JavaScript
- Fetch API

### Development

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd HireGuard-AI-Recruitment-Fraud-Detection
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it using:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents script execution for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running HireGuard AI

HireGuard consists of a backend and frontend, so run them in two terminals.

## Terminal 1 — Start FastAPI

From the project root:

```bash
python -m uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Keep this terminal running.

---

## Terminal 2 — Start the Frontend

### Option A — Python HTTP Server

Move into the frontend directory:

```bash
cd frontend
```

Start the server:

```bash
python -m http.server 5501
```

Open:

```text
http://127.0.0.1:5501
```

### Option B — VS Code Live Server

Open:

```text
frontend/index.html
```

and select:

```text
Open with Live Server
```

The exact Live Server port may vary depending on the local VS Code configuration.

---

# 🧪 Testing the Application

### Suspicious Example

```text
URGENT WORK FROM HOME OPPORTUNITY!

Earn 50000 per week with no experience required.
No interview required.

Candidates must pay a registration and training fee before
receiving the appointment letter.

Contact immediately through WhatsApp.
```

Expected classification:

```text
Potentially Fraudulent
```

### Legitimate Example

```text
Software Engineer Intern

We are looking for a Software Engineering Intern to join our
development team.

Candidates should be pursuing a degree in Computer Science
or a related field.

Required skills include Python, SQL and Git.

Selected candidates will participate in a structured technical
interview process. There are no registration fees or payments
required from applicants.
```

Expected classification:

```text
Likely Legitimate
```

---

# 📓 Data Science Notebook

The complete Data Science workflow is available at:

```text
notebooks/HireGuard_AI_Final_Project.ipynb
```

The notebook documents the complete journey from raw data to the final Machine Learning model, including:

```text
Dataset
   ↓
EDA
   ↓
Data Cleaning
   ↓
NLP Preprocessing
   ↓
TF-IDF
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Evaluation
   ↓
Model Serialization
   ↓
Real-Time Prediction
```

---

# 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.59% | 70.87% | **85.38%** | 77.45% |
| Multinomial Naive Bayes | 96.34% | 65.67% | 51.46% | 57.70% |
| Random Forest | 98.44% | **96.77%** | 70.18% | 81.36% |
| **Linear SVM** | **98.50%** | 86.42% | 81.87% | **84.08%** |

🏆 **Selected Model: Linear SVM**

The final model was selected based on its overall balance across precision, recall and F1-score rather than accuracy alone.

---

# 💡 Key Learnings

This project demonstrates a complete Data Science lifecycle rather than stopping at model training.

Key concepts applied include:

- Real-world dataset exploration
- Missing-value handling
- Imbalanced classification
- NLP preprocessing
- Text vectorization
- Model comparison
- Classification metrics
- Confusion-matrix interpretation
- Hyperparameter tuning
- Model persistence
- REST API development
- Frontend/backend integration
- Real-time Machine Learning inference

---

# ⚠️ Limitations

HireGuard AI is an experimental Machine Learning system and has several limitations.

- Predictions depend on patterns present in the training dataset.
- New fraud strategies may differ from historical examples.
- Text alone cannot verify whether a company legally exists.
- A legitimate advertisement may occasionally be flagged.
- A fraudulent advertisement may occasionally evade detection.
- The decision score is a model output, not a calibrated probability of fraud.

Therefore, HireGuard should be treated as a **risk-screening system**, not an automated final authority.

---

# 🔮 Future Improvements

Potential extensions include:

- Transformer-based NLP models such as BERT
- Explainable AI for highlighting suspicious phrases
- Probability calibration
- URL and domain reputation analysis
- Company verification APIs
- Recruiter email-domain analysis
- Scam-pattern extraction
- User feedback and model retraining
- Database integration
- Authentication
- Cloud deployment
- Automated model monitoring

---

# 🌟 Project Highlights

✔ End-to-end Data Science workflow  
✔ 17K+ processed recruitment advertisements  
✔ Real-world imbalanced classification problem  
✔ NLP + TF-IDF feature engineering  
✔ Four Machine Learning algorithms compared  
✔ Hyperparameter tuning  
✔ 98.50% final test accuracy  
✔ 84.08% fraud-class F1-score  
✔ Serialized production model  
✔ FastAPI REST API  
✔ Interactive Swagger documentation  
✔ Custom responsive frontend  
✔ Real-time ML predictions  
✔ Responsible AI disclaimer  
✔ Portfolio-ready full-stack ML architecture  

---

# 👩‍💻 Author

**Yashika Mohanty**

B.Tech — Computer Science & Engineering  
Data Science

Built as a final end-to-end Data Science project demonstrating the integration of **Data Analysis, Natural Language Processing, Machine Learning, API Development and Frontend Engineering**.

---

## ⭐ Support

If you find HireGuard AI interesting, consider giving the repository a ⭐.

---

### 🛡️ HireGuard AI

**Detect patterns. Assess risk. Apply smarter.**
