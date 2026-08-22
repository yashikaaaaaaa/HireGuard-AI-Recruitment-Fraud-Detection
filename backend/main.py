from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "hireguard_svm_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "hireguard_tfidf_vectorizer.pkl"


# ============================================================
# LOAD MACHINE LEARNING ARTIFACTS
# ============================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="HireGuard AI API",
    description="NLP-powered recruitment fraud detection API",
    version="1.0.0"
)


# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class JobPosting(BaseModel):
    text: str


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text: str) -> str:

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Keep alphabetic characters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "application": "HireGuard AI",
        "status": "online",
        "model": "Linear SVM",
        "message": "Recruitment fraud detection API is running."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "vectorizer_loaded": True
    }


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict_job(job: JobPosting):

    if not job.text.strip():

        return {
            "prediction": "Invalid Input",
            "decision_score": None,
            "message": "Please provide job advertisement text."
        }

    cleaned_text = clean_text(
        job.text
    )

    vectorized_text = vectorizer.transform(
        [cleaned_text]
    )

    prediction = int(
        model.predict(vectorized_text)[0]
    )

    decision_score = float(
        model.decision_function(vectorized_text)[0]
    )

    if prediction == 1:
        label = "Potentially Fraudulent"
    else:
        label = "Likely Legitimate"

    return {
        "prediction": label,
        "decision_score": round(decision_score, 4),
        "model": "Linear SVM",
        "disclaimer": (
            "Prediction is a risk indicator and not "
            "definitive proof of fraud."
        )
    }