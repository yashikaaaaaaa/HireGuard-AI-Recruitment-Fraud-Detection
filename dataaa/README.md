# 📊 HireGuard AI Dataset

HireGuard AI uses the **Real / Fake Job Posting Prediction** dataset for training and evaluating the recruitment fraud detection model.

## Dataset Overview

The original dataset contains approximately 18,000 job advertisements with both legitimate and fraudulent examples.

After preprocessing, the dataset used in this project contains:

| Category | Count |
|---|---:|
| Total Job Postings | 17,632 |
| Legitimate | 16,776 |
| Fraudulent | 856 |
| Fraud Rate | 4.85% |

## Dataset Source

**Kaggle — Real / Fake Job Posting Prediction**

Dataset author: Shivam Bansal

https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Local Setup

The complete dataset is not included in this repository because of GitHub browser file-size limitations.

Download `fake_job_postings.csv` from the Kaggle dataset and create the following structure:

data/
├── raw/
│   └── fake_job_postings.csv
└── processed/
    └── cleaned_job_postings.csv

The processed dataset can be generated through the project's Jupyter Notebook.

## Note

The trained HireGuard AI model and TF-IDF vectorizer are already available in the `models/` directory for inference.

The raw dataset is required only when reproducing the complete training and Data Science workflow.
