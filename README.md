# Personal Expense Anomaly Detector

A machine learning system that detects unusual spending patterns in personal transaction data using **K-Means clustering** and **Isolation Forest** — surfacing potential budget overruns or fraudulent charges automatically.

Built as a BYOP (Bring Your Own Project) capstone for an AI/ML course.

# Problem Statement

Most people have no easy way to know when a transaction is genuinely unusual versus just a slightly higher-than-average bill. This project automates that judgement: given a history of personal expenses, the model learns what "normal" looks like for each spending category and flags transactions that deviate significantly — even without having been trained on labelled fraud examples.

# Demo
streamlit run app.py


The interactive dashboard lets you:
- Upload your own CSV or generate synthetic data
- Tune model sensitivity with sliders
- Explore flagged anomalies with explanations
- Compare K-Means vs Isolation Forest vs ensemble accuracy

# How It Works

Raw transactions
      │
      ▼
Feature Engineering          ← amount, category, time-of-day, weekend flag,
      │                         deviation from category mean
      ▼
K-Means Clustering           ← learns N cluster centroids (one per spending type)
      │                         flags points far from any centroid
      ▼
Isolation Forest             ← explicitly models anomalies as easy-to-isolate points
      │                         in random decision trees
      ▼
Ensemble Vote                ← transaction flagged only when BOTH methods agree
      │                         (reduces false positives)
      ▼
Flagged Anomalies + Dashboard
```

---

## Project Structure

```
expense-anomaly-detector/
├── data/
│   └── generate_data.py     # Synthetic transaction generator with seeded anomalies
├── src/
│   ├── preprocess.py        # Feature engineering pipeline
│   └── model.py             # ExpenseAnomalyDetector class (K-Means + IForest)
├── app.py                   # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

git clone https://github.com/nafees25bai10390/Digital-Literacy-Project
cd expense-anomaly-detector


# 2. Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate


# 3. Install dependencies

pip install -r requirements.txt


# 4. Run the app

streamlit run app.py


The dashboard opens at `http://localhost:8501`.

# Using Your Own Data

Upload a CSV file from the sidebar with these columns:

| Column     | Type     | Example              |
|------------|----------|----------------------|
| `date`     | datetime | `2024-03-15 14:32`   |
| `category` | string   | `Groceries`          |
| `amount`   | float    | `47.80`              |

An optional `is_anomaly` boolean column enables the Evaluation tab with precision/recall metrics.

# ML Concepts Applied

| Concept               | Where Used                                  |
|-----------------------|---------------------------------------------|
| Unsupervised Learning | K-Means clustering of spending patterns     |
| Anomaly Detection     | Isolation Forest for outlier scoring        |
| Feature Engineering   | Temporal features, z-score deviation        |
| Dimensionality Reduction | PCA for 2D cluster visualisation         |
| Ensemble Methods      | AND-vote across two independent detectors   |
| Model Evaluation      | Precision, recall, F1, confusion matrix     |


# Results (on synthetic data)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~98%   |
| Recall    | ~75%   |
| Precision | ~55%   |

The ensemble approach achieves high precision (few false alarms) at the cost of some recall — a deliberate trade-off, since false positives in a spending alert system erode user trust.

# Key Design Decisions

Why two models?
Each model has different failure modes. K-Means misses anomalies that happen to land inside a cluster. Isolation Forest can over-flag dense normal regions. Requiring both to agree cuts false positives significantly.

Why synthetic data?
Real personal finance data is sensitive and hard to share. Synthetic data with seeded anomalies lets anyone run and verify the full pipeline without privacy concerns.

Why Streamlit?
Streamlit lets a data science project ship a real, interactive UI in the same Python codebase without front-end overhead — ideal for a capstone demo.

# Dependencies

- `pandas` — data manipulation
- `numpy` — numerical operations
- `scikit-learn` — K-Means, Isolation Forest, preprocessing, evaluation
- `streamlit` — interactive dashboard
- `plotly` — interactive charts


