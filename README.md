# Week 2: Supervised Machine Learning Models
**Student:** Binnala Esikha

## Objective
Build supervised learning models with scikit-learn and compare their performance.

## Files
- `week2_ml_models.py` — trains regression and classification models.
- `student_performance.csv` — small illustrative sample dataset.
- `Week2_Report.docx` — report describing the workflow and results.
- `requirements.txt` — Python dependencies.

## Models used
1. Linear Regression — predicts a numeric final score.
2. Random Forest Regressor — predicts a numeric final score.
3. Logistic Regression — predicts a category (`Pass` or `Needs Improvement`).

## Dataset features
`study_hours`, `attendance_percent`, and `previous_score` are used as inputs.
`final_score` is the regression target. `result` is the classification target.

## Run
Install Python 3, then run:
```bash
pip install -r requirements.txt
python week2_ml_models.py
```

The sample dataset is fictional and is used only for learning. Results from a small
dataset are demonstrations, not a reliable way to judge real students.
