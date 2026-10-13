"""
Yuva Intern Week 2: Supervised Machine Learning Models
Student: Binnala Esikha

This educational example trains regression and classification models on a small
sample student-performance dataset. The dataset is illustrative, not real student data.
Install dependencies: pip install pandas scikit-learn
Run: python week2_ml_models.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_FILE = "student_performance.csv"

def main():
    df = pd.read_csv(DATA_FILE)
    print("Dataset preview:")
    print(df.head())
    print("\\nDataset shape:", df.shape)
    print("\\nMissing values:")
    print(df.isnull().sum())

    # Regression: predict final score from study hours, attendance and previous score.
    features = ["study_hours", "attendance_percent", "previous_score"]
    X = df[features]
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        )
    }
    print("\\nREGRESSION MODEL COMPARISON")
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"{name}: MAE={mean_absolute_error(y_test, predictions):.2f}, "
              f"R2={r2_score(y_test, predictions):.3f}")

    # Classification: predict the result category (Pass / Needs Improvement).
    y_class = df["result"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.25, random_state=42, stratify=y_class
    )
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    classifier.fit(X_train_c, y_train_c)
    class_predictions = classifier.predict(X_test_c)
    print("\\nCLASSIFICATION MODEL")
    print("Logistic Regression accuracy:",
          f"{accuracy_score(y_test_c, class_predictions):.2%}")
    print(classification_report(y_test_c, class_predictions, zero_division=0))

    # Example prediction for a hypothetical student.
    example = pd.DataFrame([{
        "study_hours": 5,
        "attendance_percent": 80,
        "previous_score": 60
    }])
    print("\\nExample predicted score (Linear Regression):",
          f"{regressors['Linear Regression'].predict(example)[0]:.1f}")

if __name__ == "__main__":
    main()
