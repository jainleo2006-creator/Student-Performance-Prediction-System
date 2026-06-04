import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

LABEL_MAPPING = {"Fail": 0, "Pass": 1}
INVERSE_LABEL = {0: "Fail", 1: "Pass"}


def load_data(csv_path: str = "student_data.csv") -> pd.DataFrame:
    """Load the student dataset from CSV."""
    return pd.read_csv(csv_path)


def get_dataframe_info(df: pd.DataFrame) -> str:
    """Return the DataFrame info summary as a text string."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def preprocess_data(df: pd.DataFrame):
    """Clean, encode, and split the dataset for training."""
    df_clean = df.copy()

    # Handle numeric missing values by median imputation
    for column in ["Study_Hours", "Attendance", "Previous_Score"]:
        if df_clean[column].isnull().any():
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())

    # Ensure label column has no missing values
    df_clean["Result"] = df_clean["Result"].fillna("Fail")
    df_clean["Target"] = df_clean["Result"].map(LABEL_MAPPING)

    X = df_clean[["Study_Hours", "Attendance", "Previous_Score"]]
    y = df_clean["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, df_clean


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train logistic regression and random forest models."""
    logistic_model = LogisticRegression(max_iter=250, random_state=42)
    random_forest_model = RandomForestClassifier(n_estimators=120, random_state=42)

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Fail", "Pass"],
            zero_division=0,
        ),
    }


def compare_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compare models and choose the best performer."""
    results = {}
    for name, model in models.items():
        evaluation = evaluate_model(model, X_test, y_test)
        results[name] = evaluation

    best_model_name = max(results, key=lambda m: results[m]["accuracy"])
    best_accuracy = results[best_model_name]["accuracy"]

    return {
        "results": results,
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
    }


def plot_feature_relationships(df: pd.DataFrame) -> plt.Figure:
    """Create exploratory plots to visualize the dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(
        data=df,
        x="Study_Hours",
        hue="Result",
        multiple="stack",
        palette=["#FF6B6B", "#4ECDC4"],
        ax=axes[0],
    )
    axes[0].set_title("Study Hours Distribution by Result")
    axes[0].set_xlabel("Study Hours")

    sns.scatterplot(
        data=df,
        x="Attendance",
        y="Previous_Score",
        hue="Result",
        palette=["#FF6B6B", "#4ECDC4"],
        ax=axes[1],
    )
    axes[1].set_title("Attendance vs Previous Score")
    axes[1].set_xlabel("Attendance (%)")
    axes[1].set_ylabel("Previous Score")

    correlation = df[["Study_Hours", "Attendance", "Previous_Score"]].corr()
    sns.heatmap(
        correlation,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.6,
        ax=axes[2],
    )
    axes[2].set_title("Feature Correlation Matrix")

    fig.tight_layout()
    return fig


def predict_student_performance(model, study_hours: float, attendance: float, previous_score: float) -> str:
    """Predict whether a student will pass or fail."""
    features = np.array([[study_hours, attendance, previous_score]])
    prediction = model.predict(features)[0]
    return INVERSE_LABEL[int(prediction)]


def build_best_model(csv_path: str = "student_data.csv") -> dict:
    """Load data, preprocess, train models, and return the best model with metrics."""
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, df_clean = preprocess_data(df)
    models = train_models(X_train, y_train)
    comparison = compare_models(models, X_test, y_test)
    comparison["X_test"] = X_test
    comparison["y_test"] = y_test
    comparison["dataframe"] = df_clean
    comparison["models"] = models
    return comparison


if __name__ == "__main__":
    summary = build_best_model()
    print("Best model:", summary["best_model"])
    print("Best accuracy:", f"{summary['best_accuracy']:.2%}")
    print("Model performance details:")
    for name, output in summary["results"].items():
        print(f"\n{name}")
        print(f"Accuracy: {output['accuracy']:.2%}")
        print(output["classification_report"])
