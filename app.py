import streamlit as st
import pandas as pd
from model import (
    load_data,
    get_dataframe_info,
    preprocess_data,
    train_models,
    compare_models,
    plot_feature_relationships,
    predict_student_performance,
)

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


def load_project_data():
    """Load dataset once for the Streamlit app."""
    return load_data("student_data.csv")


def show_overview():
    st.header("Student Performance Prediction System")
    st.markdown(
        """
        This project predicts whether a student will pass or fail based on study hours,
        attendance, and previous exam scores.
        """
    )
    st.write(
        "Use the sidebar to review data exploration, model evaluation, and make predictions with the deployed model."
    )
    st.markdown("**Core prediction features:**")
    st.write("- Study Hours")
    st.write("- Attendance Percentage")
    st.write("- Previous Exam Scores")


def show_data_exploration(df: pd.DataFrame):
    st.header("Data Loading and Exploration")
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10))

    st.subheader("Dataset Information")
    st.text(get_dataframe_info(df))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().transpose())

    st.subheader("Data Visualizations")
    st.pyplot(plot_feature_relationships(df))


def show_model_training(df: pd.DataFrame):
    st.header("Model Training and Evaluation")
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    models = train_models(X_train, y_train)
    comparison = compare_models(models, X_test, y_test)

    st.subheader("Model Accuracy Comparison")
    accuracy_data = {
        name: metrics["accuracy"]
        for name, metrics in comparison["results"].items()
    }
    st.write(pd.DataFrame.from_dict(accuracy_data, orient="index", columns=["Accuracy"]).style.format("{:.2%}"))

    st.success(f"Best model: {comparison['best_model']} ({comparison['best_accuracy']:.2%})")

    best_metrics = comparison["results"][comparison["best_model"]]
    st.subheader("Best Model Evaluation")
    st.write("**Accuracy Score**: ", f"{best_metrics['accuracy']:.2%}")
    st.write("**Confusion Matrix**")
    st.write(best_metrics["confusion_matrix"])
    st.write("**Classification Report**")
    st.text(best_metrics["classification_report"])


def show_prediction_system(df: pd.DataFrame):
    st.header("Prediction System")
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    models = train_models(X_train, y_train)
    comparison = compare_models(models, X_test, y_test)
    best_model = models[comparison["best_model"]]

    st.write(
        "Enter new student details below and click **Predict** to see whether the student is likely to pass or fail."
    )

    with st.form("prediction_form"):
        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
        )
        attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=75)
        previous_score = st.number_input(
            "Previous Exam Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.5,
        )
        submitted = st.form_submit_button("Predict")

        if submitted:
            if study_hours < 0 or attendance < 0 or attendance > 100 or previous_score < 0 or previous_score > 100:
                st.error("Please enter valid values for all fields.")
            else:
                prediction = predict_student_performance(
                    best_model, study_hours, attendance, previous_score
                )
                if prediction == "Pass":
                    st.success(f"The student is predicted to: **{prediction}**")
                else:
                    st.error(f"The student is predicted to: **{prediction}**")

                st.markdown("---")
                st.write(f"**Model used:** {comparison['best_model']}")
                st.write(f"**Model accuracy:** {comparison['best_accuracy']:.2%}")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Overview", "Data Exploration", "Model Training", "Prediction System"],
    )

    df = load_project_data()

    if page == "Overview":
        show_overview()
    elif page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Model Training":
        show_model_training(df)
    elif page == "Prediction System":
        show_prediction_system(df)

    st.sidebar.markdown("---")
    st.sidebar.write("Built with Python, pandas, scikit-learn, Matplotlib, Seaborn, and Streamlit.")


if __name__ == "__main__":
    main()
