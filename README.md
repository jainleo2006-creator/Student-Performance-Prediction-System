# Student Performance Prediction System

## Project Overview
This project predicts whether a student will **Pass** or **Fail** based on:
- Study Hours
- Attendance Percentage
- Previous Exam Scores

The system uses a sample dataset, performs data exploration, preprocesses data, trains machine learning models, and offers an interactive Streamlit UI for predictions.

## Project Structure
- `student_data.csv` - Sample dataset with 100 student records
- `app.py` - Streamlit application for interaction and prediction
- `model.py` - Data loading, preprocessing, training, and evaluation logic
- `requirements.txt` - Python dependencies
- `.gitignore` - Files and folders excluded from Git
- `README.md` - Project documentation

## Technologies Used
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How It Works
1. Loads the dataset with Pandas.
2. Displays dataset information, missing values, and summary statistics.
3. Visualizes student performance using Matplotlib and Seaborn.
4. Preprocesses data by handling missing values and encoding labels.
5. Splits the dataset into training and test sets.
6. Trains two models:
   - Logistic Regression
   - Random Forest
7. Evaluates both models and selects the best one.
8. Provides a user-facing prediction system via Streamlit.

## Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report

## Streamlit Features
- Modern sidebar navigation
- Input form for student details
- Prediction button
- Success/error message display
- Model accuracy information

## GitHub Integration
Use these commands to initialize and manage your repository:
```bash
git init
git add .
git commit -m "Initial commit: Student Performance Prediction System"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

### Branching and Collaboration Workflow
1. Create a branch for each feature or fix:
   ```bash
git checkout -b feature/prediction-ui
```
2. Make your changes and commit them.
3. Push the branch to the remote repository:
   ```bash
git push -u origin feature/prediction-ui
```
4. Open a pull request for review.
5. After review, merge into `main`.

This workflow helps teams work in parallel and keeps the main branch stable.

## Future Improvements
- Add model persistence with `joblib` or `pickle`.
- Enable user-uploaded CSV files for custom predictions.
- Explore additional algorithms like XGBoost or SVM.
- Add cross-validation and hyperparameter tuning.
- Improve UI layout with charts and KPI cards.

## Notes
The dataset in `student_data.csv` is synthetic and intended for demonstration and learning purposes.
