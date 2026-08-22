import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("Generating synthetic dataset...")

np.random.seed(42)
n_samples = 1000

data = {
    'StudentID': np.random.randint(1000, 9999, n_samples),
    'Age': np.random.randint(15, 23, n_samples),
    'Gender': np.random.randint(0, 2, n_samples),
    'ParentalEducation': np.random.randint(0, 5, n_samples),
    'StudyTimeWeekly': np.random.uniform(0, 40, n_samples),
    'Absences': np.random.randint(0, 31, n_samples),
    'Tutoring': np.random.randint(0, 2, n_samples),
    'ParentalSupport': np.random.randint(0, 5, n_samples),
    'Extracurricular': np.random.randint(0, 2, n_samples),
    'Sports': np.random.randint(0, 2, n_samples),
    'Volunteering': np.random.randint(0, 2, n_samples),
    'Midterm_Score': np.random.uniform(30, 100, n_samples), # 🆕 নতুন ফিচার
    'Quiz_Score': np.random.uniform(30, 100, n_samples)     # 🆕 নতুন ফিচার
}

df = pd.DataFrame(data)


performance_score = (df['Midterm_Score'] * 0.4) + (df['Quiz_Score'] * 0.4) + (df['StudyTimeWeekly'] * 0.5) - (df['Absences'] * 1.5)
df['Result'] = np.where(performance_score > 55, 1, 0) # 1 = Pass, 0 = Fail


features = [
    'StudentID', 'Age', 'Gender', 'ParentalEducation', 
    'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
    'Extracurricular', 'Sports', 'Volunteering', 
    'Midterm_Score', 'Quiz_Score'
]

X = df[features]
y = df['Result']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Machine Learning Model...")

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")


joblib.dump(model, "student_pass_fail_model.pkl")
joblib.dump({"features": features}, "feature_info.pkl")

print("✅ Success! 'student_pass_fail_model.pkl' and 'feature_info.pkl' created.")