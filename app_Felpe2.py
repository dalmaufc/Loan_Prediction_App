import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Load dataset
DATA_FILE = "train_ctrUa4K.csv"
if not os.path.exists(DATA_FILE):
    st.error(f"Error: Missing dataset file `{DATA_FILE}`. Please upload it.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Fix Pandas FutureWarnings (Avoid inplace=True)
df = df.copy()
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df.dropna(inplace=True)

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'No': 0, 'Yes': 1})
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})

# Define features and target
X = df[['Gender', 'Married', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Self_Employed', 'Property_Area']]
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize GridSearchCV for faster training
param_grid = {
    'n_estimators': [50, 100],  # Reduce number of trees
    'max_depth': [3, 5],  # Limit depth to avoid long training time
    'min_samples_split': [2, 5],  # Reduce options
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,  # Reduce folds for speed
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Get best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Optimized Model Accuracy: {accuracy:.2f}")
st.write("Classification Report:", classification_report(y_test, y_pred))
st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# Save the best model
MODEL_FILE = "best_model.pkl"
with open(MODEL_FILE, "wb") as file:
    pickle.dump(best_model, file)
st.write(f"Model saved as `{MODEL_FILE}`")

# Streamlit UI for Loan Prediction
st.title("Loan Approval Prediction")

# Model Selection
model_option = st.selectbox("Choose a model", ["Use Pretrained Model", "Train New Random Forest"])

# Load the best model if available
if model_option == "Use Pretrained Model":
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as file:
            best_model = pickle.load(file)
        st.write("Loaded pretrained model from `best_model.pkl`.")
    else:
        st.error("No pretrained model found. Please train a new model first.")

# User Input Form
st.sidebar.header("Enter Loan Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["No", "Yes"])
income = st.sidebar.number_input("Applicant Income", min_value=1000, max_value=100000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1, max_value=1000, step=1)
credit_history = st.sidebar.selectbox("Credit History", ["Unclear Debts", "Clear Debts"])

# Convert user inputs
gender = 0 if gender == "Male" else 1
married = 0 if married == "No" else 1
credit_history = 0 if credit_history == "Unclear Debts" else 1
loan_amount /= 1000  # Normalize loan amount

# Make prediction
if st.sidebar.button("Predict Loan Approval"):
    user_input = np.array([[gender, married, income, loan_amount, credit_history]])
    prediction = best_model.predict(user_input)
    pred_result = "Approved" if prediction[0] == 1 else "Rejected"
    st.write(f"Loan Prediction: **{pred_result}**")