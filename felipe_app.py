import streamlit as st
import joblib

# Load the trained model
model_path = "classifier.pkl"
classifier = joblib.load(model_path)

@st.cache_data()
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
    # Pre-processing user input    
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1  
    
    # Ensure consistency with model training (No division by 1000)
    LoanAmount = LoanAmount  
    
    # Making predictions 
    prediction = classifier.predict([[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
    return 'Approved' if prediction == 1 else 'Rejected'

# Main Streamlit app
def main():       
    # Frontend design
    html_temp = """ 
    <div style="background-color:yellow;padding:13px"> 
    <h1 style="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True) 
    
    # User input fields
    Gender = st.selectbox('Gender',("Male", "Female"))
    Married = st.selectbox('Marital Status',("Unmarried", "Married")) 
    ApplicantIncome = st.number_input("Applicant's Monthly Income", min_value=0)
    LoanAmount = st.number_input("Total Loan Amount", min_value=0)
    Credit_History = st.selectbox('Credit History',("Unclear Debts", "No Unclear Debts"))
    
    result = ""
    
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success(f'Your loan is {result}')
     
if __name__ == '__main__': 
    main()
