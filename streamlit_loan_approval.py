import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
  
# loading in the model to predict on the data
pickle_in = open('loan_approval.pkl', 'rb')
classifier = pickle.load(pickle_in)

df = pd.read_excel('Rocket_Loans.xlsx')
df.drop('Loan_ID',axis=1,inplace=True)

def welcome():
    return 'welcome all'

def input_conversion(Sex,Age,Married,No_People,Qualification,Self_Employed,Loan_Bearer_Income,Loan_Cobearer_Income,Amount,Loan_Tenure,Credit_Score,Location_type):
    if Sex == 'Male':
        sex_op = 1
    else:
        sex_op = 0
    
    age_op = (float(Age) - df['Age'].mean())/df['Age'].std()
    
    if Married == 'Yes':
        married_op = 1
    else:
        married_op = 0
    
    if No_People > 3:
        no_people_op = 3
    else:
        no_people_op = No_People
    
    if Qualification == 'Not Graduate':
        qual_op = 1
    else:
        qual_op = 0
    
    if Self_Employed == 'Yes':
        selfemp_op = 1
    else:
        selfemp_op = 0
        
    lbi_op = (float(Loan_Bearer_Income) - df['Loan_Bearer_Income'].mean())/df['Loan_Bearer_Income'].std()
    lci_op = (float(Loan_Cobearer_Income) - df['Loan_Cobearer_Income'].mean())/df['Loan_Cobearer_Income'].std()
    amount_op = (float(Amount) - df['Amount Disbursed'].mean())/df['Amount Disbursed'].std()
    
    if Loan_Tenure == 12:
        loan_ten_op = 1
    elif Loan_Tenure == 36:
        loan_ten_op = 2
    elif Loan_Tenure == 60:
        loan_ten_op = 3
    elif Loan_Tenure == 84:
        loan_ten_op = 4
    elif Loan_Tenure == 120:
        loan_ten_op = 5
    elif Loan_Tenure == 180:
        loan_ten_op = 6
    elif Loan_Tenure == 240:
        loan_ten_op = 7
    elif Loan_Tenure == 300:
        loan_ten_op = 8
    elif Loan_Tenure == 360:
        loan_ten_op = 9
    elif Loan_Tenure == 480:
        loan_ten_op = 10
    else:
        loan_ten_op = 11
    
    cred_op = Credit_Score
    
    if Location_type == 'Rural':
        loc_op = 0
    elif Location_type == 'Urban':
        loc_op = 2
    else:
        loc_op = 1
        
    return (sex_op,age_op,married_op,no_people_op,qual_op,selfemp_op,lbi_op,lci_op,amount_op,loan_ten_op,cred_op,loc_op)


# defining the function which will make the prediction using the data which the user inputs
def prediction(sex_op,age_op,married_op,no_people_op,qual_op,selfemp_op,lbi_op,lci_op,amount_op,loan_ten_opo,cred_op,loc_op):  
   
    prediction = classifier.predict([[sex_op,age_op,married_op,no_people_op,qual_op,selfemp_op,lbi_op,lci_op,amount_op,loan_ten_opo,cred_op,loc_op]]) 

    return prediction[0]
      
  
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Loan Approval Decision Maker")
          
    # the following lines create text boxes in which the user can enter the data required to make the prediction
    Sex = st.selectbox("Sex", ("Male","Female"))
    Age = st.number_input("Age")
    Married = st.selectbox("Married", ("Yes","No"))
    No_People = st.number_input("No of people")
    Qualification = st.selectbox("Qualification", ("Graduate","Not Graduate"))
    Self_Employed = st.selectbox("Self Employed", ("Yes","No"))
    Loan_Bearer_Income = st.number_input("Loan Bearer Income")
    Loan_Cobearer_Income = st.number_input("Loan Cobearer Income")
    Amount = st.number_input("Amount disbursed")
    Loan_Tenure = st.selectbox("Loan Tenure", (12,36,60,84,120,180,240,300,360,480))
    Credit_Score = st.selectbox("Credit Score", (0,1))
    Location_type = st.selectbox("Location_type", ("Rural","Urban","Semiurban"))
    result =''
    
    sex_op,age_op,married_op,no_people_op,qual_op,selfemp_op,lbi_op,lci_op,amount_op,loan_ten_opo,cred_op,loc_op = input_conversion(Sex,Age,Married,No_People,Qualification,Self_Employed,Loan_Bearer_Income,Loan_Cobearer_Income,Amount,Loan_Tenure,Credit_Score,Location_type)
    
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(sex_op,age_op,married_op,no_people_op,qual_op,selfemp_op,lbi_op,lci_op,amount_op,loan_ten_opo,cred_op,loc_op) 
    if result == 0:
        st.error('NO')
    else:
        st.success('YES')
     
if __name__=='__main__':
    main()