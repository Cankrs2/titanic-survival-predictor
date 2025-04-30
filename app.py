import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

# Get user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("Number of Parents/Children Aboard", 0, 5, 0)
fare = st.slider("Fare Paid", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Convert sex to numerical
sex = 0 if sex == "male" else 1

# Embarked columns (S is reference since we used drop_first=True)
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create prediction dataframe
data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_Q': [embarked_Q],
    'Embarked_S': [embarked_S]
})

if st.button("Predict"):
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.success("🛟 This person would likely survive!")
    else:
        st.error("💀 Unfortunately, this person would likely not survive.")
