import helper
import streamlit as st 
import pickle

# Load models
models = {
    "Logistic Regression": pickle.load(open("artifacts/lr.pkl", 'rb')),
    "SVM": pickle.load(open("artifacts/svc.pkl", 'rb')),
    "Decision Tree": pickle.load(open("artifacts/dt.pkl", 'rb'))
}

# Allow user to select the desired model
model_choice = st.selectbox("Select the model you want to use:", list(models.keys()))
model = models[model_choice]

# User input for the review text
text = st.text_input('Enter your review:')

# Process the input text
if text:
    processed_text = helper.text_preprocessing(text)

    # Make a prediction
    if st.button('Predict'):
        pred = model.predict(processed_text)[0]
        
        # Display appropriate message based on the prediction
        if pred == 1:
            st.success("The review is Positive!")
        else:
            st.error("The review is Negative!")
