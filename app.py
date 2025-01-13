import streamlit as st
import pickle
import nltk
import gdown
import os
import helper

# Download resources if they are not already present
try:
    nltk.data.find('tokenizers/punkt')  # Check if 'punkt' file exists
except LookupError:
    nltk.download('punkt', quiet=True)  # If not, download it

try:
    nltk.data.find('tokenizers/punkt_tab')  # Check if 'punkt_tab' file exists
except LookupError:
    nltk.download('punkt_tab', quiet=True)  # If not, download it

# Download stopwords
nltk.download('stopwords', quiet=True)

# Google Drive model URLs (Make sure these are publicly accessible)
MODEL_URLS = {
    "Logistic Regression": "https://drive.google.com/uc?id=1887AgoAPiU5QxcjAt6WJVC0sXfIbMXl1",
    "SVM": "https://drive.google.com/uc?id=17Y3eoOCUeEaN1CpIMCz3dFuslhafuJWK",
    "Decision Tree": "https://drive.google.com/uc?id=11WV6qGw2KTGDoEzOY_C9oBdBAQi--MH7"
}

# Load the model only once
@st.cache_resource
def load_model(model_name):
    """Download and load the ML model from Google Drive only once."""
    model_path = f"{model_name}.pkl"

    # Download the model if it doesn't already exist
    if not os.path.exists(model_path):
        gdown.download(MODEL_URLS[model_name], model_path, quiet=False)

    # Load the model from the file
    return pickle.load(open(model_path, 'rb'))

# Allow the user to select the desired model
model_choice = st.selectbox("Select the model you want to use:", list(MODEL_URLS.keys()))

# Load the selected model (cached)
model = load_model(model_choice)

# User input for the review text
text = st.text_input("Enter your review:")

# Process the input text
if text:
    processed_text = helper.text_preprocessing(text)  # Preprocess the input text

    # Make a prediction
    if st.button("Predict"):
        try:
            pred = model.predict(processed_text)[0]  # Get the prediction from the model
            
            # Display the appropriate message based on the prediction
            if pred == 1:
                st.success("The review is Positive! 😊")
            else:
                st.error("The review is Negative! 😞")
        except Exception as e:
            st.error(f"Error processing the text: {e}")  # Handle errors gracefully
