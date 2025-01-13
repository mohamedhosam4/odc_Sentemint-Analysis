import streamlit as st
import pickle
import nltk
import gdown
import os
import helper



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


#nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Google Drive model links (Change these to your actual Google Drive file IDs)
MODEL_URLS = {
    "Logistic Regression": "https://drive.google.com/uc?id=1887AgoAPiU5QxcjAt6WJVC0sXfIbMXl1",
    "SVM": "https://drive.google.com/uc?id=17Y3eoOCUeEaN1CpIMCz3dFuslhafuJWK",
    "Decision Tree": "https://drive.google.com/uc?id=11WV6qGw2KTGDoEzOY_C9oBdBAQi--MH7"
}

@st.cache_resource
def load_model(model_name):
    """Download and load the ML model from Google Drive only once."""
    model_path = f"{model_name}.pkl"

    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        gdown.download(MODEL_URLS[model_name], model_path, quiet=False)

    # Load model from file
    return pickle.load(open(model_path, 'rb'))

# Allow user to select the desired model
model_choice = st.selectbox("Select the model you want to use:", list(MODEL_URLS.keys()))

# Load the selected model (cached)
model = load_model(model_choice)

# User input for the review text
text = st.text_input("Enter your review:")

# Process the input text
if text:
    processed_text = helper.text_preprocessing(text)

    # Make a prediction
    if st.button("Predict"):
        pred = model.predict(processed_text)[0]
        
        # Display appropriate message based on the prediction
        if pred == 1:
            st.success("The review is Positive! 😊")
        else:
            st.error("The review is Negative! 😞")
