from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import pickle
import nltk
import gdown  # Import gdown for downloading from Google Drive

# Google Drive link for the model file (make sure it's publicly accessible)
MODEL_URL = "https://drive.google.com/uc?id=1rvMcc2L7typfaGvKoZshfNninLXi0wKl"
MODEL_PATH = "tf.pkl"

# Download the model from Google Drive
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the downloaded model
tf = pickle.load(open(MODEL_PATH, 'rb'))

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Initialize text processing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def text_preprocessing(text):
    """
    Preprocesses the input text by converting to lowercase, removing non-alphabetic characters,
    tokenizing, removing stopwords, applying stemming, and transforming it using the TF-IDF model.
    """
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only alphabets
    text = word_tokenize(text)  # Tokenize words
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    text = [stemmer.stem(word) for word in text]  # Apply stemming
    text = ' '.join(text)
    text = tf.transform([text])  # Convert text into numerical representation
    return text
