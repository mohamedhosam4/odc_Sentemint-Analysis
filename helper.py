from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import pickle
import nltk

tf = pickle.load(open("artifacts/tf.pkl",'rb'))

nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def text_preprocessing(text):
  text = text.lower()
  text = re.sub('[^a-zA-z]', ' ', text)
  text = word_tokenize(text)
  text = [word for word in text if word not in stop_words]
  text = [stemmer.stem(word) for word in text]
  text = ' '.join(text)
  text = tf.transform([text])
  return text
