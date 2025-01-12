from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import pickle
import nltk


tf = pickle.load(open("D:/ODC/projct2/artifacts/tf.pkl",'rb'))


nltk.download('punkt_tab')
nltk.download('stopwords')


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))



### SVM , dt , lr
def text_preprocessing(text):
  ## lower case
  text = text.lower()
  ## special charcter
  text = re.sub('[^a-zA-z]', ' ', text)
  ## Tokinzation
  text = word_tokenize(text)
  ## stopwords
  text = [word for word in text if word not in stop_words]
  ## lemmetization
  text = [stemmer.stem(word) for word in text]
  text = ' '.join(text)
  text = tf.transform([text])
  return text
