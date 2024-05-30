import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
nltk.download('stopwords')

# Load the Passive Aggressive Classifier and TF-IDF Vectorizer
pa_classifier = pickle.load(open('lr.pkl', 'rb'))
tfidf_v = pickle.load(open('tfidfvect.pkl', 'rb'))

# Define a function to preprocess the input text
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Function to predict the label (True/False) of a news headline
def predict_news_headline(headline):
    processed_headline = preprocess_text(headline)
    tfidf_headline = tfidf_v.transform([processed_headline])
    prediction = pa_classifier.predict(tfidf_headline)
    return "True" if prediction[0] == 1 else "False"

# Example usage
news_headline = "Scientists discover new species of dinosaur"
prediction = predict_news_headline(news_headline)
print("Prediction for the headline '{}': {}".format(news_headline, prediction))