from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__ , template_folder='templates')
CORS(app)

# Load the Passive Aggressive Classifier and TF-IDF Vectorizer
pa_classifier = pickle.load(open('pa.pkl', 'rb'))
tfidf_v = pickle.load(open('tfidfvect.pkl', 'rb'))

# Preprocess function
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    headline = data['headline']
    processed_headline = preprocess_text(headline)
    tfidf_headline = tfidf_v.transform([processed_headline])
    prediction = pa_classifier.predict(tfidf_headline)
    result = "True" if prediction[0] == 1 else "False"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
