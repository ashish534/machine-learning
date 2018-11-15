from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np
import pickle 

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	dataset = pd.read_csv("data/Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
	# Features and Labels
	print(dataset)
    
	corpus = []
	for i in range(0, 1000):
		review = re.sub('[^a-zA-Z]', ' ', dataset.Review[i])
		review = review.lower()
		review = review.split()
		review = ' '.join(review)
		corpus.append(review)

    # Vectorization
	cv = CountVectorizer(max_features = 1500)
	X = cv.fit_transform(corpus).toarray() 
	
	print(X)
	# Loading our ML Model
	naivebayes_model = open("models/final_model.sav", "rb")
	modell =  pickle.load(naivebayes_model)

	# Receives the input query from form
	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [namequery]
		cps = []
		review = re.sub('[^a-zA-Z]', ' ', data[0])
		review = review.lower()
		review = review.split()
		ps = PorterStemmer()
		review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		cps.append(review)

		vect = cv.transform(cps).toarray()
		my_prediction = modell.predict(vect)
	return render_template('results.html',prediction = my_prediction,name = namequery.upper())


if __name__ == '__main__':
	app.run(debug=True)