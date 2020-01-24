from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from mlxtend.preprocessing import DenseTransformer



app = Flask(__name__)
Bootstrap(app)


@app.route('/')

def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
	new_features = pd.DataFrame([[39, "State-gov",  "Bachelors", "Never-married", "Adm-clerical", "Black", "Female", 40, "United-States"]])

	dataset = "census"
	classifier_name = "Naive Bayes"
	filename1 = "models/" + dataset + "/" + classifier_name + ".sav"

	pipe = pickle.load(open(filename1, 'rb'))
	print(pipe)
	prediction = pipe.predict(new_features)
	print("FODA SE")
	print(prediction)

	# Receives the input query from form
	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [namequery]

		print(request.form['age'])
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction,name = "fock")


if __name__ == '__main__':
	app.run(debug=True)
