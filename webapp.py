import os
import numpy as np
import cv2
import keras
from keras.models import load_model as load
from keras import backend
from flask import Flask, flash
from flask import request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpeg', 'jpg', 'png']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No File.')
			
			return redirect(request.url)

		file = request.files['file']
		if file.filename == '':
			flash('Please Select a File.')
			
			return redirect(request.url)

		if file and allowed_file(file.filename):
			name = secure_filename(file.filename)
			file.save(os.path.join('./uploads/', name))
			photo = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/" + name)

			main_color = mainColor(photo)
			animal = classify(photo)

			redirect(url_for('upload_file',filename=name))
			
			return '''
			<!doctype html>
			<body style="background-color:#9fdf9f;">
			<title>CatDogs</title>
			<h1 style="color: #FFFFFF;font-family: Ubuntu;">There is a <u>'''+animal+'''</u> in the photo that is mainly <u>'''+main_color+'''</u>.</h1>
			<form style="  background-color: #4CAF50; 
				  border: none;
				  color: white;
				  padding: 15px 32px;
				  text-align: center;
				  text-decoration: none;
				  display: inline-block;
				  font-size: 16px;" method=post enctype=multipart/form-data>
	  		<input type=file name=file>
	  		<input style="  background-color: #808080; 
				  border: none;
				  color: white;
				  padding: 15px 32px;
				  text-align: center;
				  text-decoration: none;
				  display: inline-block;
				  font-size: 16px;"
				  type=submit value=Upload>
			</form>
			'''
	return '''
	<!doctype html>
	<body style="background-color:#9fdf9f;">
	<title >CatDogs</title>
	<h1 style="color: #FFFFFF;font-family: Ubuntu;">Upload a Portrait of a Dog or a Cat.</h1>
	<h2 style="color: #FFFFFF;font-family: Ubuntu;"> We'll tell you what animal it is and if it is predominantly green, blue or red.</h2>	
	<form style="  background-color: #4CAF50; 
			  border: none;
			  color: white;
			  padding: 15px 32px;
			  text-align: center;
			  text-decoration: none;
			  display: inline-block;
			  font-size: 16px;" method=post enctype=multipart/form-data>
	  <input type=file name=file>
	  <input style="  background-color: #808080; 
			  border: none;
			  color: white;
			  padding: 15px 32px;
			  text-align: center;
			  text-decoration: none;
			  display: inline-block;
			  font-size: 16px;"
			  type=submit value=Upload>
	</form>
	'''

def mainColor(photo):
	B, G, R = cv2.split(photo)
	sums = [np.sum(B), np.sum(G), np.sum(R)]
	values = {"0": "Blue", "1":"Green", "2": "Red"}
	main_color = values[str(np.argmax(sums))]
	return main_color

def classify(photo):
	clf = load('./model.h5')
	photo = cv2.resize(photo, (150,150), interpolation = cv2.INTER_AREA).reshape(1,150,150,3) 
	
	if str(clf.predict_classes(photo, 1, verbose = 0)[0][0]) == "0":
		answer = "Cat"

	else:
		answer = "Dog"

	backend.clear_session()
	
	return answer

	
if __name__ == "__main__":
	app.run(host= '127.0.0.1', port=5000)


