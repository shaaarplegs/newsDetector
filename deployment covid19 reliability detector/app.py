import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__,template_folder='template')
model = pickle.load(open('newsDetector_covid19tweets.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    textfeature = [x for x in request.form.values()]
    final_features =  pd.Series([np.array(textfeature)][0][0])

    output = model.predict(final_features)[0]

    return render_template('home.html', prediction_text='The news is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
