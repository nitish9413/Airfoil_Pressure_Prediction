from flask import Flask, render_template, request, jsonify,app,url_for
import pickle
import pandas as pd
import numpy as np
from flask import Response

app = Flask(__name__)
model = pickle.load(open('regressor.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')

# create route to predict_api function post request
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # get data from post request
    data = request.json['data']
    # convert data to list 
    new_list = [list(data.values())]
    # predict dataframe
    prediction = model.predict(new_list)[0]
    # return prediction
    return jsonify(prediction)


@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    #new_data=[list(data.values())]
    output=model.predict(final_features)[0]
    # render template to show prediction home.html text Airfoil pressure is { output }
    return render_template('home.html',prediction_text='Airfoil pressure is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)