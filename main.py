from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from wide_resnet import WideResNet

app = Flask(__name__)

MODEL_PATH = 'models/xxx.hdf5'

img_size = 64
model = load_model(MODEL_PATH)
model = WideResNet(img_size, depth=16, k=8)()
model.load_weights(MODEL_PATH)
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, 0)

    # predict
    results = model.predict(img)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    predicted_ages = str(int(predicted_ages.item(0)))
    if predicted_genders.item(0) > 0.5:
        gender_ = "F"
    else:
        gender_ = "M"
    preds = "Age: " + predicted_ages + " Gender: " +  gender_
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        #Save in server
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        result = preds
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    #Server
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()