import numpy as np
#need to 'conda install flask' for this to work
from flask import Flask, abort, jsonify, request, Response
import jsonpickle
from keras.models import model_from_json, Model, Sequential
import cPickle as pickle
import cv2
import csv, sqlite3
import math
import os
import random

from keras import applications, optimizers
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.utils import np_utils

##MODEL STUFF UP TOP, FLASK STUFF BELOW

##MODEL PARAMETERS AND SAVED WEIGHTS
# If you want to specify input tensor
img_rows = 100
img_cols = 100

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

# Custom Optimizer
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=1e-6)

# Do not forget to compile it
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
model.load_weights("/home/roiana_reid/Imagion_old/model_regress.h5")


##FLASK
app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def make_predict():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(100,100))
    img = img/255
    img = np.expand_dims(img, axis=0)
    # do some fancy processing here....
    #np array goes into nueral network, prediction comes out
    y_hat = model.predict(img)[0][0]
    # build a response dict to send back to client
    response = '%.1f' %(float(y_hat)*10)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
