import numpy as np
#need to 'conda install flask' for this to work
from flask import Flask, abort, jsonify, request, Response
import jsonpickle
from keras.models import model_from_json
import cPickle as pickle
import cv2
import csv, sqlite3

from keras import applications
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils

##MODEL STUFF UP TOP, FLASK STUFF BELOW

##MODEL PARAMETERS AND SAVED WEIGHTS
# If you want to specify input tensor
input_tensor = Input(shape=(160, 160, 3))
vgg_model = applications.VGG16(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)

# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
x = layer_dict['block4_pool'].output

# Stacking a new simple convolutional network on top of it
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)

#my_model = pickle.load(open("file_predict.pkl", "rb"))
#my_model = model_from_json(json_data)
# Creating new model. Please note that this is NOT a Sequential() model.
from keras.models import Model
custom_model = Model(inputs=vgg_model.input, outputs=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:15]:
    layer.trainable = False

# Do not forget to compile it
custom_model.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])
custom_model.load_weights("/home/roiana_reid/Imagion/basic_model.h5")


##FLASK
app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def make_predict():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(160,160))
    img = np.expand_dims(img, axis=0)
    # do some fancy processing here....
    #np array goes into nueral network, prediction comes out
    y_hat = np.argmax(custom_model.predict(img))
    # build a response dict to send back to client
    response = {'message': int(y_hat)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
