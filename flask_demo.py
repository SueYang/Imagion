import numpy as np
#need to 'conda install flask' for this to work
from flask import Flask, abort, jsonify, request
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
my_model = custom_model.load_weights("/home/roiana_reid/Imagion/basic_model.h5")


##FLASK
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():
  #all kinds of error checking should go here
  data = request.get_json(force=True)
  #predict_request = data
  #predict_request = np.array(request.json, dtype=np.uint8)
  read_img = cv2.imread(data)
  read_img = cv2.resize(read_img,(160,160))
  read_img = np.expand_dims(read_img, axis=0)
  #np array goes into random forest, prediction comes out
  y_hat = np.argmax(my_model.predict(read_img))
  #return our prediction
  output = y_hat
  return jsonify(results=output)

if __name__ == '__main__':
  app.run(port = 9000, debug = True)
