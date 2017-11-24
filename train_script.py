from keras import applications, optimizers
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import cv2
import numpy as np
import csv, sqlite3
import math
import os
import random


def init_model():
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
    x = Dense(1)(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    from keras.models import Model
    custom_model = Model(inputs=vgg_model.input, outputs=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:15]:
        layer.trainable = False

    # Custom Optimizer
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

    # Do not forget to compile it
    custom_model.compile(loss='mse',
                         optimizer=opt,
                         metrics=['accuracy'])
    return custom_model

def get_filenames():
    files_dict = {}

    cur.execute("SELECT filename, norm_score FROM scores")

    count = 0

    for file_, score in cur.fetchall():
        if count == 0:
            count += 1
            continue

        files_dict[file_] = score

        count += 1

    return files_dict

def create_user_dict(dataset_dir):
    user_dict = {}

    for filename in os.listdir(dataset_dir):
        filename = filename.rsplit('.', 1)[0]
        alias = filename.rsplit('_', 1)[0]

        if alias not in user_dict:
            user_dict[alias] = [filename]
        else:
            user_dict[alias].append(filename)

    return user_dict



def split_sets():
    """Split training and test images"""

    PERCENT_TRAINING = 0.75

    random.seed(10)
    keys = user_dict.keys()
    split = int(len(user_dict.keys()) * PERCENT_TRAINING)

    random.shuffle(keys) # revisit this shuffle function

    train_users = keys[:split]
    test_users = keys[split:]

    train_keys = []
    test_keys = []

    for user in train_users:
        for filename in user_dict[user]:
            train_keys.append(filename)

    for user in test_users:
        for filename in user_dict[user]:
            test_keys.append(filename)

    return [train_users, test_users, train_keys, test_keys]


def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def get_train_data(chunk, img_row, img_col):
    X_train = []
    Y_train = []

    for imgname in chunk:
        try:
            filename = 'data_images'+'/'+imgname+'.png'
            img = cv2.imread(filename)
            img = cv2.resize(img,(img_row,img_col))
            X_train.append(img)
            Y_train.append(files_dict[imgname])
        except:
            continue
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    return X_train,Y_train

def get_test_data(chunk, img_row, img_col):
    X_test = []
    Y_test = []

    for imgname in chunk:
        try:
            filename = './data_images'+'/'+imgname+'.png'
            img = cv2.imread(filename)
            img = cv2.resize(img,(img_row,img_col))
            X_test.append(img)
            Y_test.append(files_dict[imgname])
        except:
            continue
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    return X_test,Y_test

def getTrainData(chunk,img_rows,img_cols):
    X_train,Y_train = get_train_data(chunk,img_rows,img_cols)
    if (X_train is not None and Y_train is not None):
        X_train/=255
    return (X_train,Y_train)

def getTestData(chunk,img_rows,img_cols):
    X_test,Y_test = get_test_data(chunk,img_rows,img_cols)
    if (X_test is not None and Y_test is not None):
        X_test/=255
    return (X_test,Y_test)

def test(model, nb_epoch, spatial_test_data, img_rows, img_cols):
    X_test,Y_test = getTestData(test_keys,img_rows,img_cols)
    return (X_test, Y_test)


if __name__ == "__main__":
    # Create Dictionaries
    DATASET_DIR = 'data_images'
    con = sqlite3.connect("imagion.db")
    cur = con.cursor()

    files_dict = get_filenames()
    user_dict = create_user_dict(DATASET_DIR)

    # Split training and test sets by users and files
    train_users, test_users, train_keys, test_keys = split_sets()

    nb_epoch = 100
    num_epochs = 1
    batch_size = 2
    chunk_size = 32
    img_rows = 160
    img_cols = 160

    custom_model = init_model()

    for e in range(nb_epoch):
        print('-'*40)
        print 'Epoch', e
        print('-'*40)
        print "Training..."
        instance_count=0


        for chunk in chunks(train_keys, chunk_size):
            X_chunk,Y_chunk=getTrainData(chunk,img_rows,img_cols)

            if (X_chunk is not None and Y_chunk is not None):
                loss = custom_model.fit(X_chunk, Y_chunk, verbose=1, batch_size=batch_size, epochs=num_epochs)
                instance_count+=chunk_size

                print "Instance Count:", instance_count

                if instance_count%100==0:
                    custom_model.save_weights('basic_model.h5',overwrite=True)
