from keras import applications, optimizers
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.utils import np_utils
import cv2
import numpy as np
import csv, sqlite3
import math
import os
import random


def init_model(num_classes):
    # If you want to specify input tensor
    img_rows = 100
    img_cols = 100
    num_classes = num_classes

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
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Custom Optimizer
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=1e-6)

    # Do not forget to compile it
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    return model

def get_filenames():
    files_dict = {}

    cur.execute("SELECT filename, class_score FROM slimscoresclass")

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

        # do not include outliers
        if filename not in files_dict.keys():
            continue

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

    random.shuffle(train_keys)
    random.shuffle(test_keys)

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

def getTrainData(chunk,img_rows,img_cols,num_classes):
    X_train,Y_train = get_train_data(chunk,img_rows,img_cols)
    if (X_train is not None and Y_train is not None):
        X_train/=255
        encoded_Y = encoder.transform(Y_train)
        Y_train=np_utils.to_categorical(encoded_Y,num_classes)
    return (X_train,Y_train)

def getTestData(chunk,img_rows,img_cols,num_classes):
    X_test,Y_test = get_test_data(chunk,img_rows,img_cols)
    if (X_test is not None and Y_test is not None):
        X_test/=255
        encoded_Y = encoder.transform(Y_test)
        Y_test=np_utils.to_categorical(encoded_Y,num_classes)
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

    nb_epoch = 1000
    num_epochs = 1
    batch_size = 2
    chunk_size = 32
    img_rows = 100
    img_cols = 100
    num_classes = 101

    model = init_model(num_classes)

    # weights_path = 'model_regress.h5'
    weights_path = None

    if weights_path:
        model.load_weights(weights_path)

    Y = np.arange(0, 10.1, 0.1)
    Y = [str(num) for num in Y]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)

    for e in range(nb_epoch):
        print('-'*40)
        print 'Epoch', e
        print('-'*40)
        print "Training..."
        instance_count=0


        for chunk in chunks(train_keys, chunk_size):
            X_chunk,Y_chunk=getTrainData(chunk,img_rows,img_cols,num_classes)

            if (X_chunk is not None and Y_chunk is not None):
                loss = model.fit(X_chunk, Y_chunk, verbose=1, batch_size=batch_size, epochs=num_epochs)
                instance_count+=chunk_size

                print "Epoch Count:", e
                print "Instance Count:", instance_count

                if instance_count%640==0:
                    model.save_weights('model_regress.h5',overwrite=True)
