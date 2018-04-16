from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
import random, os
import cv2

import mobilenet_v2 as mn2

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

np.random.seed(1)
number = random.randint(1,100000)
print("Current Random Number:", number)
# tensorboard --logdir=./NN/keras/MobileNetV2_VOC_Keras/keras_logs
tbCallBack = TensorBoard(log_dir='./keras_logs/%1s' % number,
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)

classes=5
pic_numb = 23195
pic_train = 20000
pic_val = 0
pic_test = pic_numb-pic_val-pic_train

net_w = 224
net_h = 224
batch=8
epochs=80
learn_rate=0.001#0.001-def

in_data_train = np.zeros((pic_train, net_w, net_h, 3), dtype = np.uint8)
in_data_val = np.zeros((pic_val, net_w, net_h, 3), dtype = np.uint8)
in_data_test = np.zeros((pic_test, net_w, net_h, 3), dtype = np.uint8)
in_labels_train = np.zeros((pic_train,5), dtype = np.uint8)
in_labels_val = np.zeros((pic_val,5), dtype = np.uint8)
in_labels_test = np.zeros((pic_test,5), dtype = np.uint8)

def Get_input_data(f):
    #Resize Input data 23195 files 5 classes
    f = open('Dataset/Labels_mixed.txt', 'r')
    i_pic_train=0
    i_pic_val=0
    i_pic_test=0
    for line in f.readlines():
        i=0
        for i in range(16):
            if (line[i] == " "):#search for the end of the name
                end_line = i
                break
        name = line[0:i]
        img = cv2.imread('Dataset/Images/%1s.jpg' % name, -1)

        if(i_pic_train < pic_train):
            in_data_train[i_pic_train] = cv2.resize(img, (net_w, net_h))
            in_labels_train[i_pic_train, int(line[15])] = 1
            i_pic_train += 1
        else:
            in_data_test[i_pic_test] = cv2.resize(img, (net_w, net_h))
            in_labels_test[i_pic_test, int(line[15])] = 1
            i_pic_test += 1
        """
        elif(i_pic_train == pic_train and i_pic_val < pic_val):
            in_data_val[i_pic_val] = cv2.resize(img, (net_w, net_h))
            in_labels_val[i_pic_val, int(line[15])] = 1
            i_pic_val += 1"""
        if((i_pic_train+i_pic_val+i_pic_test)%1000 == 0):
            print("Import data progress:",i_pic_train,i_pic_val,i_pic_test)

    f.close()

def fine_tune(num_classes, weights, model):

    model.load_weights(weights)

    x = model.get_layer('Dropout').output
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs=model.input, outputs=output)


    return model

def train(src, begin):
    model = mn2.MobileNetv2((net_w, net_h, 3), classes)
    model.load_weights(src)

    opt = Nadam(lr=learn_rate)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train = model.fit(
        x=in_data_train,
        y=in_labels_train,
        batch_size=batch,
        validation_data=(in_data_test, in_labels_test),
        #steps_per_epoch=pic_train // batch,
        #validation_steps=count2 // batch,
        epochs=epochs,
        callbacks=[tbCallBack],
        initial_epoch=begin)

    test = model.evaluate(x=in_data_test,
                          y=in_labels_test,
                          batch_size=batch)

    print("TRAIN:",train)
    print("TEST:", test)
    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(train.history)
    df.to_csv('model/%1s.csv' % number, encoding='utf-8', index=False)
    model.save_weights('model/%1s.h5' % number)
    model.save('model/full_%1s.h5' % number)


Get_input_data(1)
train('model/14_04_18_second_continue/97471.h5', 0)
