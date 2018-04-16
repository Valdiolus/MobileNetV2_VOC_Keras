import keras

from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model

import numpy as np
import cv2

import mobilenet_v2 as mn2
CLASSES = ['person', 'bicycle', 'car', 'bus', 'motorbike']

classes=5
pic_numb = 23195
pic_train = 20000
pic_test = pic_numb-pic_train

net_w = 224
net_h = 224
class_acc = np.zeros((5,2))#1 line - right answers, 2 line - total

in_data_test = np.zeros((pic_test, net_w, net_h, 3), dtype = np.uint8)
in_labels_test = np.zeros((pic_test,5))

def Get_data(src):
    #Resize Input data 23195 files 5 classes
    f = open('Dataset/Labels_mixed.txt', 'r')
    i_pic_train=0
    i_pic_test=0
    for line in f.readlines():
        i=0
        for i in range(16):
            if (line[i] == " "):#search for the end of the name
                end_line = i
                break
        name = line[0:i]


        if(i_pic_train < pic_train):
            i_pic_train += 1
        else:
            if (i_pic_test >= pic_test):
                break
            img = cv2.imread('Dataset/Images/%1s.jpg' % name, -1)
            in_data_test[i_pic_test] = cv2.resize(img, (net_w, net_h))
            in_labels_test[i_pic_test, int(line[15])] = 1
            i_pic_test += 1

        if((i_pic_train+i_pic_test)%1000 == 0):
            print("Import data progress:",i_pic_train,i_pic_test)

    f.close()

def Load_Model(mdl,x,y,batch):
    # model = mn2.MobileNetv2((net_w, net_h, 3), classes)
    # model.load_weights('model/13_4_18_first/19432.h5')
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(mdl)
        # output = model.predict(in_data_test)
        # print('Model output:\n', output)
        # print('Labels:\n', in_labels_test)
        test1 = model.evaluate(x=x,
                               y=y,
                               batch_size=batch)

        print("TEST:", test1)

def Test_model_classes(mdl,dsc):
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(mdl)
        output = model.predict(in_data_test)
        #print('shape:', output.shape)
        #print('Model output:\n', output[0])
        #print('Labels:\n', in_labels_test[0])
        print('Testing classes...')
        for i in range(pic_test):
            for j in range(5):
                if(output[i,j]>0.5 and in_labels_test[i,j]==1):
                    class_acc[j, 0] += 1
                if(in_labels_test[i,j]==1):
                    class_acc[j,1] += 1
        print(class_acc)
        f = open(dsc, 'w')
        for i in range(5):
            print('Class %1s acc=%1s' % (CLASSES[i], (class_acc[i,0]/class_acc[i,1])))
            f.write('Class %1s acc=%1s\n' % (CLASSES[i], (class_acc[i, 0] / class_acc[i, 1])))
        f.close()



Get_data(1)
#Load_Model('model/15_04_18_first_continue2/full_30059.h5', x=in_data_test, y=in_labels_test, batch=8)
Test_model_classes('model/16_04_18_first_continue2/full_25986.h5',
                   'model/16_04_18_first_continue2/class_test.txt')