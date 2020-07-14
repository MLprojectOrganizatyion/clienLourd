
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import cv2
from tqdm import tqdm
from PIL import Image
import random
import re
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import image
#from IPython.display import display
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras
import keras.backend as kb
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

dst_path = "./data_resize_rename/"

#path_sign_20 = "./dataV0/0/"
path_sign_30 = "./dataV0/1/"
path_sign_50 = "./dataV0/2/"
path_sign_60 = "./dataV0/3/"

#path_tab = [path_sign_20, path_sign_30, path_sign_50, path_sign_60]
path_tab = [path_sign_30, path_sign_50, path_sign_60]
#obj_20 = "traffic_sg_20_"
obj_30 = "traffic_sgn_30_"
obj_50 = "traffic_sgn_50_"
obj_60 = "traffic_sgn_60_"

#obj_tab = [obj_20, obj_30, obj_50, obj_60]
obj_tab = [obj_30, obj_50, obj_60]

#nb_sign_20 = 0
nb_sign_30 = 0
nb_sign_50 = 0
nb_sign_60 = 0

#dst_path_20 = "./data_resize_rename/sign_20"
dst_path_30 = "./data_resize_rename/sign_30/"
dst_path_50 = "./data_resize_rename/sign_50/"
dst_path_60 = "./data_resize_rename/sign_60/"
#dst_path_tab = [dst_path_20, dst_path_30, dst_path_50, dst_path_60]
dst_path_tab = [dst_path_30, dst_path_50, dst_path_60]

# ## 1. Rename data


def rename_multiple_files(path, obj):
    for i, filename in enumerate(os.listdir(path)):
        os.rename(path + filename, path + obj + str(i) + ".png")


# rename_multiple_files(path_sign_20,obj_20)


for p in path_tab:

    # if p == path_sign_20:
    #     rename_multiple_files(path_sign_20, obj_20)
    #     print('Rename successful.')
    if p == path_sign_30:
        rename_multiple_files(path_sign_30, obj_30)
        print('Rename successful.')
    if p == path_sign_50:
        rename_multiple_files(path_sign_50, obj_50)
        print('Rename successful.')
    if p == path_sign_60:
        rename_multiple_files(path_sign_60, obj_60)
        print('Rename successful.')


#
# ## 2. Resize


def resize_multiple_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in os.listdir(src_path):
        try:
            img = Image.open(src_path + filename)
            new_img = img.resize((64, 64))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path + filename)
        except:
            continue


for p in path_tab:

    # if p == path_sign_20:
    #     resize_multiple_images(path_sign_20, dst_path + "sign_20/")
    if p == path_sign_30:
        resize_multiple_images(path_sign_30, dst_path + "sign_30/")
    if p == path_sign_50:
        resize_multiple_images(path_sign_50, dst_path + "sign_50/")
    if p == path_sign_60:
        resize_multiple_images(path_sign_60, dst_path + "sign_60/")

    print('Resize successful.')

# ## 3. Split Data

# ### 3.1. Count image per folder


#dst_path_20 = "./data_resize_rename/sign_20/"
dst_path_30 = "./data_resize_rename/sign_30/"
dst_path_50 = "./data_resize_rename/sign_50/"
dst_path_60 = "./data_resize_rename/sign_60/"

#dst_path_tab = [dst_path_20, dst_path_30, dst_path_50, dst_path_60]
dst_path_tab = [dst_path_30, dst_path_50, dst_path_60]

def count_images(path):
    nb_images = len([f for f in os.listdir(path) if f.endswith('.png')])
    return nb_images


for p in dst_path_tab:

    # if p == dst_path_20:
    #     nb_sign_20 = count_images(p)
    if p == dst_path_30:
        nb_sign_30 = count_images(p)
    if p == dst_path_50:
        nb_sign_50 = count_images(p)
    if p == dst_path_60:
        nb_sign_60 = count_images(p)

#print('nb_sign_20=', nb_sign_20)
print('nb_sign_30=', nb_sign_30)
print('nb_sign_50=', nb_sign_50)
print('nb_sign_60=', nb_sign_60)

# ### 3.2. Get percentages


#nb_test_sign_20 = int((nb_sign_20 * 20) / 100)
nb_test_sign_30 = int((nb_sign_30 * 20) / 100)
nb_test_sign_50 = int((nb_sign_50 * 20) / 100)
nb_test_sign_60 = int((nb_sign_60 * 20) / 100)
#print('nb_test_sign_20=', nb_test_sign_20)
print('nb_test_sign_30=', nb_test_sign_30)
print('nb_test_sign_50=', nb_test_sign_50)
print('nb_test_sign_60=', nb_test_sign_60)


# ### 3.3. Create Testing Data


def split_data(path, nb_images_test):
    images_tab = os.listdir(path)
    return images_tab[:nb_images_test]


import shutil


def create_testing_data(src_path, nb_images_test, dst_path):
    images_tab = split_data(src_path, nb_images_test)
    for filename in images_tab:
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        shutil.move(src_path + filename, dst_path)


data_test_path = "./test/"
for p in dst_path_tab:

    # if p == dst_path_20:
    #     create_testing_data(p, nb_test_sign_20, data_test_path)
    if p == dst_path_30:
        create_testing_data(p, nb_test_sign_30, data_test_path)
    if p == dst_path_50:
        create_testing_data(p, nb_test_sign_50, data_test_path)
    if p == dst_path_60:
        create_testing_data(p, nb_test_sign_60, data_test_path)

print('Testing Data Done.')


### 3.4. Create Training Datat


def create_training_data(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in os.listdir(src_path):
        try:
            img = Image.open(src_path + filename)
            # new_img = img.resize((64,64))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            img.save(dst_path + filename)
        except:
            continue


data_train_path = "./train/"
for p in dst_path_tab:

    # if p == dst_path_20:
    #     create_training_data(p, data_train_path)
    if p == dst_path_30:
        create_training_data(p, data_train_path)
    if p == dst_path_50:
        create_training_data(p, data_train_path)
    if p == dst_path_60:
        create_training_data(p, data_train_path)

print('Training Data Done.')
#
#
# # ## 4. Transform Data
#
#
# def get_label(path):
#     label = []
#     for filename in os.listdir(path):
#         try:
#             # if re.match(r'traffic_sg_20_', filename):
#             #     label.append(20)
#             if re.match(r'traffic_sg_30_', filename):
#                 label.append(30)
#             if re.match(r'traffic_sg_50_', filename):
#                 label.append(50)
#             if re.match(r'traffic_sg_60_', filename):
#                 label.append(60)
#         except:
#             continue
#     return np.array(label)
#
#
# path_to_train_set = "./train/"
# y_train = get_label(path_to_train_set)
# y_train = to_categorical(y_train)
# print('y_train', y_train.shape)
#
# path_to_test_set = "./test/"
# y_test = get_label(path_to_test_set)
# y_test = to_categorical(y_test)
# print('y_test', y_test)
#
#
# def get_data(path):
#     train_image = []
#
#     for filename in os.listdir(path):
#         try:
#             img = image.load_img(path + filename, target_size=(28, 28, 3))
#             img = image.img_to_array(img)
#             img = img / 255
#             train_image.append(img)
#         except:
#             continue
#     return np.array(train_image)
#
#
# path_to_train_set = "./train/"
# x_train = get_data(path_to_train_set)
# print('x_train', x_train)
#
# # In[29]:
#
#
# path_to_test_set = "./test/"
# x_test = get_data(path_to_test_set)
# print('x_test', x_test)
#
# # ## 5. create linear model with keras
#
#
# model = Sequential()
# # first layer - input_shape is necessary
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(61, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# model.fit(X, y, epochs=10, validation_data=(x_test, y_test))


# model = keras.Sequential([
#     keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[1]),
#     keras.layers.Dense(32, activation=tf.nn.relu),
#     keras.layers.Dense(32, activation=tf.nn.relu),
#     keras.layers.Dense(1)
# ])
#
# optimizer = tf.keras.optimizers.RMSprop(0.0099)
# model.compile(loss='mean_squared_error', optimizer=optimizer)
# model.fit(X_train, y_train, epochs=500)


"""
def get_data(path):
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        try:
            if re.match(r'traffic_sign_20_',filename):
                label.append(20)
            if re.match(r'traffic_sign_30_',filename):
                label.append(30)
            if re.match(r'traffic_sign_50_',filename):
                label.append(50)
            if re.match(r'traffic_sign_60_',filename):
                label.append(60)
            img=Image.open(path + filename)
            #x = np.array([[house_size, bhks]])
            np_array = np.asarray(img)
            #print(len(np_array))
            l,b,c = np_array.shape
            np_array = np_array.reshape(l*b*c,)
            all_images_as_array.append(np_array)
        except:
            continue
    return np.array([all_images_as_array]), np.array([label])


path_to_train_set = "C:/Users/33769/MLproject/treatMyDataSet/train/"
path_to_test_set = "C:/Users/33769/MLproject/treatMyDataSet/test/"
X_train,y_train = get_data(path_to_train_set)
X_test, y_test = get_data(path_to_test_set)
print('X_train set : ', X_train)
print('y_train set : ', y_train)
print('X_test  set : ', X_test)
print('y_test  set : ', y_test)
"""

"""from PIL import Image
import os
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from keras.preprocessing import image
from IPython.display import display
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def get_data(path):
    train_image =[]

    for filename in os.listdir(path):
        try:
            img=Image.open(path + filename)
            #x = np.array([[house_size, bhks]])
            np_array = np.asarray(img)
            #print(len(np_array))
            l,b,c = np_array.shape
            np_array = np_array.reshape(l*b*c,)
            train_image.append(np_array)
        except:
            continue
    return np.array(train_image)"""
