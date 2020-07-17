try:
    import pika
    import json
    import logging
    import pika
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import os.path
    import re
    import keras
    from os import listdir
    import numpy as np
    import tensorflow.keras as keras
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import json
    from keras.models import load_model
    import pickle
    import os, shutil

except Exception as e:
    print("Sone Modules are missings {}".format_map(e))

import pika, os, logging

target_size = (64, 64)


def get_label(path):
    label = []

    # for filename in os.listdir(path):
    for i, filename in enumerate(os.listdir(path)):
        if re.match(r'traffic_sg_30_', filename):
            label.append(np.array([1, 0, 0]))
            # print("30", filename)
        if re.match(r'traffic_sg_50_', filename):
            label.append(np.array([0, 1, 0]))
            # print("50", i)
        if re.match(r'traffic_sg_60_', filename):
            label.append(np.array([0, 0, 1]))
            # print("60", i)
    print("label", np.array(label).shape)
    return np.array(label)


def get_data(path):
    train_image = []

    for filename in os.listdir(path):
        try:
            img = np.array(Image.open(f'{path}{filename}').convert('RGB').resize(target_size)) / 255.0
            train_image.append(img)
        except:
            continue
    return np.array(train_image)


def y_train():
    path_to_train_set = "./train/"
    y_train = get_label(path_to_train_set)
    # y_train = to_categorical(y_train)
    print("labels+++++++++++", y_train.shape)
    return y_train


def x_train():
    path_to_train_set = "./train/"
    x_train = get_data(path_to_train_set)
    print("train+++++++++++", x_train.shape)
    return x_train


def x_test():
    path_to_test_set = "./test/"
    x_test = get_data(path_to_test_set)
    return x_test


def y_test():
    path_to_test_set = "./test/"
    y_test = get_label(path_to_test_set)
    # y_test = to_categorical(y_test)
    return y_test


def resize_multiple_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in listdir(src_path):
        try:
            img = Image.open(src_path + filename)
            new_img = img.resize((64, 64))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            new_img.save(dst_path + filename)
        except:
            continue


def validation():
    path = "./validation/"
    path_from = "./loadImages/"
    resize_multiple_images(path_from, path)
    for img_name in listdir(path):
        validation = []
        validation.append(
            np.array(Image.open(f'{path}{img_name}').convert('RGB').resize(target_size)) / 255.0)  # color
    return np.array(validation)


def deleteValidationFolder():
    folder = "./validation/"
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # Elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def deleteLoadImageFolder():
    folder = "./loadImages/"
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # Elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def create_linear_model():
    m = keras.models.Sequential()
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def create_mlp_model():
    m = keras.models.Sequential()
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def create_conv_nn_model():
    m = keras.models.Sequential()

    m.add(keras.layers.Conv2D(4, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))#filtres
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Conv2D(8, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation=keras.activations.relu, padding='same'))
    m.add(keras.layers.MaxPool2D((2, 2)))

    m.add(keras.layers.Flatten())

    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def create_residual_nn_model():
    input_tensor = keras.layers.Input((target_size[0], target_size[1], 3))

    previous_tensor = keras.layers.Flatten()(input_tensor)

    next_tensor = keras.layers.Dense(64, activation=keras.activations.tanh)(
        previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = keras.layers.Dense(64, activation=keras.activations.tanh)(
        previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = keras.layers.Dense(64, activation=keras.activations.tanh)(
        previous_tensor)

    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])

    next_tensor = keras.layers.Dense(64, activation=keras.activations.tanh)(previous_tensor)
    next_tensor = keras.layers.Dense(3, activation=keras.activations.sigmoid)(next_tensor)

    m = keras.models.Model(input_tensor, next_tensor)
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def create_u_net_model():
    input_tensor = keras.layers.Input((target_size[0], target_size[1], 3))

    flattened_input = keras.layers.Flatten()(input_tensor)

    output1 = keras.layers.Dense(64, activation=keras.activations.tanh)(
        flattened_input)

    output2 = keras.layers.Dense(64, activation=keras.activations.tanh)(
        output1)

    previous_tensor = keras.layers.Concatenate()([output1, output2])

    output3 = keras.layers.Dense(64, activation=keras.activations.tanh)(
        previous_tensor)

    previous_tensor = keras.layers.Concatenate()([output3, flattened_input])

    next_tensor = keras.layers.Dense(64, activation=keras.activations.tanh)(previous_tensor)
    next_tensor = keras.layers.Dense(3, activation=keras.activations.sigmoid)(next_tensor)

    m = keras.models.Model(input_tensor, next_tensor)
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def show_confusion_matrix(m, x, y, show_errors: bool = True):
    predicted_values = m.predict(x)
    predicted_labels = np.argmax(predicted_values, axis=1)
    true_labels = np.argmax(y, axis=1)
    print(confusion_matrix(true_labels, predicted_labels))
    # if show_errors:
    #     for i in range(len(predicted_labels)):
    #         if predicted_labels[i] != true_labels[i]:
    #             plt.imshow(x[i])
    #             plt.show()


def training(m):
    if m == 0:
        print("***linear_model***")
        model = create_linear_model()
        logs = model.fit(x_train(), y_train(), epochs=200, batch_size=4)
        show_confusion_matrix(model, x_test(), y_test())
        model.save("linear_model.keras")

    if m == 1:
        print("***mlp***")
        model = create_mlp_model()
        model.fit(x_train(), y_train(), epochs=40, batch_size=4)
        show_confusion_matrix(model, x_test(), y_test())
        model.save("mlp_model.keras")
        # saveModel(model, 'mlp_mode.pkl')
    if m == 2:
        print("***conv_nn***")
        model = create_conv_nn_model()
        logs = model.fit(x_train(), y_train(), epochs=100, batch_size=4)
        show_confusion_matrix(model, x_test(), y_test())
        model.save("conv_nn_model.keras")

    if m == 3:
        print("***residual_nn***")
        model = create_residual_nn_model()
        logs = model.fit(x_train(), y_train(), epochs=200, batch_size=4)
        show_confusion_matrix(model, x_test(), y_test())
        model.save("residual_nn_model.keras")

    if m == 4:
        print("***u_net***")
        model = create_u_net_model()
        logs = model.fit(x_train(), y_train(), epochs=200, batch_size=4)
        show_confusion_matrix(model, x_test(), y_test())
        model.save("u_net_model.keras")


def checking(m, picture):
    if m == "0":
        loaded = keras.models.load_model("linear_model.keras")
        print("lineare model")
    if m == "1":
        loaded = keras.models.load_model('mlp_model.keras')
    if m == "2":
        loaded = keras.models.load_model("conv_nn_model.keras")
    if m == "3":
        loaded = keras.models.load_model("residual_nn_model.keras")
    if m == "4":
        loaded = keras.models.load_model("u_net_model.keras")

    predicted = loaded.predict(picture)
    print("x_validation", predicted)
    for i in range(len(predicted)):  # Loops through each game

        sign_30 = predicted[i][0]
        sign_50 = predicted[i][1]
        sign_60 = predicted[i][2]
        predicted_30 = round(sign_30, 4)
        predicted_50 = round(sign_50, 4)
        predicted_60 = round(sign_60, 4)
        Percent_30 = "{:.2%}".format(predicted_30)
        Percent_50 = "{:.2%}".format(predicted_50)
        Percent_60 = "{:.2%}".format(predicted_60)

        print("Percent_30", Percent_30)
        print("Percent_50", Percent_50)
        print("Percent_60", Percent_60)
    return Percent_30, Percent_50, Percent_60


def saveModel(model, filename):
    # Change to where you want to save the model
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def openModel(filename):
    with open(filename, 'rb') as file:  # Change filename here if model is named differently
        pickleModel = pickle.load(file)
    return pickleModel


if __name__ == "__main__":

    m = 3
    train = 1

    if train == 1:
        training(m)
    if train == 0:
        checking(m, validation())

    deleteValidationFolder()
    deleteLoadImageFolder()


# # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
# plt.plot(logs.history['loss'])
# plt.plot(logs.history['val_loss'])
# plt.show()
#
# # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
# plt.plot(logs.history['accuracy'])
# plt.plot(logs.history['val_accuracy'])
# plt.show()

# show_confusion_matrix(model, x_test(), y_test())
# model.save("my_model.keras")
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(61, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam',
#               metrics=['accuracy'])
#
# model.fit(x_train(), y_train(), epochs=10, batch_size=4, validation_data=(x_test(), y_test()))
