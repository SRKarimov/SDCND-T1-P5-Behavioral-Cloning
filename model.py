# libs
import csv
import numpy as np
import cv2
# from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import Cropping2D
# from utils import INPUT_SHAPE, batch_generator
import argparse
import os

test_size = 0.2
keep_prob = 0.5
save_best_only = True
learning_rate = 1.0e-4
samples_per_epoch = 128
nb_epoch = 9


def crop_image(image):
    return image[60:-25, :, :] # remove the sky and the car front


def resize_image(image):
    return cv2.resize(image, (66, 200), cv2.INTER_AREA)


def rgb_to_yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocessor_image(image):
    image = crop_image(image)
    image = resize_image(image)
    image = rgb_to_yuv(image)
    return image


def load_data():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:

        # center image
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        current_path = 'data/IMG/' + file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

        # left image
        source_path = line[1]
        file_name = source_path.split('/')[-1]
        current_path = 'data/IMG/' + file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(measurement + 0.2)

        # right image
        source_path = line[2]
        file_name = source_path.split('/')[-1]
        current_path = 'data/IMG/' + file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(measurement - 0.2)

    X = np.array(images)
    y = np.array(measurements)

    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7007)

    return X, y


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 1.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((65, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, X, y):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=save_best_only,
                                 mode='auto')

    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, validation_split=0.2, nb_epoch=nb_epoch)
    model.save('model.h5')
    print('Done!\n')


def main():
    # load data
    X, y = load_data()

    print(X.shape, y.shape, "\n")

    # build model
    model = build_model()

    # train model on data, it saves as model.h5
    train_model(model, X, y)
    exit(0)


if __name__ == '__main__':
    main()
