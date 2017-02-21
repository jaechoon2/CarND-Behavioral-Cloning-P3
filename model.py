import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout, Lambda, ELU
from os import listdir
import csv
import cv2
import json
#from imageutil import batch_flip_left_right, batch_generate_random_brightness, batch_generate_random_shadow

""" Global Constants """


DATA_DIR = 'data_aug/'


class DrivingLogIndex:
    Center = 0
    Left = 1
    Right = 2
    Steering = 3
    Throttle = 4
    Brake = 5
    Speed = 6

    def __init__(self):
        pass


STEERING_OFFSET = 0.3
IMG_WIDTH, IMG_HEIGHT, COLOR_IMG_DEPTH = 200, 66, 3
    
""" Helper Methods"""


# Load the driving_log.csv file.
def load_data_from_log():
    with open(DATA_DIR + 'driving_log.csv', 'r') as f:
        log = list(csv.reader(f))
    return log[1:] # Remove header


# Return the image that is optimized for the Nvidia spec (200x66x3)
# This method crops the top 1/3 of the image as the cropped region is not useful for training.
def get_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Crop the top 1/3 of the image
    img_shape = img.shape
    img = img[int(img_shape[0]/3):img_shape[0], 0:img_shape[1]]
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return img    


# Return the image of the left camera
def get_left_image(pos):
    img_path = '{}{}'.format(DATA_DIR, _driving_log[pos][DrivingLogIndex.Left].strip())
    return get_image(img_path)


# Return the image of the right camera
def get_right_image(pos):
    img_path = '{}{}'.format(DATA_DIR, _driving_log[pos][DrivingLogIndex.Right].strip())
    return get_image(img_path)


# Return the image of the center camera
def get_center_image(pos):
    img_path = '{}{}'.format(DATA_DIR, _driving_log[pos][DrivingLogIndex.Center].strip())
    return get_image(img_path)


# Return the steering angle of the image for the left camera
def get_left_steering(pos):
    return get_center_steering(pos) + STEERING_OFFSET


# Return the steering angle of the image for the right camera
def get_right_steering(pos):
    return get_center_steering(pos) - STEERING_OFFSET


# Return the steering angle of the image for the center camera
def get_center_steering(pos):
    return float(_driving_log[pos][DrivingLogIndex.Steering])


# Load training data from the driving_log file
def load_training_data():
    len_driving_log = len(_driving_log)
    steps = 3 # center, left, right
    X_res = np.ndarray((len_driving_log * steps, IMG_HEIGHT, IMG_WIDTH, COLOR_IMG_DEPTH), dtype=np.uint8)
    y_res = np.ndarray((len_driving_log * steps), dtype=np.float32)
    
    # Load center, left and right images and steering data
    for i in range(0, len_driving_log):
        X_res[i] = get_center_image(i)
        y_res[i] = get_center_steering(i)
        X_res[i+len_driving_log] = get_left_image(i)
        y_res[i+len_driving_log] = get_left_steering(i)
        X_res[i+len_driving_log*2] = get_right_image(i)
        y_res[i+len_driving_log*2] = get_right_steering(i)

    return X_res, y_res


""" The Model """


# Reference: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def nvidia_model():
    model = Sequential()

    # normalize image values between -.5 : .5
    model.add(Lambda(lambda x: x/255 - .5, input_shape=(IMG_HEIGHT, IMG_WIDTH, COLOR_IMG_DEPTH),
                     output_shape=(IMG_HEIGHT, IMG_WIDTH, COLOR_IMG_DEPTH)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))

    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


""" Image Utils for Data Augmentation """


# flip the provided list of images from left to right 
def batch_flip_left_right(X, y):
    X_result = np.zeros_like(X)
    y_result = np.zeros_like(y)
    for i in range(len(X)):
        X_result[i] = np.fliplr(X[i])
        y_result[i] = -1 * y[i]
    return X_result, y_result


# Randomly change the brightness of the provided list of images
def batch_generate_random_brightness(X):
    X_result = np.zeros_like(X)
    for i in range(len(X)):
        X_result[i] = get_random_brightness(X[i])
    return X_result


# Generate random shadows for the provided list of images
def batch_generate_random_shadow(X):
    X_result = np.array(X)
    for i in range(len(X_result)):
        X_result[i] = get_random_shadow_overlay(X_result[i])
    return X_result    


# Generate random brightness based on the provided image.
# Note: It does not alter the original image.
def get_random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
    random_light = 0.25 + np.random.uniform()
    hsv[:,:, 2] = hsv[:,:, 2] * random_light
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# Generate random shadow based on the provided image.
# Note: It does not alter the original image.
def get_random_shadow_overlay(img):
    img_result = np.array(img)
    h, w = img_result.shape[0], img_result.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = -k * x1
    for i in range(h):
        c = int ((i - b) / k)
        img_result[i, :c, :] = (img_result[i, :c, :] * .5).astype(np.int32)
    return img_result


# Simulates the bumpy road by randomly shift the image array vertically
def shift_random_vertically(img):
    top = int(random.uniform(.075, .175) * img.shape[0])
    bottom = int(random.uniform(.075, .175) * img.shape[0])
    return img[top:-bottom, :]    


# Generate a new image from the provided image using random shifts in the horizontal direction
def jiggle_data(img, angle):
    # Reference: https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319#.xzwtzhnaa
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    angle_result = angle + transX/transRange * 2 * valPixels

    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1,0, transX], [0,1, transY]])
    img_result = cv2.warpAffine(img, transMat, (IMG_WIDTH, IMG_HEIGHT))
    return img_result, angle_result    


""" Training and Validation """


# Generate batch data with data augmentation
def generate_batch(X, y, batch_size=10):
    n = len(y)
    X_result = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, COLOR_IMG_DEPTH))
    y_result = np.zeros(batch_size)
    while 1:
        counter = 0
        while counter <= batch_size - 1:
            idx = np.random.randint(n - 1)
            steering_angle = y[idx]
            img = X[idx]
            if np.random.rand() > 0.5: # 50% change to see the right angle
                img = cv2.flip(img, 1)
                steering_angle = -steering_angle
            
            # Randomly transpose image and steering angle
            img, steering_angle = jiggle_data(img, steering_angle)
            
            # Randomly adjust brightness
            img = get_random_brightness(img)
                        
            X_result[counter] = get_random_brightness(img)
            y_result[counter] = steering_angle
            
            counter += 1
        yield X_result, y_result   


def generate_train_data(X, y):

    # Flip left and right
    X_flipped, y_flipped = batch_flip_left_right(X, y)
    
    # Adjust brightness randomly
    X_random_brightness = batch_generate_random_brightness(X)
    
    # Add randomly generated shadows
    X_random_shadows = batch_generate_random_shadow(X)

    X_generated = np.concatenate([X_flipped, X, X_random_brightness, X_random_shadows])
    y_generated = np.concatenate([y_flipped, y, y, y])
    return X_generated, y_generated
    

def train_generator(X, y, samples_per_epoch, batch_size):
    n = int(samples_per_epoch/batch_size)
    X_generated, y_generated = generate_train_data(X, y)
    while 1:
        X_shuffled, y_shuffled = shuffle(X_generated, y_generated)
        for i in range(n):            
            start, end = i * batch_size, (i + 1) * batch_size
            yield(X_shuffled[start:end], y_shuffled[start:end])


def validation_generator(X, y, samples_per_epoch, batch_size):
    n = int(samples_per_epoch/batch_size)
    while 1:
        for i in range(n):
            start, end = i * batch_size, (i + 1) * batch_size
            yield(X[start:end], y[start:end])


def train_model():
    iteration = 2
    batch_size = 50
    nb_epoch = 200 #1000
    train_samples_per_epoch = len(X_train) - len(X_train) % batch_size
    val_samples_per_epoch = len(X_val) - len(X_val) % batch_size
    for i in range(iteration):
        print('Iteration {}'.format(i + 1))    
        train_generator = generate_batch(X_train, y_train, batch_size)
        valid_generator = validation_generator(X_val, y_val, val_samples_per_epoch, batch_size)
        _model.fit_generator(train_generator, \
                            samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch, \
                            validation_data=valid_generator, \
                            nb_val_samples=train_samples_per_epoch)
          

""" Keras Helper Methods """


def save_model_to_disk():
    with open('model.json', 'w') as f:
        json.dump(_model.to_json(), f)
    _model.save_weights('model.h5', True)


def load_model_from_disk():
    json_path = 'model.json'
    with open(json_path, 'r') as f:
        _model = model_from_json(json.load(f))
        _model.compile('adam', 'mse')
    weights_path = json_path.replace('json', 'h5')
    _model.load_weights(weights_path)


""" Main Entry Point """    
if __name__ == '__main__':
    # Step1: Load images and csv file (driving_log)
    _driving_log = load_data_from_log()
    
    # Step2: Create training data from the driving_log
    X_data, y_data = load_training_data()
    X_data, y_data = shuffle(X_data, y_data)    
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=0)
    
    print('original data set: {}'.format(X_data.shape))
    print('training data set: {}'.format(X_train.shape))
    print('validation data set: {}'.format(X_val.shape))

    # Step3: Train
    _model = nvidia_model()
    train_model()
    
    # Step4: Generate the h5 file using a trained model
    save_model_to_disk()
