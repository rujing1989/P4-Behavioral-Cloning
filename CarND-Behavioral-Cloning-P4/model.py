import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

correction = 0.25

for line in lines:
    # images of multipy camera
    image_path = './data/data/IMG/'
    
    measurement = float(line[3])
    # image of center
    image_center = cv2.imread(image_path + line[0].split('/')[-1])
    images.append(image_center)
    measurements.append(measurement)
    # flipped 
    images.append(cv2.flip(image_center, 1))
    measurements.append(-measurement)
    
    # image of left
    image_left = cv2.imread(image_path + line[1].split('/')[-1])
    images.append(image_left)
    measurements.append(measurement + correction)
    # flipped
    images.append(cv2.flip(image_left, 1))
    measurements.append(-(measurement + correction)) 
    
    # image of right
    image_right = cv2.imread(image_path + line[2].split('/')[-1])
    images.append(image_right)
    measurements.append(measurement - correction)
    # flipped
    images.append(cv2.flip(image_right, 1))
    measurements.append(-(measurement - correction))
    

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
#
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))

#model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
# test
#model.add(Dropout(0.2))
#model.add(ELU())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
print(model.summary())

model.save('model.h5')