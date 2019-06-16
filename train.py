import csv

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
import cv2

images = []
steerings = []
throttles = []
brakes = []
speeds = []
for line in lines[1:]:
    image_path = 'data/' + line[0]
    image = cv2.imread(image_path)
    images.append(image)
    steerings.append(float(line[3]))
    throttles.append(float(line[4]))
    brakes.append(float(line[5]))
    speeds.append(float(line[6]))

import numpy as np

X_train = np.array(images)
y_train = np.array(steerings)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout

model = Sequential()
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(60,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model.h5')
               