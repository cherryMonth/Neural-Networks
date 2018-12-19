from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

"""
数据集 https://www.kaggle.com/tongpython/nattawut-5920421014-cat-vs-dog-dl/data
"""

model = Sequential()
model.add(
    Conv2D(6, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))
sgd = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
train_dategen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_dategen.flow_from_directory('input/training_set', target_size=(64, 64), batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('input/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

model.fit_generator(training_set, steps_per_epoch=1000, epochs=5, validation_data=test_set, validation_steps=100)

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('input/test_set/dogs/dog.4001.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
