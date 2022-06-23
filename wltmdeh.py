
import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.layers import Dense, Flatten, Activation
from keras.callbacks import Callback

import numpy as np
import matplotlib.pyplot as plt
from ClassificationAndLabeling_basicCL import obs_Xdata, obs_Ylabel


imagedata = np.array(obs_Xdata)
imagelabel = np.array(obs_Ylabel)

train_images, test_images, train_labels, test_labels = train_test_split(imagedata, imagelabel, test_size=0.2)


train_images = train_images.astype('float32')/255.0
test_images = test_images.astype('float32')/ 255.0


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape = (128,128,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add (Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add( Dense(64, activation='relu'))
model.add( Dense(6, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=15  , batch_size=4)

plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.show()

plt.title('Loss')
plt.plot(history.history['loss'])
plt.show()

loss_and_accuracy = model.evaluate(test_images, test_labels)
print('accuray=' + str(loss_and_accuracy[1])) #at accuray= 0.20
print('loss= ' +  str(loss_and_accuracy[0])) # loss= 1.60

model.save('obs_model.h5')

































