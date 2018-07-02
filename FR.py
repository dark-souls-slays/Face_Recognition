import numpy as np
from PIL import Image
import glob
import os, os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.optimizers import RMSprop

def inputs(ID):
    imgs = []
    dataset = []
    y = []
    check = 0
    size = 128, 128
    path = "/Users/ClaudiaEspinoza/Desktop/Face Recognition System/"+ str(ID)
    for f in os.listdir(path):
        if f == '.DS_Store':
            continue
        im = Image.open(os.path.join(path,f))
        im = im.convert('L')
        im = im.resize((size), Image.ANTIALIAS)
        #im.load()
        data = np.array(im)
        im.close()
        data = np.reshape(data,16384)
        dataset.append(data)
        y.append(int(f[1:3]))

    dataset = np.array(dataset)
    y = np.array(y)
    #CHECK THAT THIS PARTICULAR IMAGE CORRESPONDS WITH ITS Y LABEL
    """
    m = Image.fromarray(dataset[499].reshape(128,128), 'L')
    m.save("this.jpg")
    print(y[499])
    """
    return (dataset,y)

trainingset, labels = inputs("training_data")
testset, labelsTest = inputs("test_data")
epochs = 20
#Preprocess
trainingset = trainingset.astype('float32')
testset = testset.astype('float32')
trainingset /= 255.0
testset /= 255.0

# one hot encoding
y_train = to_categorical(labels)
y_test = to_categorical(labelsTest)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(16384,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(51, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(trainingset, y_train,
                    batch_size=5,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(testset, y_test))
score = model.evaluate(testset, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (500,128, 128), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Flatten to 1D
classifier.add(Flatten())
#Add Layers
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.fit_generator(trainingset,steps_per_epoch = 8000,epochs = 25,validation_data = testset,validation_steps = 2000)
classifier.fit(trainingset / 255.0, # pixel value normalization
			to_categorical(labels), # one-hot encoding
			batch_size=32, epochs=15)
"""
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(500,1,128 ,128))) # the 1st layer requires input_shape
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Dropout(0.2))
#model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
				metrics=['accuracy'])

model.fit(trainingset / 255.0, # pixel value normalization
			to_categorical(labels), # one-hot encoding
			batch_size=32, epochs=15)
"""
"""
Testing
scores = model.evaluate(testset / 255.0, to_categorical(labelsTest))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
"""
