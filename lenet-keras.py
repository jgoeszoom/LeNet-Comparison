from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random
import time 

def index_at(arr):
    count = 0;
    for i in arr:
        if i == 1:
            return count
        else:
            count+=1
    return None

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28,28

(x_train,y_train),(x_test,y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test = x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape = (1, img_rows, img_cols)
else: 
    x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows,img_cols,1)
    input_shape = (img_rows,img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

model = Sequential()
model.add(Conv2D(5,(5,5),activation = 'relu',input_shape =input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(12,(5,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print('Time taken without training', time1, 'seconds in process time')
model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data = (x_test,y_test))

score = model.evaluate(x_test,y_test,verbose = 0)
print('Test loss:',score[0])
print('Test accuracy:', score[0])
print('Time taken with training', time, 'seconds in process time')

#print(y_train[0])
#print(index_at(y_train[0]))
#fig = plt.figure()
#for num in range(9):
#   plt.subplot(3,3,num+1)
#   plt.tight_layout()
#   plt.title('Number: {}'.format(index_at(y_train[num])))
#   plt.imshow(x_train[num].reshape(28,28))
#   #plt.show()
#   plt.xticks([])
#   plt.yticks([])
#plt.show()

start = time.clock()
for i in range(0, len(y_test)): 
	probs = model.predict(x_test[np.newaxis, i])
	prediction = probs.argmax(axis=1)
end = time.clock()
time = end-start
printf("Elapsed time(s) ", time)
