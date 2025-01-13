

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, Input
import numpy as np
import tf2onnx
import onnx 

# Input
model = Sequential()  
model.add(Input(shape=(128, 128, 16), batch_size=1))


# Conv
model.add(Conv2D(4, kernel_size=3, strides=2, padding="SAME", activation='relu'))
model.add(Conv2D(8, kernel_size=3, strides=2, padding="SAME", activation='relu'))
model.add(Conv2D(16, kernel_size=3, strides=2, padding="SAME", activation='relu'))
model.add(Conv2D(32, kernel_size=3, strides=2, padding="SAME", activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=2, padding="SAME", activation='relu'))
model.add(Conv2D(128, kernel_size=3, strides=2, padding="SAME", activation='relu'))


# Dense
model.add(Flatten())
model.add(Dense(units = 2056, activation = 'relu'))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 11, activation = 'softmax'))



# parameters = np.sum([np.prod(v._shape) for v in model.trainable_variables])
# print('-'*50)
# print(f'parameters: {parameters:,}')


onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
onnx.save(onnx_model, "./acnn2_tensor13.onnx") 