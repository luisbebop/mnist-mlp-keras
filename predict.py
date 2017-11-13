import numpy as np
import keras
from keras.models import load_model
from keras.datasets import mnist

# load model
model = load_model('mnist.h5')

# load mnist training dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255

# predict
out = model.predict(x_train[0:1])
print(np.argmax(out, axis=1))
