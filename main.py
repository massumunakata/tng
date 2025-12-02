import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=(256,)),
    keras.layers.Dense(2, activation="softmax")
])

model.save("./models/dataType0-epoch45-size256.h5")
