from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
Y_train = pd.get_dummies(Y_train).values
Y_test = pd.get_dummies(Y_test).values

X_train=X_train.astype(float)/255
X_test=X_test.astype(float)/255
X_train = X_train.reshape(-1,28,28,1)
X_test =  X_test.reshape(-1,28,28,1)

np.shape(X_test)

model = Sequential()
model.add(Input(shape=(28,28,1)))
model.add(Conv2D(64, (3,3),  activation='relu' ))
model.add(Conv2D(64, (3,3),  activation='relu' ))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# model = Sequential(
#     [
#         Input(shape=(28,28,1)),
#         Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         MaxPooling2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(10, activation="softmax"),
#     ]
# )
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

mf = model.fit(X_train, Y_train, batch_size=32, epochs=1, validation_data=(X_test, Y_test), verbose=2)