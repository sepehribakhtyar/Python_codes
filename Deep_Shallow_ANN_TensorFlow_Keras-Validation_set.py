# Shallow and deep fed forward neural networks by Keras package
# In Xtrain and Xtest matrices, objects are in rows and independent variables are in columns.
# Xtrain is independent vriable(s) matrix for train set, Xvalidation is independent variables matrix for validation set and Xtest is independent variabl(s) matrix for test set.
# Ytrain is dependent variable vector (a clumn vector) for train set,Yvalidation is a dependent variable vector (a clumn vector) for validation set and Ytest is dependent variable vector (a column vector) for test set.
# Ytrain, Yvalidation and Ytest should be defined as matrices that have two dimentions (Object numbers * 1).
# In input shape, the first dimension should be 1 and the second should be equal to the number of indepentent variables in Xtrain matrix that has been put in columns.
# Variables_number is the number of variables in Xtrain matrix that have been put in columns. In other words, Variables_number=the number of columns in Xtrain.
##############################################################################################################################################################################################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
Variables_number=
initializer_1 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(1,Variables_number=)))
model.add(tf.keras.layers.Dense(units=10, activation='relu',kernel_initializer=initializer_1, bias_initializer='ones', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5),kernel_constraint=None, bias_constraint=None)
model.add(tf.keras.layers.Dropout(rate=))
model.add(tf.keras.layers.Dense(units=, activation='relu',kernel_initializer=initializer_1, bias_initializer='ones', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5),kernel_constraint=None, bias_constraint=None)
model.add(tf.keras.layers.Dropout(rate=))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),loss=tf.keras.losses.MeanAbsoluteError(),  metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.fit(x=Xtrain, y=Ytrain, batch_size=1, epochs=1000, validation_split=0.0, validation_data=(x_val=Xvalidation, y_val=Yvalidation), shuffle=True)
model.predict(x=Xtest)
model.summary()

 

